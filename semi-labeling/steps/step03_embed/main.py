#!/usr/bin/env python3
"""Step 03 — DINOv2 Embedding.

Embeds resemi crop_views once with DINOv2 and caches L2-normalized float32
vectors. This is the gate for step04-09 (they all read embeddings from here).
Run after step02. Use --resume to continue an interrupted run, --force to make
a fresh embedding run (e.g. after changing model).

Inputs:  crop_views in resemi.sqlite3 + crop PNGs
Outputs: embedding_runs, crop_embeddings, skipped_crop_embeddings
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from shared.runtime import bootstrap

bootstrap.ensure_repo_root_on_path()

from shared.db.embedder import DinoV2Embedder  # noqa: E402
from shared.db.embedding_cache import (  # noqa: E402
    CropEmbeddingRow,
    existing_embedding_keys,
    existing_skip_keys,
    latest_embedding_run,
    read_crop_views,
    update_embedding_counts,
)
from shared.runtime.paths import default_resemi_db  # noqa: E402
from shared.db.schema import connect_output, utc_now  # noqa: E402
from shared.runtime.prefetch import ordered_prefetch  # noqa: E402


DEFAULT_MODEL_NAME = "facebook/dinov2-small"


def parse_result_ids(raw: list[str]) -> list[int]:
    result: list[int] = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                result.append(int(part))
    return result


def open_crop(path: str) -> Image.Image:
    crop_path = Path(path).expanduser()
    if not crop_path.is_file():
        raise FileNotFoundError(f"Crop not found: {crop_path}")
    with Image.open(crop_path) as image:
        rgb = image.convert("RGB")
        rgb.load()
        return rgb


def skip_reason(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "crop_not_found"
    if isinstance(exc, UnidentifiedImageError):
        return "decode_error"
    return "embed_error"


def create_embedding_run(
    conn: sqlite3.Connection,
    *,
    embedding_run_id: str,
    run_id: str,
    model_name: str,
    model_version: str,
    embedding_type: str,
    device: str,
    view_name: str,
    dim: int,
    total_crops: int,
    options: dict,
) -> None:
    conn.execute(
        """
        INSERT INTO embedding_runs (
            embedding_run_id, run_id, created_at_utc, model_name, dim, options_json,
            model_version, embedding_type, device, view_name, total_crops,
            embedded_count, skipped_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
        """,
        (
            embedding_run_id,
            run_id,
            utc_now(),
            model_name,
            int(dim),
            json.dumps(options, ensure_ascii=False, sort_keys=True),
            model_version,
            embedding_type,
            device,
            view_name,
            int(total_crops),
        ),
    )
    conn.commit()


def insert_embeddings(
    conn: sqlite3.Connection,
    *,
    embedding_run_id: str,
    dim: int,
    rows: list[tuple[CropEmbeddingRow, np.ndarray]],
) -> int:
    if not rows:
        return 0
    now = utc_now()
    conn.executemany(
        """
        INSERT OR REPLACE INTO crop_embeddings (
            embedding_run_id, result_id, view_name, embedding_blob,
            crop_view_id, crop_path, dim, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                embedding_run_id,
                row.result_id,
                row.view_name,
                np.asarray(vector, dtype="<f4").tobytes(),
                row.crop_view_id,
                row.crop_path,
                int(dim),
                now,
            )
            for row, vector in rows
        ],
    )
    conn.commit()
    return len(rows)


def insert_skips(conn: sqlite3.Connection, *, embedding_run_id: str, rows: list[tuple[CropEmbeddingRow, str, str]]) -> int:
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO skipped_crop_embeddings (
            embedding_run_id, result_id, view_name, crop_path, reason, error_message
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (embedding_run_id, row.result_id, row.view_name, row.crop_path, reason, message)
            for row, reason, message in rows
        ],
    )
    conn.commit()
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed resemi crop_views once and cache L2-normalized float32 vectors.")
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", required=True, help="Resemi run_id containing crop_views.")
    parser.add_argument("--view-name", default="tight", help="Crop view to embed: tight, pad10, pad25, context, openclip_crop.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="DINOv2 HF id or local folder.")
    parser.add_argument("--model-version", default="", help="Optional audited model version/checkpoint tag.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Background threads to decode crop PNGs while the GPU embeds. 0 = sequential (default).")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--result-id", action="append", default=[], help="Specific result_id(s), comma-separated or repeated.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Create a new embedding run even if a matching run exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate crop rows without loading the embedding model.")
    parser.add_argument("--log-every", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")
    result_ids = parse_result_ids(list(args.result_id or []))

    conn = connect_output(db_path)
    try:
        crop_rows = read_crop_views(conn, run_id=str(args.run_id), view_name=str(args.view_name), result_ids=result_ids or None, limit=int(args.limit))
        if not crop_rows:
            raise RuntimeError("No crop_views matched the requested run/view filters.")
        missing = [row for row in crop_rows if not Path(row.crop_path).expanduser().is_file()]
        print(f"matched_crops={len(crop_rows)} missing_files={len(missing)} view_name={args.view_name}", flush=True)
        if bool(args.dry_run):
            return 0

        existing = latest_embedding_run(conn, model_name=str(args.model_name), view_name=str(args.view_name))
        options = {
            "db": str(db_path),
            "run_id": str(args.run_id),
            "view_name": str(args.view_name),
            "model_name": str(args.model_name),
            "model_version": str(args.model_version),
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "limit": int(args.limit),
            "result_ids": result_ids,
            "resume": bool(args.resume),
            "force": bool(args.force),
        }

        if existing is not None and not bool(args.resume) and not bool(args.force):
            print(
                f"Existing embedding run found: embedding_run_id={existing['embedding_run_id']}. Use --resume or --force.",
                flush=True,
            )
            return 2

        if bool(args.resume) and existing is not None and not bool(args.force):
            embedding_run_id = str(existing["embedding_run_id"])
            dim = int(existing["dim"])
            done = existing_embedding_keys(conn, embedding_run_id=embedding_run_id) | existing_skip_keys(conn, embedding_run_id=embedding_run_id)
            crop_rows = [row for row in crop_rows if (row.result_id, row.view_name) not in done]
            print(f"resume embedding_run_id={embedding_run_id} remaining={len(crop_rows)} done={len(done)}", flush=True)
            if not crop_rows:
                embedded, skipped = update_embedding_counts(conn, embedding_run_id=embedding_run_id)
                print(f"Done: embedded={embedded} skipped={skipped}", flush=True)
                return 0
            embedder = DinoV2Embedder(model_name=str(args.model_name), device=str(args.device))
        else:
            embedder = DinoV2Embedder(model_name=str(args.model_name), device=str(args.device))
            dim = int(embedder.dim)
            embedding_run_id = f"emb_{uuid.uuid4().hex[:12]}"
            create_embedding_run(
                conn,
                embedding_run_id=embedding_run_id,
                run_id=str(args.run_id),
                model_name=str(args.model_name),
                model_version=str(args.model_version),
                embedding_type="dinov2_crop",
                device=str(embedder.device),
                view_name=str(args.view_name),
                dim=dim,
                total_crops=len(crop_rows),
                options=options,
            )
            print(f"embedding_run_id={embedding_run_id} crops={len(crop_rows)} model={args.model_name}", flush=True)

        processed_since_log = 0
        batch_size = max(1, int(args.batch_size))
        num_workers = max(0, int(args.num_workers or 0))

        # Prefetch crop decode (CPU/disk bound) on worker threads while the GPU
        # embeds the previous batch. Order preserved; one bad crop -> skip, not
        # a crash. Buffers accumulate up to batch_size before each GPU call.
        ok_rows: list[CropEmbeddingRow] = []
        images: list[Image.Image] = []

        def _flush_embed(dim_value: int) -> int:
            if not images:
                return dim_value
            try:
                vectors = embedder.embed(images, batch_size=batch_size)
            finally:
                for image in images:
                    image.close()
            if vectors.shape[0] != len(ok_rows):
                raise RuntimeError(f"Embedding row mismatch: got {vectors.shape[0]}, expected {len(ok_rows)}")
            if dim_value <= 0 and vectors.ndim == 2 and vectors.shape[1] > 0:
                dim_value = int(vectors.shape[1])
                conn.execute("UPDATE embedding_runs SET dim = ? WHERE embedding_run_id = ?", (dim_value, embedding_run_id))
                conn.commit()
            insert_embeddings(conn, embedding_run_id=embedding_run_id, dim=dim_value, rows=list(zip(ok_rows, vectors, strict=True)))
            ok_rows.clear()
            images.clear()
            return dim_value

        for row, image, exc in ordered_prefetch(
            crop_rows, lambda r: open_crop(r.crop_path),
            num_workers=num_workers, max_inflight=max(batch_size, num_workers * 2),
        ):
            if exc is not None:
                insert_skips(conn, embedding_run_id=embedding_run_id, rows=[(row, skip_reason(exc), str(exc))])
                processed_since_log += 1
            else:
                ok_rows.append(row)
                images.append(image)
                if len(images) >= batch_size:
                    dim = _flush_embed(dim)
                    processed_since_log += batch_size

            if int(args.log_every) > 0 and processed_since_log >= int(args.log_every):
                embedded, skipped = update_embedding_counts(conn, embedding_run_id=embedding_run_id)
                print(f"[embed] embedded={embedded} skipped={skipped} total={len(crop_rows)}", flush=True)
                processed_since_log = 0

        dim = _flush_embed(dim)

        embedded, skipped = update_embedding_counts(conn, embedding_run_id=embedding_run_id)
        print(f"Done: embedding_run_id={embedding_run_id} embedded={embedded} skipped={skipped} db={db_path}", flush=True)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
