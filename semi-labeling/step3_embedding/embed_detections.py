#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = resolve_repo_root()
LAB_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedder import DinoV2Embedder
from output_store import (
    bulk_insert_embeddings,
    bulk_insert_skipped,
    connect_output,
    connect_readonly,
    embedded_result_ids,
    ensure_schema,
    insert_run_metadata,
    latest_matching_run,
    skipped_result_ids,
    update_run_counts,
)

DEFAULT_MODEL_NAME = "facebook/dinov2-giant"
DEFAULT_LABELS = "crack,spall,mold"


@dataclass(frozen=True)
class SourceDetection:
    result_id: int
    image_rel_path: str
    image_path: str
    source_input_dir: str
    predicted_label: str
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int

    @property
    def width(self) -> float:
        return max(0.0, float(self.x2) - float(self.x1))

    @property
    def height(self) -> float:
        return max(0.0, float(self.y2) - float(self.y1))


def default_source_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step2_sematic" / "damage_scan.sqlite3"


def default_image_root() -> Path:
    return LAB_ROOT / "data" / "HinhAnh"


def default_output_db() -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "step3_embedding" / "embeddings.sqlite3"


def parse_labels(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def resolve_semantic_run_id(conn: sqlite3.Connection, requested: str) -> str:
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        row = conn.execute(
            "SELECT semantic_run_id FROM openclip_semantic_runs WHERE semantic_run_id = ?",
            (raw,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Semantic run not found: {raw}")
        return raw
    row = conn.execute(
        "SELECT semantic_run_id FROM openclip_semantic_runs ORDER BY created_at_utc DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No semantic run found in source DB.")
    return str(row["semantic_run_id"])


def read_detections(
    conn: sqlite3.Connection,
    *,
    semantic_run_id: str,
    min_confidence_pct: float,
    labels: list[str],
    limit: int,
) -> list[SourceDetection]:
    clauses = ["res.semantic_run_id = ?", "res.status = 'ok'", "res.predicted_probability_pct >= ?"]
    params: list[Any] = [semantic_run_id, float(min_confidence_pct)]
    if labels:
        placeholders = ", ".join("?" for _ in labels)
        clauses.append(f"res.predicted_label IN ({placeholders})")
        params.extend(labels)
    sql = f"""
        SELECT res.result_id, res.image_rel_path, res.image_path, src.input_dir AS source_input_dir,
               res.predicted_label, res.x1, res.y1, res.x2, res.y2,
               img.width AS image_width, img.height AS image_height
        FROM openclip_semantic_results res
        JOIN images img ON img.image_id = res.image_id
        JOIN runs src ON src.run_id = res.source_run_id
        WHERE {' AND '.join(clauses)}
        ORDER BY res.result_id
    """
    if int(limit) > 0:
        sql = f"{sql} LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    return [SourceDetection(**dict(row)) for row in rows]


def resolve_image_path(detection: SourceDetection, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(detection.image_rel_path or "").strip()
    stored_path = str(detection.image_path or "").strip()
    source_input_dir = Path(str(detection.source_input_dir or "")).expanduser()

    if image_root is not None:
        root = image_root.expanduser().resolve()
        if rel_path:
            candidates.append(root / rel_path)
        if stored_path:
            candidates.append(root / Path(stored_path).name)

    if stored_path:
        stored = Path(stored_path).expanduser()
        candidates.append(stored if stored.is_absolute() else source_input_dir / stored_path)
    if rel_path:
        candidates.append(source_input_dir / rel_path)
    if stored_path:
        candidates.append(source_input_dir / Path(stored_path).name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    if image_root is not None:
        return (image_root.expanduser().resolve() / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()


def crop_with_padding(detection: SourceDetection, image_root: Path | None, *, padding_ratio: float) -> Image.Image:
    image_path = resolve_image_path(detection, image_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            pad_x = detection.width * float(padding_ratio)
            pad_y = detection.height * float(padding_ratio)
            x1 = max(0, int(math.floor(float(detection.x1) - pad_x)))
            y1 = max(0, int(math.floor(float(detection.y1) - pad_y)))
            x2 = min(width, int(math.ceil(float(detection.x2) + pad_x)))
            y2 = min(height, int(math.ceil(float(detection.y2) + pad_y)))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid crop box result_id={detection.result_id}: {(x1, y1, x2, y2)}")
            crop = rgb.crop((x1, y1, x2, y2))
            crop.load()
            return crop
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Cannot decode image: {image_path}") from exc


def skip_reason(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "image_not_found"
    if isinstance(exc, ValueError):
        return "invalid_bbox"
    return "decode_error"


def chunks(items: list[SourceDetection], size: int) -> Iterable[list[SourceDetection]]:
    effective_size = max(1, int(size))
    for start in range(0, len(items), effective_size):
        yield items[start : start + effective_size]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed Step 2 detections once with DINOv2 and persist vectors to SQLite.")
    parser.add_argument("--source-db", default=str(default_source_db()), help="Source Step 2 damage_scan.sqlite3.")
    parser.add_argument("--semantic-run-id", default="latest", help="Semantic run id, or latest.")
    parser.add_argument("--image-root", default=str(default_image_root()), help="Image root override, usually /path/to/data/HinhAnh.")
    parser.add_argument("--output-db", default=str(default_output_db()), help="Output embeddings SQLite path.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="DINOv2 HF model id or local model folder.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--min-confidence-pct", type=float, default=0.0)
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Comma-separated labels. Empty = all.")
    parser.add_argument("--limit", type=int, default=0, help="Debug mode: embed first N matching detections. 0 = all.")
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--resume", action="store_true", help="Resume the latest matching embedding run.")
    parser.add_argument("--force", action="store_true", help="Create a new run even if model + semantic_run_id already exists.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source_db = Path(args.source_db).expanduser().resolve()
    if not source_db.is_file():
        raise FileNotFoundError(f"Source DB not found: {source_db}")
    image_root = Path(args.image_root).expanduser().resolve() if str(args.image_root or "").strip() else None
    output_db = Path(args.output_db).expanduser().resolve()
    labels = parse_labels(str(args.labels))

    source_conn = connect_readonly(source_db)
    try:
        semantic_run_id = resolve_semantic_run_id(source_conn, str(args.semantic_run_id))
        detections = read_detections(
            source_conn,
            semantic_run_id=semantic_run_id,
            min_confidence_pct=float(args.min_confidence_pct),
            labels=labels,
            limit=int(args.limit),
        )
    finally:
        source_conn.close()
    if not detections:
        raise RuntimeError("No detections matched the current filters.")

    options = {
        "source_db": str(source_db),
        "semantic_run_id": str(args.semantic_run_id),
        "resolved_semantic_run_id": semantic_run_id,
        "image_root": "" if image_root is None else str(image_root),
        "output_db": str(output_db),
        "model_name": str(args.model_name),
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "padding_ratio": float(args.padding_ratio),
        "min_confidence_pct": float(args.min_confidence_pct),
        "labels": labels,
        "limit": int(args.limit),
        "log_every": int(args.log_every),
        "resume": bool(args.resume),
        "force": bool(args.force),
    }

    out_conn = connect_output(output_db)
    try:
        ensure_schema(out_conn)
        existing = latest_matching_run(out_conn, model_name=str(args.model_name), semantic_run_id=semantic_run_id)
        if existing is not None and not bool(args.resume) and not bool(args.force):
            print(
                "Existing embedding run found for the same model_name + semantic_run_id. "
                f"embedding_run_id={existing['embedding_run_id']}. Use --resume or --force.",
                flush=True,
            )
            return 2

        embedder: DinoV2Embedder
        run_dim = 0
        if bool(args.resume) and existing is not None and not bool(args.force):
            embedding_run_id = str(existing["embedding_run_id"])
            run_dim = int(existing["dim"])
            done_ids = embedded_result_ids(out_conn, embedding_run_id=embedding_run_id) | skipped_result_ids(
                out_conn, embedding_run_id=embedding_run_id
            )
            detections = [item for item in detections if int(item.result_id) not in done_ids]
            print(f"resume embedding_run_id={embedding_run_id} remaining={len(detections)} done={len(done_ids)}", flush=True)
            if not detections:
                embedded, skipped = update_run_counts(out_conn, embedding_run_id=embedding_run_id)
                print(f"Done: {embedded} embedded, {skipped} skipped db={output_db}", flush=True)
                return 0
            embedder = DinoV2Embedder(model_name=str(args.model_name), device=str(args.device))
        else:
            embedder = DinoV2Embedder(model_name=str(args.model_name), device=str(args.device))
            run_dim = int(embedder.dim)
            embedding_run_id = uuid.uuid4().hex
            insert_run_metadata(
                out_conn,
                embedding_run_id=embedding_run_id,
                source_db_path=source_db,
                source_semantic_run_id=semantic_run_id,
                model_name=str(args.model_name),
                dim=int(embedder.dim),
                device=str(embedder.device),
                padding_ratio=float(args.padding_ratio),
                total_detections=len(detections),
                options=options,
            )
            print(
                f"embedding_run_id={embedding_run_id} detections={len(detections)} semantic_run_id={semantic_run_id} model={args.model_name}",
                flush=True,
            )

        if not detections:
            embedded, skipped = update_run_counts(out_conn, embedding_run_id=embedding_run_id)
            print(f"Done: {embedded} embedded, {skipped} skipped db={output_db}", flush=True)
            return 0

        processed_since_log = 0
        for batch in chunks(detections, int(args.batch_size)):
            batch_ok: list[SourceDetection] = []
            crops: list[Image.Image] = []
            skipped_rows: list[tuple[str, int, str]] = []
            for detection in batch:
                try:
                    crop = crop_with_padding(detection, image_root, padding_ratio=float(args.padding_ratio))
                    crops.append(crop)
                    batch_ok.append(detection)
                except Exception as exc:
                    reason = skip_reason(exc)
                    skipped_rows.append((embedding_run_id, int(detection.result_id), reason))
                    print(f"[skip] result_id={detection.result_id} reason={reason} error={exc}", flush=True)

            if skipped_rows:
                bulk_insert_skipped(out_conn, skipped_rows)
            if not crops:
                update_run_counts(out_conn, embedding_run_id=embedding_run_id)
                continue

            try:
                embeddings = embedder.embed(crops, batch_size=int(args.batch_size))
            finally:
                for crop in crops:
                    crop.close()

            if embeddings.shape[0] != len(batch_ok):
                raise RuntimeError(f"Embedding row mismatch: got {embeddings.shape[0]}, expected {len(batch_ok)}")
            if run_dim <= 0 and embeddings.ndim == 2 and embeddings.shape[1] > 0:
                run_dim = int(embeddings.shape[1])
                out_conn.execute("UPDATE embedding_runs SET dim = ? WHERE embedding_run_id = ?", (int(embeddings.shape[1]), embedding_run_id))
                out_conn.commit()

            rows = []
            for detection, emb in zip(batch_ok, embeddings, strict=True):
                arr = np.asarray(emb, dtype="<f4")
                rows.append(
                    (
                        embedding_run_id,
                        int(detection.result_id),
                        detection.image_rel_path,
                        detection.predicted_label,
                        arr.tobytes(),
                    )
                )
            bulk_insert_embeddings(out_conn, rows)
            embedded, skipped = update_run_counts(out_conn, embedding_run_id=embedding_run_id)
            processed_since_log += len(rows) + len(skipped_rows)
            if int(args.log_every) > 0 and processed_since_log >= int(args.log_every):
                print(f"[embed] embedded={embedded} skipped={skipped} total={len(detections)}", flush=True)
                processed_since_log = 0

        embedded, skipped = update_run_counts(out_conn, embedding_run_id=embedding_run_id)
        print(f"Done: {embedded} embedded, {skipped} skipped db={output_db}", flush=True)
        return 0
    finally:
        out_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
