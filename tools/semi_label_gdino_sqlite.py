from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from object_detection.dino.client import get_dino_service
from object_detection.dino.engine import default_gdino_checkpoint
from tools.gdino_detect_folder import (
    _IMAGE_EXTS,
    _iter_images,
    _max_dim_fast,
    _rows_from_detections,
    _write_box_csv,
    _write_error_text,
    _write_overlay,
)


DEFAULT_PROMPT_GROUPS: list[tuple[str, str]] = [
    (
        "line",
        "line, irregular line, thin line, scratch, crack, fracture, dark line, random line on surface",
    ),
    (
        "damage",
        "damage, broken, rough, hole, chipped, irregular surface, destroyed surface",
    ),
    (
        "dirty",
        "dirty, stain, spot, dark area, patch, discoloration, abnormal color",
    ),
]


@dataclass(frozen=True)
class PromptGroup:
    group_id: int
    name: str
    slug: str
    prompt_text: str
    queries: tuple[str, ...]


def _slugify(text: str) -> str:
    parts: list[str] = []
    prev_sep = False
    for ch in str(text or "").strip().lower():
        if ch.isalnum():
            parts.append(ch)
            prev_sep = False
        elif not prev_sep:
            parts.append("-")
            prev_sep = True
    slug = "".join(parts).strip("-")
    return slug or "group"


def _parse_prompt_group(value: str, index: int) -> PromptGroup:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Prompt group must not be empty.")
    if "=" in raw:
        name, prompt_text = raw.split("=", 1)
    elif "::" in raw:
        name, prompt_text = raw.split("::", 1)
    else:
        name = f"group_{index:02d}"
        prompt_text = raw
    name = str(name).strip() or f"group_{index:02d}"
    queries = tuple(q.strip() for q in str(prompt_text).split(",") if q.strip())
    if not queries:
        raise ValueError(f"Prompt group '{name}' has no valid queries.")
    return PromptGroup(
        group_id=index,
        name=name,
        slug=f"{index:02d}-{_slugify(name)}",
        prompt_text=", ".join(queries),
        queries=queries,
    )


def _load_prompt_groups(args: argparse.Namespace) -> list[PromptGroup]:
    raw_values = list(args.prompt_group or [])
    if not raw_values:
        raw_values = [f"{name}={prompt}" for name, prompt in DEFAULT_PROMPT_GROUPS]
    return [_parse_prompt_group(value, idx) for idx, value in enumerate(raw_values, start=1)]


def _parse_request_names(raw: str) -> list[str]:
    names = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for item in names:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out or ["dino", "groundingdino"]


def _combined_queries(prompt_groups: list[PromptGroup]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in prompt_groups:
        for query in group.queries:
            key = query.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(query)
    return merged


def _match_prompt_groups(label: str, prompt_groups: list[PromptGroup]) -> list[int]:
    text = str(label or "").strip().lower()
    if not text:
        return []
    matched: list[int] = []
    for group in prompt_groups:
        for query in group.queries:
            q = query.strip().lower()
            if q and q in text:
                matched.append(int(group.group_id))
                break
    return matched


def _rel_image_key(input_dir: Path, image_path: Path) -> str:
    return image_path.relative_to(input_dir).as_posix()


def _artifact_dir(output_dir: Path, *, prompt_group: PromptGroup, input_dir: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(input_dir)
    return output_dir / "artifacts" / prompt_group.slug / rel.parent / rel.name


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS run_info (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            input_dir TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            checkpoint TEXT NOT NULL,
            device TEXT NOT NULL,
            box_threshold REAL NOT NULL,
            text_threshold REAL NOT NULL,
            max_dets INTEGER NOT NULL,
            tiled_threshold INTEGER NOT NULL,
            recursive_find INTEGER NOT NULL,
            recursive_max_depth INTEGER NOT NULL,
            recursive_min_box_px INTEGER NOT NULL,
            image_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS request_aliases (
            run_id TEXT NOT NULL,
            request_name TEXT NOT NULL,
            engine_name TEXT NOT NULL,
            notes TEXT NOT NULL,
            PRIMARY KEY (run_id, request_name)
        );

        CREATE TABLE IF NOT EXISTS prompt_groups (
            run_id TEXT NOT NULL,
            prompt_group_id INTEGER NOT NULL,
            prompt_group_name TEXT NOT NULL,
            prompt_group_slug TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            PRIMARY KEY (run_id, prompt_group_id)
        );

        CREATE TABLE IF NOT EXISTS images (
            run_id TEXT NOT NULL,
            image_rel_path TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_name TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            max_dim INTEGER NOT NULL,
            PRIMARY KEY (run_id, image_rel_path)
        );

        CREATE TABLE IF NOT EXISTS image_runs (
            run_id TEXT NOT NULL,
            prompt_group_id INTEGER NOT NULL,
            image_rel_path TEXT NOT NULL,
            mode TEXT NOT NULL,
            artifact_dir TEXT NOT NULL,
            status TEXT NOT NULL,
            detection_count INTEGER NOT NULL,
            error_type TEXT,
            error_message TEXT,
            PRIMARY KEY (run_id, prompt_group_id, image_rel_path)
        );

        CREATE TABLE IF NOT EXISTS detections (
            run_id TEXT NOT NULL,
            request_name TEXT NOT NULL,
            engine_name TEXT NOT NULL,
            prompt_group_id INTEGER NOT NULL,
            image_rel_path TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_name TEXT NOT NULL,
            label TEXT NOT NULL,
            score REAL NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL,
            y2 INTEGER NOT NULL,
            box_w INTEGER NOT NULL,
            box_h INTEGER NOT NULL,
            area_px2 INTEGER NOT NULL,
            artifact_dir TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_detections_lookup
        ON detections (run_id, request_name, prompt_group_id, image_rel_path);

        CREATE INDEX IF NOT EXISTS idx_image_runs_status
        ON image_runs (run_id, prompt_group_id, status);
        """
    )
    conn.commit()


def _insert_run_metadata(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    input_dir: Path,
    output_dir: Path,
    checkpoint: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    tiled_threshold: int,
    recursive_find: bool,
    recursive_max_depth: int,
    recursive_min_box_px: int,
    image_count: int,
    request_names: list[str],
    prompt_groups: list[PromptGroup],
) -> None:
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO run_info (
            run_id, created_at_utc, input_dir, output_dir, checkpoint, device,
            box_threshold, text_threshold, max_dets, tiled_threshold,
            recursive_find, recursive_max_depth, recursive_min_box_px, image_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            created_at_utc,
            str(input_dir),
            str(output_dir),
            checkpoint,
            device,
            float(box_threshold),
            float(text_threshold),
            int(max_dets),
            int(tiled_threshold),
            1 if recursive_find else 0,
            int(recursive_max_depth),
            int(recursive_min_box_px),
            int(image_count),
        ),
    )
    for request_name in request_names:
        conn.execute(
            """
            INSERT OR REPLACE INTO request_aliases (run_id, request_name, engine_name, notes)
            VALUES (?, ?, ?, ?)
            """,
            (
                run_id,
                request_name,
                "groundingdino",
                "This repository routes prompt-based DINO detection through the GroundingDINO engine.",
            ),
        )
    for group in prompt_groups:
        conn.execute(
            """
            INSERT OR REPLACE INTO prompt_groups (
                run_id, prompt_group_id, prompt_group_name, prompt_group_slug, prompt_text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, int(group.group_id), group.name, group.slug, group.prompt_text),
        )
    conn.commit()


def _insert_image_metadata(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    image_rel_path: str,
    image_path: Path,
    width: int,
    height: int,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO images (
            run_id, image_rel_path, image_path, image_name, width, height, max_dim
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            image_rel_path,
            str(image_path),
            image_path.name,
            int(width),
            int(height),
            int(max(width, height)),
        ),
    )


def _record_image_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    prompt_group_id: int,
    image_rel_path: str,
    mode: str,
    artifact_dir: Path,
    status: str,
    detection_count: int,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO image_runs (
            run_id, prompt_group_id, image_rel_path, mode, artifact_dir, status,
            detection_count, error_type, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            int(prompt_group_id),
            image_rel_path,
            mode,
            str(artifact_dir),
            status,
            int(detection_count),
            error_type,
            error_message,
        ),
    )


def _delete_existing_detections(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    prompt_group_id: int,
    image_rel_path: str,
) -> None:
    conn.execute(
        """
        DELETE FROM detections
        WHERE run_id = ? AND prompt_group_id = ? AND image_rel_path = ?
        """,
        (run_id, int(prompt_group_id), image_rel_path),
    )


def _insert_detections(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    request_names: list[str],
    prompt_group_id: int,
    image_rel_path: str,
    image_path: Path,
    rows: Iterable,
    artifact_dir: Path,
) -> None:
    payload = []
    for row in rows:
        for request_name in request_names:
            payload.append(
                (
                    run_id,
                    request_name,
                    "groundingdino",
                    int(prompt_group_id),
                    image_rel_path,
                    str(image_path),
                    image_path.name,
                    row.label,
                    float(row.score),
                    int(row.x1),
                    int(row.y1),
                    int(row.x2),
                    int(row.y2),
                    int(row.w),
                    int(row.h),
                    int(row.area_px2),
                    str(artifact_dir),
                )
            )
    if payload:
        conn.executemany(
            """
            INSERT INTO detections (
                run_id, request_name, engine_name, prompt_group_id, image_rel_path, image_path,
                image_name, label, score, x1, y1, x2, y2, box_w, box_h, area_px2, artifact_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )


def _write_manifest(
    *,
    manifest_path: Path,
    run_id: str,
    prompt_groups: list[PromptGroup],
    request_names: list[str],
    input_dir: Path,
    output_dir: Path,
    checkpoint: str,
    total_images: int,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": run_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint": checkpoint,
        "request_names": request_names,
        "engine_name": "groundingdino",
        "prompt_groups": [
            {
                "prompt_group_id": group.group_id,
                "name": group.name,
                "slug": group.slug,
                "prompt_text": group.prompt_text,
                "queries": list(group.queries),
            }
            for group in prompt_groups
        ],
        "image_count": int(total_images),
    }
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_summary_csv(summary_path: Path, summary_rows: list[tuple]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prompt_group_id",
                "prompt_group_name",
                "image_rel_path",
                "status",
                "mode",
                "detection_count",
                "artifact_dir",
                "error_type",
                "error_message",
            ]
        )
        writer.writerows(summary_rows)


def _write_readme(readme_path: Path) -> None:
    text = (
        "semi-labeling output\n"
        "\n"
        "- boxes.sqlite3: aggregated box database.\n"
        "- prompt_groups.json: prompt set manifest.\n"
        "- image_runs.csv: per-image status summary.\n"
        "- artifacts/<prompt-group>/<relative-image-name>/...: optional per-image exports when enabled.\n"
        "\n"
        "Note: in this repository, prompt-based `dino` requests are served by the GroundingDINO engine.\n"
        "The SQLite table `request_aliases` records this mapping explicitly.\n"
    )
    readme_path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run prompt-based GroundingDINO semi-labeling on a folder and export SQLite + artifacts."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing images.")
    parser.add_argument("--output-dir", required=True, help="Root output folder, e.g. /path/to/semi-labeling")
    parser.add_argument(
        "--prompt-group",
        action="append",
        default=[],
        help="Prompt group in the form name=query1, query2, ...; repeat up to many groups.",
    )
    parser.add_argument(
        "--request-names",
        default="dino,groundingdino",
        help="Comma-separated request names to record in SQLite. They share the GroundingDINO engine in this repo.",
    )
    parser.add_argument("--checkpoint", default="", help="GroundingDINO checkpoint path/folder/id. Empty = repo default.")
    parser.add_argument("--box-threshold", type=float, default=0.16)
    parser.add_argument("--text-threshold", type=float, default=0.16)
    parser.add_argument("--max-dets", type=int, default=80)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--recursive-find", action="store_true", help="Scan subfolders for images too.")
    parser.add_argument("--tiled-threshold", type=int, default=512, help="Use tiled recursive detect above this max dimension.")
    parser.add_argument(
        "--tile-scales",
        default="small,medium,large",
        help="Comma-separated recursive tile scales from {small,medium,large}.",
    )
    parser.add_argument("--recursive-max-depth", type=int, default=3)
    parser.add_argument("--recursive-min-box-px", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0, help="Only process first N images (0 = all).")
    parser.add_argument("--verbose-logs", action="store_true", help="Stream detector logs.")
    parser.add_argument("--save-overlay", action="store_true", help="Write overlay.png per image and prompt group.")
    parser.add_argument("--save-box-csv", action="store_true", help="Write box.csv per image and prompt group.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_dir / "boxes.sqlite3"
    readme_path = output_dir / "README.txt"
    prompt_groups_path = output_dir / "prompt_groups.json"
    summary_csv_path = output_dir / "image_runs.csv"

    prompt_groups = _load_prompt_groups(args)
    request_names = _parse_request_names(args.request_names)
    images = _iter_images(input_dir, recursive=bool(args.recursive_find))
    if int(args.limit or 0) > 0:
        images = images[: int(args.limit)]
    if not images:
        print(f"No images found in {input_dir} with extensions: {sorted(_IMAGE_EXTS)}", flush=True)
        return 2

    checkpoint = str(args.checkpoint or "").strip() or str(default_gdino_checkpoint() or "").strip()
    if not checkpoint:
        raise RuntimeError("No GroundingDINO checkpoint available. Pass --checkpoint or download the default model.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = sqlite3.connect(str(sqlite_path))
    _ensure_schema(conn)
    _insert_run_metadata(
        conn,
        run_id=run_id,
        input_dir=input_dir,
        output_dir=output_dir,
        checkpoint=checkpoint,
        device=args.device,
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        max_dets=int(args.max_dets),
        tiled_threshold=int(args.tiled_threshold),
        recursive_find=bool(args.recursive_find),
        recursive_max_depth=int(args.recursive_max_depth),
        recursive_min_box_px=int(args.recursive_min_box_px),
        image_count=len(images),
        request_names=request_names,
        prompt_groups=prompt_groups,
    )
    _write_manifest(
        manifest_path=prompt_groups_path,
        run_id=run_id,
        prompt_groups=prompt_groups,
        request_names=request_names,
        input_dir=input_dir,
        output_dir=output_dir,
        checkpoint=checkpoint,
        total_images=len(images),
    )
    _write_readme(readme_path)

    merged_queries = _combined_queries(prompt_groups)
    params = {
        "gdino_checkpoint": checkpoint,
        "gdino_config_id": "auto",
        "text_queries": merged_queries,
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "max_dets": int(args.max_dets),
        "device": args.device,
        "recursive_tile_scales": [s.strip() for s in str(args.tile_scales).split(",") if s.strip()],
    }
    summary_rows: list[tuple] = []

    service = get_dino_service()
    try:
        for group in prompt_groups:
            print(
                f"Prompt group {group.group_id}/{len(prompt_groups)}: {group.name} -> {group.prompt_text}",
                flush=True,
            )
        total_jobs = len(images)
        for job_idx, img_path in enumerate(images, start=1):
            from PIL import Image

            with Image.open(img_path) as pil_img:
                width, height = pil_img.size
            image_rel_path = _rel_image_key(input_dir, img_path)
            _insert_image_metadata(
                conn,
                run_id=run_id,
                image_rel_path=image_rel_path,
                image_path=img_path,
                width=int(width),
                height=int(height),
            )
            conn.commit()

            max_dim = _max_dim_fast(img_path)
            use_tiled = int(max_dim) > int(args.tiled_threshold)
            mode = "tiled" if use_tiled else "single"
            print(
                f"[{job_idx}/{total_jobs}] {img_path.name} mode={mode}",
                flush=True,
            )
            log_fn = (lambda s: print(s, flush=True)) if bool(args.verbose_logs) else None
            try:
                if use_tiled:
                    result = service.call(
                        "recursive_detect",
                        {
                            "image_path": str(img_path),
                            "params": params,
                            "target_labels": merged_queries,
                            "max_depth": int(args.recursive_max_depth),
                            "min_box_px": int(args.recursive_min_box_px),
                        },
                        log_fn=log_fn,
                    )
                else:
                    result = service.call(
                        "predict",
                        {"image_path": str(img_path), "params": params},
                        log_fn=log_fn,
                    )

                detections = list(result.get("detections") or [])
                grouped_detections: dict[int, list[dict]] = {int(group.group_id): [] for group in prompt_groups}
                for det in detections:
                    group_ids = _match_prompt_groups(det.get("label") or "", prompt_groups)
                    for group_id in group_ids:
                        grouped_detections.setdefault(int(group_id), []).append(dict(det))

                for group in prompt_groups:
                    artifact_dir = _artifact_dir(output_dir, prompt_group=group, input_dir=input_dir, image_path=img_path)
                    group_detections = grouped_detections.get(int(group.group_id), [])
                    rows = _rows_from_detections(
                        image_path=img_path,
                        detections=group_detections,
                        width=int(width),
                        height=int(height),
                    )
                    if args.save_overlay or args.save_box_csv:
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                    if args.save_overlay:
                        overlay_path = artifact_dir / "overlay.png"
                        _write_overlay(image_path=img_path, overlay_path=overlay_path, detections=group_detections)
                    if args.save_box_csv:
                        csv_path = artifact_dir / "box.csv"
                        _write_box_csv(csv_path=csv_path, rows=rows)

                    _delete_existing_detections(
                        conn,
                        run_id=run_id,
                        prompt_group_id=group.group_id,
                        image_rel_path=image_rel_path,
                    )
                    _insert_detections(
                        conn,
                        run_id=run_id,
                        request_names=request_names,
                        prompt_group_id=group.group_id,
                        image_rel_path=image_rel_path,
                        image_path=img_path,
                        rows=rows,
                        artifact_dir=artifact_dir,
                    )
                    _record_image_run(
                        conn,
                        run_id=run_id,
                        prompt_group_id=group.group_id,
                        image_rel_path=image_rel_path,
                        mode=mode,
                        artifact_dir=artifact_dir,
                        status="ok",
                        detection_count=len(rows),
                    )
                    summary_rows.append(
                        (
                            group.group_id,
                            group.name,
                            image_rel_path,
                            "ok",
                            mode,
                            len(rows),
                            str(artifact_dir),
                            "",
                            "",
                        )
                    )
                conn.commit()
            except Exception as exc:
                for group in prompt_groups:
                    artifact_dir = _artifact_dir(output_dir, prompt_group=group, input_dir=input_dir, image_path=img_path)
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    _delete_existing_detections(
                        conn,
                        run_id=run_id,
                        prompt_group_id=group.group_id,
                        image_rel_path=image_rel_path,
                    )
                    _record_image_run(
                        conn,
                        run_id=run_id,
                        prompt_group_id=group.group_id,
                        image_rel_path=image_rel_path,
                        mode=mode,
                        artifact_dir=artifact_dir,
                        status="error",
                        detection_count=0,
                        error_type=exc.__class__.__name__,
                        error_message=str(exc),
                    )
                    _write_error_text(error_path=artifact_dir / "error.txt", image_path=img_path, exc=exc)
                    summary_rows.append(
                        (
                            group.group_id,
                            group.name,
                            image_rel_path,
                            "error",
                            mode,
                            0,
                            str(artifact_dir),
                            exc.__class__.__name__,
                            str(exc),
                        )
                    )
                conn.commit()
                print(f"  error: {img_path.name}: {exc}", flush=True)
                continue

        _write_summary_csv(summary_csv_path, summary_rows)
        print(f"SQLite saved to: {sqlite_path}", flush=True)
        print(f"Artifacts saved to: {output_dir / 'artifacts'}", flush=True)
        print(f"Summary CSV: {summary_csv_path}", flush=True)
        return 0
    finally:
        service.close()
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
