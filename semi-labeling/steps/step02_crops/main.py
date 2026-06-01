#!/usr/bin/env python3
"""Step 02 — Multi-View Crop Generation.

Reads an existing resemi run (written by step01), re-reads source detections,
generates multi-view crop PNGs (tight / pad10 / pad25 / context), and updates
crop_views + crop_consistency_features.

Separated from step01 because crop I/O is slow and can be rerun independently
(different crop-dir or view set) without re-running the semantic pass.

Inputs:  resemi.sqlite3 (run from step01) + source images
Outputs: crop_views, crop_consistency_features in resemi.sqlite3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.crop.crop_generation import DEFAULT_VIEW_SPECS, generate_crop_views, parse_crop_encoding, parse_view_specs  # noqa: E402
from shared.runtime.paths import default_image_root, default_resemi_db  # noqa: E402
from steps.step01_semantic.pipeline import ResemiPipeline  # noqa: E402
from shared.db.schema import connect_output  # noqa: E402
from shared.db.source_store import connect_readonly, read_source_detections  # noqa: E402


def _read_run_row(conn, run_id: str) -> dict:
    row = conn.execute(
        "SELECT source_db_path, source_semantic_run_id, options_json FROM resemi_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"Resemi run not found: '{run_id}'. Run step01 first.\n"
            "  python -m run_pipeline run step01 --run-id <id>"
        )
    return {
        "source_db_path": str(row["source_db_path"]),
        "source_semantic_run_id": str(row["source_semantic_run_id"]),
        "options": json.loads(str(row["options_json"])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 02: generate multi-view crop images (tight/pad10/pad25/context) "
            "for an existing resemi run and cache them in crop_views."
        )
    )
    parser.add_argument("--db", default=str(default_resemi_db()),
                        help="Resemi SQLite DB (output of step01).")
    parser.add_argument("--run-id", required=True,
                        help="Resemi run_id (must exist in DB from step01).")
    parser.add_argument("--image-root", default="",
                        help="Override image root. Default: from run options_json.")
    parser.add_argument("--crop-dir", default="",
                        help="Override crop output dir. Default: db_parent/crops/run_id.")
    parser.add_argument("--crop-views", default="",
                        help="Override view list (tight,pad10,pad25,context). Default: from run options.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process first N detections only (debug).")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Parallel threads for crop decode/encode (PNG encode is the bottleneck). 0 = sequential (default).")
    parser.add_argument("--crop-format", default="png", choices=["png", "jpeg"],
                        help="Crop file format. png=lossless (default). jpeg=~11x faster encode + ~40%% file size, but LOSSY.")
    parser.add_argument("--crop-compress-level", type=int, default=1,
                        help="PNG compression level 0-9. Lower=faster, same pixels. Default 1 (PIL default is 6).")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality 1-100 (only when --crop-format jpeg). Default 95.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without generating any files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")

    conn = connect_output(db_path)
    try:
        run_meta = _read_run_row(conn, str(args.run_id))
        options = run_meta["options"]

        image_root_raw = str(args.image_root or "").strip() or options.get("image_root", "")
        image_root = Path(image_root_raw or str(default_image_root())).expanduser().resolve()

        crop_dir_raw = str(args.crop_dir or "").strip() or options.get("crop_dir", "")
        crop_dir = Path(crop_dir_raw).expanduser().resolve() if crop_dir_raw else (
            db_path.parent / "crops" / str(args.run_id)
        )

        crop_views_raw = str(args.crop_views or "").strip()
        if crop_views_raw:
            view_specs = parse_view_specs(crop_views_raw)
        elif options.get("crop_views"):
            view_specs = parse_view_specs(",".join(options["crop_views"]))
        else:
            view_specs = DEFAULT_VIEW_SPECS

        view_names = [spec.name for spec in view_specs]
        print(f"run_id={args.run_id}")
        print(f"image_root={image_root}")
        print(f"crop_dir={crop_dir}")
        print(f"views={view_names}")

        if bool(args.dry_run):
            print("dry_run=True — skipping crop generation")
            return 0

        source_db = Path(run_meta["source_db_path"]).expanduser().resolve()
        if not source_db.is_file():
            raise FileNotFoundError(
                f"Source DB not found: {source_db}\n"
                "The source DB path is stored in the resemi run record."
            )

        source_conn = connect_readonly(source_db)
        try:
            detections = read_source_detections(
                source_conn,
                semantic_run_id=run_meta["source_semantic_run_id"],
                limit=int(args.limit),
            )
        finally:
            source_conn.close()

        print(f"detections={len(detections)}")

        crop_encoding = parse_crop_encoding(
            fmt=str(args.crop_format),
            compress_level=int(args.crop_compress_level),
            jpeg_quality=int(args.jpeg_quality),
        )
        crop_views, crop_errors = generate_crop_views(
            detections,
            image_root=image_root,
            crop_dir=crop_dir,
            view_specs=view_specs,
            num_workers=int(args.num_workers),
            encoding=crop_encoding,
            log_fn=lambda message: print(message, flush=True),
        )

        ok_count = sum(1 for view in crop_views if getattr(view, "status", "ok") == "ok")
        print(f"crop_views_generated={len(crop_views)}  ok={ok_count}  errors={len(crop_errors)}")

        ResemiPipeline._write_crop_views(conn, str(args.run_id), crop_views)
        ResemiPipeline._write_crop_consistency_features(conn, str(args.run_id), detections, crop_views, crop_errors)

        print(f"db={db_path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
