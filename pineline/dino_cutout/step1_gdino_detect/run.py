from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "object_detection").exists() and (parent / "inference_api").exists():
            return parent
    return here.parents[3]


def main(argv: list[str] | None = None) -> int:
    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pineline.dino_cutout.common.paths import (
        DEFAULT_INPUT_DIR,
        STEP1_DB,
        STEP1_OVERLAY_DIR,
        STEP1_RGB_DIR,
        STEP1_SUMMARY_CSV,
        ensure_dirs,
    )
    from pineline.dino_cutout.step1_gdino_detect.prompts import parse_prompt_groups
    from pineline.dino_cutout.step1_gdino_detect.runner import run_step1

    parser = argparse.ArgumentParser(
        description="dino_cutout step1 — tiled GroundingDINO damage detect on pre-cropped cutouts.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory of RGBA cutout images to detect on.")
    parser.add_argument("--db", type=Path, default=STEP1_DB)
    parser.add_argument("--rgb-dir", type=Path, default=STEP1_RGB_DIR,
                        help="Where RGBA cutouts are exported to RGB-on-black for GDINO input.")
    parser.add_argument("--overlay-dir", type=Path, default=STEP1_OVERLAY_DIR)
    parser.add_argument("--summary-csv", type=Path, default=STEP1_SUMMARY_CSV)
    parser.add_argument("--no-overlay", dest="write_overlays", action="store_false")
    parser.set_defaults(write_overlays=True)
    parser.add_argument("--prompt-group", action="append", default=[],
                        help="'name=q1,q2,...' style. Repeat per group. Defaults to crack/mold/stain.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.10)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--max-dets", type=int, default=150)
    parser.add_argument("--tiled-threshold", type=int, default=400,
                        help="Images with max dim > this use recursive_detect (tiled).")
    parser.add_argument("--tile-scales", type=str, default="small,medium",
                        help="recursive_tile_scales; small/medium maps to ~512 px tiles.")
    parser.add_argument("--recursive-max-depth", type=int, default=2)
    parser.add_argument("--min-box-px", type=int, default=12)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.50,
                        help="Drop GDINO box whose area / image_area exceeds this ratio.")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = no limit. Debug: --limit 5.")
    parser.add_argument("--source-run-id", type=str, default="",
                        help="Override run_id (append into a specific run). Empty = new timestamp.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images already present in the DB for the selected run_id.")
    args = parser.parse_args(argv)

    ensure_dirs()
    groups = parse_prompt_groups(args.prompt_group)
    tile_scales = [s.strip() for s in args.tile_scales.split(",") if s.strip()]

    res = run_step1(
        input_dir=args.input_dir,
        db_path=args.db,
        rgb_dir=args.rgb_dir,
        overlay_dir=args.overlay_dir,
        summary_csv=args.summary_csv,
        write_overlays=args.write_overlays,
        prompt_groups=groups,
        checkpoint=args.checkpoint,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_dets=args.max_dets,
        tiled_threshold=args.tiled_threshold,
        tile_scales=tile_scales,
        recursive_max_depth=args.recursive_max_depth,
        min_box_px=args.min_box_px,
        max_box_area_ratio=args.max_box_area_ratio,
        limit=args.limit,
        source_run_id=(args.source_run_id or None),
        skip_existing=args.skip_existing,
    )
    print(res, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
