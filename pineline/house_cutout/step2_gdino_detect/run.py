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
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pineline.house_cutout.common.paths import (
        STEP1_CUTOUTS_DIR,
        STEP2_DB,
        STEP2_OVERLAY_DIR,
        STEP2_RGB_DIR,
        STEP2_SUMMARY_CSV,
        ensure_step2_dirs,
    )
    from pineline.house_cutout.step2_gdino_detect.prompts import parse_prompt_groups
    from pineline.house_cutout.step2_gdino_detect.runner import run_step1 as run_step2

    parser = argparse.ArgumentParser(
        description=(
            "house_cutout step2 — tiled GroundingDINO detect damage (crack/mold/stain) "
            "trên cutout nhà từ step1."
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=STEP1_CUTOUTS_DIR,
                        help="Thư mục cutout RGBA (mặc định: cutouts của step1).")
    parser.add_argument("--db", type=Path, default=STEP2_DB)
    parser.add_argument("--rgb-dir", type=Path, default=STEP2_RGB_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=STEP2_OVERLAY_DIR)
    parser.add_argument("--summary-csv", type=Path, default=STEP2_SUMMARY_CSV)
    parser.add_argument("--no-overlay", dest="write_overlays", action="store_false")
    parser.set_defaults(write_overlays=True)
    parser.add_argument("--prompt-group", action="append", default=[],
                        help="'name=q1,q2,...'. Lặp lại được. Mặc định crack/mold/stain.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.10)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--max-dets", type=int, default=150)
    parser.add_argument("--tiled-threshold", type=int, default=400)
    parser.add_argument("--tile-scales", type=str, default="small,medium")
    parser.add_argument("--recursive-max-depth", type=int, default=2)
    parser.add_argument("--min-box-px", type=int, default=12)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-run-id", type=str, default="")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    ensure_step2_dirs()
    groups = parse_prompt_groups(args.prompt_group)
    tile_scales = [s.strip() for s in args.tile_scales.split(",") if s.strip()]

    res = run_step2(
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
