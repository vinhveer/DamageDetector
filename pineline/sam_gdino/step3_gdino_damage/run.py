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

    from pineline.sam_gdino.common.paths import (
        STEP2_DB,
        STEP3_DB,
        STEP3_OVERLAY_DIR,
        STEP3_RGB_DIR,
        STEP3_SUMMARY_CSV,
        ensure_dirs,
    )
    from pineline.sam_gdino.step3_gdino_damage.prompts import parse_prompt_groups
    from pineline.sam_gdino.step3_gdino_damage.runner import run_step3

    parser = argparse.ArgumentParser(
        description="Step 3 — multi-prompt GroundingDINO damage detect on step2 crops.",
    )
    parser.add_argument("--step2-db", type=Path, default=STEP2_DB)
    parser.add_argument("--source-run-id", type=str, default="latest")
    parser.add_argument("--db", type=Path, default=STEP3_DB)
    parser.add_argument("--rgb-dir", type=Path, default=STEP3_RGB_DIR,
                        help="Where RGBA crops are exported to RGB-on-black for GDINO input.")
    parser.add_argument("--overlay-dir", type=Path, default=STEP3_OVERLAY_DIR)
    parser.add_argument("--summary-csv", type=Path, default=STEP3_SUMMARY_CSV)
    parser.add_argument("--no-overlay", dest="write_overlays", action="store_false")
    parser.set_defaults(write_overlays=True)
    parser.add_argument("--prompt-group", action="append", default=[],
                        help="'name=q1,q2,...' style. Repeat per group. Defaults to crack/spall/exposed_rebar/stain.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.16)
    parser.add_argument("--text-threshold", type=float, default=0.16)
    parser.add_argument("--max-dets", type=int, default=80)
    parser.add_argument("--tiled-threshold", type=int, default=1600)
    parser.add_argument("--tile-scales", type=str, default="small,medium,large")
    parser.add_argument("--recursive-max-depth", type=int, default=2)
    parser.add_argument("--recursive-min-box-px", type=int, default=32)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.50,
                        help="Drop GDINO box whose area / image_area exceeds this ratio.")
    args = parser.parse_args(argv)

    ensure_dirs()
    groups = parse_prompt_groups(args.prompt_group)
    tile_scales = [s.strip() for s in args.tile_scales.split(",") if s.strip()]

    res = run_step3(
        step2_db=args.step2_db,
        db_path=args.db,
        rgb_dir=args.rgb_dir,
        overlay_dir=args.overlay_dir,
        summary_csv=args.summary_csv,
        write_overlays=args.write_overlays,
        prompt_groups=groups,
        source_run_id=args.source_run_id,
        checkpoint=args.checkpoint,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_dets=args.max_dets,
        tiled_threshold=args.tiled_threshold,
        tile_scales=tile_scales,
        recursive_max_depth=args.recursive_max_depth,
        recursive_min_box_px=args.recursive_min_box_px,
        max_box_area_ratio=args.max_box_area_ratio,
    )
    print(res, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
