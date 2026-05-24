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
        DEFAULT_SAM_CHECKPOINT,
        STEP1_DB,
        STEP2_CROPS_DIR,
        STEP2_DB,
        STEP2_MASKS_DIR,
        STEP2_OVERLAY_DIR,
        STEP2_SUMMARY_CSV,
        ensure_dirs,
    )
    from pineline.sam_gdino.step2_sam_bridge_crop.runner import run_step2

    parser = argparse.ArgumentParser(
        description="Step 2 — SAM point-prompt segmentation + bridge crop.",
    )
    parser.add_argument("--step1-db", type=Path, default=STEP1_DB,
                        help="SQLite produced by step1.")
    parser.add_argument("--source-run-id", type=str, default="latest",
                        help="Which step1 run_id to consume ('latest' picks newest).")
    parser.add_argument("--db", type=Path, default=STEP2_DB)
    parser.add_argument("--crops-dir", type=Path, default=STEP2_CROPS_DIR)
    parser.add_argument("--masks-dir", type=Path, default=STEP2_MASKS_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=STEP2_OVERLAY_DIR)
    parser.add_argument("--summary-csv", type=Path, default=STEP2_SUMMARY_CSV)
    parser.add_argument("--no-overlay", dest="write_overlays", action="store_false")
    parser.set_defaults(write_overlays=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", type=str, default="auto",
                        choices=["auto", "vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--points-per-box", type=int, default=5,
                        help="N point prompts per box: 1 center + (N-1) quarter anchors.")
    parser.add_argument("--pad-px", type=int, default=16,
                        help="Pixels to pad around mask bbox when cropping.")
    args = parser.parse_args(argv)

    ensure_dirs()
    res = run_step2(
        step1_db=args.step1_db,
        db_path=args.db,
        crops_dir=args.crops_dir,
        masks_dir=args.masks_dir,
        overlay_dir=args.overlay_dir,
        summary_csv=args.summary_csv,
        write_overlays=args.write_overlays,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
        points_per_box=args.points_per_box,
        pad_px=args.pad_px,
        source_run_id=args.source_run_id,
    )
    print(res, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
