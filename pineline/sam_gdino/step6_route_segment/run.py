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
        DEFAULT_SAM_LORA_BASE_CHECKPOINT,
        DEFAULT_SAM_LORA_DELTA,
        STEP3_DB,
        STEP3_RGB_DIR,
        STEP4_DB,
        STEP5_DB,
        STEP6_DB,
        STEP6_MASKS_DIR,
        STEP6_OVERLAY_DIR,
        STEP6_SUMMARY_CSV,
        ensure_dirs,
    )
    from pineline.sam_gdino.step6_route_segment.runner import run_step6

    parser = argparse.ArgumentParser(
        description="Step 6 — route crack -> SAM-LoRA, other -> SAM zero-shot.",
    )
    parser.add_argument("--step3-db", type=Path, default=STEP3_DB)
    parser.add_argument("--step4-db", type=Path, default=STEP4_DB)
    parser.add_argument("--step5-db", type=Path, default=STEP5_DB)
    parser.add_argument("--rgb-dir", type=Path, default=STEP3_RGB_DIR)
    parser.add_argument("--step3-run-id", type=str, default="latest")
    parser.add_argument("--step4-run-id", type=str, default="latest")
    parser.add_argument("--step5-run-id", type=str, default="latest")
    parser.add_argument("--db", type=Path, default=STEP6_DB)
    parser.add_argument("--masks-dir", type=Path, default=STEP6_MASKS_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=STEP6_OVERLAY_DIR)
    parser.add_argument("--summary-csv", type=Path, default=STEP6_SUMMARY_CSV)
    parser.add_argument("--no-overlay", dest="write_overlays", action="store_false")
    parser.set_defaults(write_overlays=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", type=str, default="auto",
                        choices=["auto", "vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam-lora-base", type=Path, default=DEFAULT_SAM_LORA_BASE_CHECKPOINT)
    parser.add_argument("--sam-lora-delta", type=Path, default=DEFAULT_SAM_LORA_DELTA)
    parser.add_argument("--sam-lora-middle-dim", type=int, default=32)
    parser.add_argument("--sam-lora-scaling-factor", type=float, default=0.2)
    parser.add_argument("--sam-lora-rank", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)

    ensure_dirs()
    res = run_step6(
        step3_db=args.step3_db,
        step4_db=args.step4_db,
        step5_db=args.step5_db,
        rgb_dir=args.rgb_dir,
        db_path=args.db,
        masks_dir=args.masks_dir,
        overlay_dir=args.overlay_dir,
        summary_csv=args.summary_csv,
        write_overlays=args.write_overlays,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        sam_lora_base=args.sam_lora_base,
        sam_lora_delta=args.sam_lora_delta,
        sam_lora_middle_dim=args.sam_lora_middle_dim,
        sam_lora_scaling_factor=args.sam_lora_scaling_factor,
        sam_lora_rank=args.sam_lora_rank,
        device=args.device,
        step3_run_id=args.step3_run_id,
        step4_run_id=args.step4_run_id,
        step5_run_id=args.step5_run_id,
    )
    print(res, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
