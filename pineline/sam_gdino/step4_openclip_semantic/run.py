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
        STEP3_DB,
        STEP3_RGB_DIR,
        STEP4_CROPS_DIR,
        STEP4_DB,
        STEP4_SUMMARY_CSV,
        ensure_dirs,
    )
    from pineline.sam_gdino.step4_openclip_semantic.runner import run_step4

    parser = argparse.ArgumentParser(
        description="Step 4 — OpenCLIP semantic top-1 label for each step3 detection.",
    )
    parser.add_argument("--step3-db", type=Path, default=STEP3_DB)
    parser.add_argument("--rgb-dir", type=Path, default=STEP3_RGB_DIR,
                        help="Directory containing <parent_image_id>.png RGB crops from step3.")
    parser.add_argument("--source-run-id", type=str, default="latest")
    parser.add_argument("--db", type=Path, default=STEP4_DB)
    parser.add_argument("--crops-dir", type=Path, default=STEP4_CROPS_DIR)
    parser.add_argument("--summary-csv", type=Path, default=STEP4_SUMMARY_CSV)
    parser.add_argument("--save-crops", action="store_true",
                        help="Persist per-detection crops to --crops-dir for inspection.")
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-prob", type=float, default=0.45,
                        help="Drop detection when top-1 CLIP probability < min-prob.")
    args = parser.parse_args(argv)

    ensure_dirs()
    res = run_step4(
        step3_db=args.step3_db,
        rgb_dir=args.rgb_dir,
        db_path=args.db,
        crops_dir=args.crops_dir,
        summary_csv=args.summary_csv,
        source_run_id=args.source_run_id,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        save_crops=args.save_crops,
        min_prob=args.min_prob,
    )
    print(res, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
