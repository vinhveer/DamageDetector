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
        STEP4_DB,
        STEP5_DB,
        STEP5_SUMMARY_CSV,
        ensure_dirs,
    )
    from pineline.sam_gdino.step5_embedding.runner import run_step5

    parser = argparse.ArgumentParser(
        description="Step 5 — DINOv2 + IoU intra-image dedup (no embeddings persisted).",
    )
    parser.add_argument("--step3-db", type=Path, default=STEP3_DB)
    parser.add_argument("--step4-db", type=Path, default=STEP4_DB)
    parser.add_argument("--rgb-dir", type=Path, default=STEP3_RGB_DIR)
    parser.add_argument("--step3-run-id", type=str, default="latest")
    parser.add_argument("--step4-run-id", type=str, default="latest")
    parser.add_argument("--db", type=Path, default=STEP5_DB)
    parser.add_argument("--summary-csv", type=Path, default=STEP5_SUMMARY_CSV)
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-small")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--cosine-threshold", type=float, default=0.85)
    args = parser.parse_args(argv)

    ensure_dirs()
    res = run_step5(
        step3_db=args.step3_db,
        step4_db=args.step4_db,
        rgb_dir=args.rgb_dir,
        db_path=args.db,
        summary_csv=args.summary_csv,
        step3_run_id=args.step3_run_id,
        step4_run_id=args.step4_run_id,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        iou_threshold=args.iou_threshold,
        cosine_threshold=args.cosine_threshold,
    )
    print(res, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
