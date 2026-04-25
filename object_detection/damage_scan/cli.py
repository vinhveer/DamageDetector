from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import DamageScanConfig, DamageScanPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-image crack/mold/spall GroundingDINO scan and store boxes in SQLite.")
    parser.add_argument("--input-dir", type=Path, default=Path("../HinhAnh"))
    parser.add_argument("--db", type=Path, default=Path("semi-labeling/step1_gdino_labeling/damage_scan/damage_scan.sqlite3"))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--final-max-dets-per-class", type=int, default=200)
    parser.add_argument("--image-workers", type=int, default=1, help="How many images to process concurrently.")
    parser.add_argument("--service-workers", type=int, default=0, help="How many DINO worker processes to keep. 0 = auto.")
    parser.add_argument("--service-queue-size", type=int, default=0, help="How many waiting DINO calls to buffer. 0 = auto.")
    parser.add_argument("--service-batch-size", type=int, default=0, help="Chunk size per DINO worker for predict-batch. 0 = auto.")
    parser.add_argument("--service-device-ids", default="", help="Optional comma-separated CUDA ids for DINO workers, e.g. 0,1.")
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-full-raw-in-overlay", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = DamageScanConfig(
        input_dir=Path(args.input_dir),
        db_path=Path(args.db),
        checkpoint=str(args.checkpoint or ""),
        device=str(args.device),
        recursive=bool(args.recursive),
        limit=int(args.limit),
        final_max_dets_per_class=int(args.final_max_dets_per_class),
        image_workers=int(args.image_workers),
        service_workers=int(args.service_workers),
        service_queue_size=int(args.service_queue_size),
        service_batch_size=int(args.service_batch_size),
        service_device_ids=str(args.service_device_ids or ""),
        save_overlays=bool(args.save_overlays),
        include_full_raw_in_overlay=bool(args.include_full_raw_in_overlay),
    )
    pipeline = DamageScanPipeline(config)
    try:
        status_log = lambda message: print(message, flush=True)
        detector_log = status_log if bool(args.verbose) else None
        run_id = pipeline.run(log_fn=status_log, detector_log_fn=detector_log)
    finally:
        pipeline.close()
    print(f"run_id={run_id}")
    print(f"sqlite={Path(args.db).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
