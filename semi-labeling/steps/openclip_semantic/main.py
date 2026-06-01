#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from shared.runtime import bootstrap

# OpenCLIP (open_clip / torch) imports are heavy; bootstrap puts the repo root
# and semi-labeling on sys.path so torch_runtime + the package resolve.
bootstrap.ensure_repo_root_on_path()

from steps.openclip_semantic.pipeline import Step2SemanticConfig, Step2SemanticPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop damage_scan boxes, classify them with OpenCLIP, and save semantic percentages to SQLite.")
    parser.add_argument("--db", required=True, help="Path to damage_scan.sqlite3.")
    parser.add_argument("--image-root", default="", help="Override image root used to resolve `images.rel_path` from the DB.")
    parser.add_argument("--source-run-id", default="latest", help="Damage scan run_id to read detections from. Default: latest")
    parser.add_argument("--stage", default="final", help="Detection stage to classify. Default: final")
    parser.add_argument("--detection-id", action="append", default=[], help="Specific detection_id to classify; repeat as needed.")
    parser.add_argument("--limit", type=int, default=0, help="Only classify first N matching detections. 0 = all.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for multi-process inference. Default: 0")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for multi-process inference. Default: 1")
    parser.add_argument("--model-name", default="ViT-B-32", help="OpenCLIP model name, e.g. ViT-B-32 or ViT-H-14.")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag, e.g. laion2b_s34b_b79k.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=16, help="How many crops to encode per OpenCLIP image batch. Default: 16")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Background threads to decode/crop images while the GPU runs. 0 = sequential (default).")
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=False, help="Save crop PNGs for each classified detection.")
    parser.add_argument("--crop-dir", default="", help="Optional crop output root. Used only when --save-crops is enabled.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Step2SemanticConfig(
        db_path=Path(args.db).expanduser().resolve(),
        source_run_id=str(args.source_run_id),
        stage=str(args.stage),
        limit=int(args.limit),
        detection_ids=tuple(int(item) for item in list(args.detection_id or [])),
        image_root=Path(args.image_root).expanduser().resolve() if str(args.image_root or "").strip() else None,
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        model_name=str(args.model_name),
        pretrained=str(args.pretrained),
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        save_crops=bool(args.save_crops),
        crop_dir=Path(args.crop_dir).expanduser().resolve() if str(args.crop_dir or "").strip() else None,
    )
    pipeline = Step2SemanticPipeline(config)
    try:
        semantic_run_id = pipeline.run(log_fn=lambda message: print(message, flush=True))
    finally:
        pipeline.close()
    print(f"semantic_run_id={semantic_run_id}")
    print(f"sqlite={config.db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
