#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "object_detection").exists() and (candidate / "tools").exists():
            return candidate
    return current.parents[2]


_REPO_ROOT = _resolve_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import Step2SemanticConfig, Step2SemanticPipeline


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
