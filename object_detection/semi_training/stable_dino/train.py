from __future__ import annotations

import argparse
from pathlib import Path

from object_detection.semi_training.common.stable_format import ensure_stable_dino_dataset_yaml
from object_detection.stable_dino.train import main as stable_train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train StableDINO on a semi-label COCO dataset.")
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--dataset-yaml", default=None)
    parser.add_argument("tail", nargs=argparse.REMAINDER, help="Arguments forwarded to object_detection.stable_dino.train")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = ensure_stable_dino_dataset_yaml(args.coco_root, args.dataset_yaml)
    tail = list(args.tail or [])
    if tail and tail[0] == "--":
        tail = tail[1:]
    if "--dataset" not in tail:
        tail = ["--dataset", str(prepared["dataset_yaml"]), *tail]
    if "--output-dir" not in tail:
        tail = [*tail, "--output-dir", str(Path(args.coco_root).expanduser().resolve() / "runs" / "stable_dino")]
    return stable_train_main(tail)


if __name__ == "__main__":
    raise SystemExit(main())
