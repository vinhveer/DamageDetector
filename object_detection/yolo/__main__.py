from __future__ import annotations

import argparse

from .Inference import main as inference_main
from .train import main as train_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m object_detection.yolo", description="YOLOv26 train/inference entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train YOLOv26 model")
    sub.add_parser("inference", help="Run YOLOv26 inference")
    args, tail = parser.parse_known_args(argv)
    if args.command == "train":
        return train_main(tail)
    return inference_main(tail)


if __name__ == "__main__":
    raise SystemExit(main())
