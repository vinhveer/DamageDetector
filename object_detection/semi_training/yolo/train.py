from __future__ import annotations

import argparse
from pathlib import Path

from object_detection.semi_training.common.yolo_format import ensure_yolo_dataset
from object_detection.yolo.train import main as yolo_train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO on a semi-label COCO dataset.")
    parser.add_argument("--coco-root", required=True, help="COCO dataset root with annotations/ and images/.")
    parser.add_argument("--yolo-dir", default=None, help="Generated YOLO dataset dir. Default: <coco-root>/yolo")
    parser.add_argument("--link-mode", default="symlink", choices=["symlink", "copy"])
    parser.add_argument("tail", nargs=argparse.REMAINDER, help="Arguments forwarded to object_detection.yolo train")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = ensure_yolo_dataset(args.coco_root, args.yolo_dir, link_mode=args.link_mode)
    tail = list(args.tail or [])
    if tail and tail[0] == "--":
        tail = tail[1:]
    if "--data" not in tail:
        tail = ["--data", str(prepared["data_yaml"]), *tail]
    if "--project" not in tail:
        tail = [*tail, "--project", str(Path(args.coco_root).expanduser().resolve() / "runs" / "yolo")]
    return yolo_train_main(tail)


if __name__ == "__main__":
    raise SystemExit(main())
