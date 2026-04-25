from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from typing import Any

from inference_api.cli_support import print_json
from object_detection.datasets import build_yolo_training_kwargs, load_detection_dataset

from .lib import load_yolo_class, resolve_device


@contextmanager
def _pushd(path: str) -> Any:
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m object_detection.yolo.train",
        description="Train YOLOv26 model with Ultralytics.",
    )
    parser.add_argument("--data", required=True, help="Path to shared detection dataset manifest")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained checkpoint or model yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--project", default="object_detection/yolo/train")
    parser.add_argument("--name", default="train")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=None, help="Override initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--augmentation-profile", default="balanced", choices=["light", "balanced", "aggressive"])
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest = load_detection_dataset(args.data)

    resolved_device = resolve_device(args.device)

    YOLO = load_yolo_class()
    model = YOLO(args.model)

    train_kwargs: dict[str, Any] = {
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": resolved_device,
        "workers": int(args.workers),
        "patience": int(args.patience),
        "project": str(args.project),
        "name": str(args.name),
        "optimizer": str(args.optimizer),
        "seed": int(args.seed),
        "cache": bool(args.cache),
        "resume": bool(args.resume),
        "pretrained": bool(args.pretrained),
        "verbose": bool(args.verbose),
    }
    train_kwargs.update(build_yolo_training_kwargs(manifest, args.augmentation_profile))
    if args.lr0 is not None:
        train_kwargs["lr0"] = float(args.lr0)
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = float(args.weight_decay)

    # Ultralytics resolves relative paths in the YAML from the current working directory,
    # not from the YAML file location. Train from the manifest directory so `path: .` works.
    with _pushd(str(manifest.yaml_path.parent)):
        results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        trainer = getattr(model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None)

    payload = {
        "status": "ok",
        "model": str(args.model),
        "data": str(manifest.yaml_path),
        "dataset_root": str(manifest.dataset_root),
        "classes": list(manifest.names),
        "augmentation_profile": str(args.augmentation_profile),
        "device": resolved_device,
        "save_dir": str(save_dir) if save_dir is not None else None,
    }
    print_json(payload, pretty=bool(args.pretty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
