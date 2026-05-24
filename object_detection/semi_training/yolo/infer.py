from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from inference_api.cli_support import print_json
from object_detection.semi_training.common.coco import list_split_image_paths_by_class
from object_detection.semi_training.common.semantic_validation import semantic_validate_predictions
from object_detection.yolo.lib import load_yolo_class, resolve_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO inference with optional DINOv2 semantic validation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coco-root", default="", help="Optional COCO root for per-class semantic coverage metrics.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--prototype-dir", default="")
    parser.add_argument("--dinov2-checkpoint", default="")
    parser.add_argument("--semantic-threshold", type=float, default=0.75)
    parser.add_argument("--semantic-reject-threshold", type=float, default=0.50)
    parser.add_argument("--semantic-margin-threshold", type=float, default=0.05)
    parser.add_argument("--semantic-mode", default="coverage", choices=["coverage", "class-consistency"])
    parser.add_argument("--semantic-batch-size", type=int, default=16)
    parser.add_argument("--expand-ratio", type=float, default=0.05)
    parser.add_argument("--prototype-cache", default="")
    parser.add_argument("--preview-limit", type=int, default=200)
    parser.add_argument("--previews", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _serialize_result(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return rows
    names = getattr(result, "names", {}) or {}
    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)
    if xyxy is None or conf is None or cls is None:
        return rows
    for box, score, class_id in zip(xyxy.detach().cpu().tolist(), conf.detach().cpu().tolist(), cls.detach().cpu().tolist()):
        idx = int(class_id)
        rows.append({
            "image_path": str(getattr(result, "path", "")),
            "label": str(names.get(idx, idx)),
            "score": float(score),
            "box": [float(v) for v in box],
        })
    return rows


def run_yolo_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    YOLO = load_yolo_class()
    model = YOLO(str(Path(args.model).expanduser().resolve()))
    device = resolve_device(args.device, num_gpus=int(args.num_gpus or 0))
    results = model.predict(
        source=str(Path(args.source).expanduser()),
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        device=device,
        save=True,
        project=str(Path(args.output_dir).expanduser().resolve()),
        name="predict",
        stream=True,
        verbose=False,
    )
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(_serialize_result(result))
    return rows


def list_source_images(source: str) -> list[Path]:
    path = Path(source).expanduser()
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        return sorted(item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run_yolo_predictions(args)
    report = semantic_validate_predictions(
        predictions=rows,
        output_dir=Path(args.output_dir).expanduser().resolve() / "semantic_validation",
        prototype_dir=args.prototype_dir or None,
        dinov2_checkpoint=args.dinov2_checkpoint or None,
        device=args.device,
        batch_size=int(args.semantic_batch_size),
        threshold=float(args.semantic_threshold),
        reject_threshold=float(args.semantic_reject_threshold),
        margin_threshold=float(args.semantic_margin_threshold),
        decision_mode=str(args.semantic_mode),
        expand_ratio=float(args.expand_ratio),
        prototype_cache_path=args.prototype_cache or None,
        expected_image_paths=list_source_images(args.source),
        expected_per_class=list_split_image_paths_by_class(args.coco_root, args.split) if args.coco_root else None,
        save_previews=bool(args.previews),
        preview_limit=int(args.preview_limit),
    )
    print_json({"status": "ok", **report}, pretty=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
