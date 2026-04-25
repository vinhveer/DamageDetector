from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from inference_api.cli_support import print_json

from .lib import load_yolo_class, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m object_detection.yolo.Inference",
        description="Run YOLOv26 inference with Ultralytics.",
    )
    parser.add_argument("--model", required=True, help="Path to model .pt")
    parser.add_argument("--source", required=True, help="Image/file/folder/video source")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="auto", help="Device selector. Examples: auto, cpu, mps, cuda, 0, 0,1, cuda:0,1")
    parser.add_argument("--num-gpus", type=int, default=0, help="How many CUDA GPUs to use when --device is auto/cuda. 0 = all available.")
    parser.add_argument("--batch", type=int, default=1, help="Ultralytics predict batch size for folder/video-style sources.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project", default="object_detection/yolo/inference")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser


def _result_to_dict(result: Any) -> dict[str, Any]:
    boxes_obj = getattr(result, "boxes", None)
    boxes: list[dict[str, Any]] = []
    if boxes_obj is not None:
        xyxy = getattr(boxes_obj, "xyxy", None)
        conf = getattr(boxes_obj, "conf", None)
        cls = getattr(boxes_obj, "cls", None)
        names = getattr(result, "names", {}) or {}
        if xyxy is not None and conf is not None and cls is not None:
            xyxy_list = xyxy.detach().cpu().tolist()
            conf_list = conf.detach().cpu().tolist()
            cls_list = cls.detach().cpu().tolist()
            for box_xyxy, score, class_id in zip(xyxy_list, conf_list, cls_list):
                class_idx = int(class_id)
                boxes.append(
                    {
                        "label": str(names.get(class_idx, class_idx)),
                        "class_id": class_idx,
                        "score": float(score),
                        "box": [float(v) for v in box_xyxy],
                    }
                )
    return {
        "path": str(getattr(result, "path", "")),
        "boxes": boxes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    source_path = Path(args.source).expanduser()
    source: str = str(source_path.resolve()) if source_path.exists() else str(args.source)

    resolved_device = resolve_device(args.device, num_gpus=int(args.num_gpus or 0))

    YOLO = load_yolo_class()
    model = YOLO(str(model_path))
    results = model.predict(
        source=source,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        batch=int(args.batch),
        device=resolved_device,
        save=bool(args.save),
        project=str(args.project),
        name=str(args.name),
    )

    serialized = [_result_to_dict(item) for item in list(results)]
    payload = {
        "status": "ok",
        "model": str(model_path),
        "source": source,
        "device": resolved_device,
        "num_results": len(serialized),
        "results": serialized,
    }
    print_json(payload, pretty=bool(args.pretty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
