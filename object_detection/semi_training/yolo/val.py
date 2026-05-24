from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from object_detection.semi_training.common.coco import discover_coco_dataset, load_json, write_json
from object_detection.semi_training.common.yolo_format import ensure_yolo_dataset
from object_detection.semi_training.yolo.infer import main as infer_main
from object_detection.yolo.lib import load_yolo_class, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run recall-oriented semantic validation for a YOLO model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--map", action=argparse.BooleanOptionalAction, default=True, help="Run YOLO native model.val() pseudo-mAP before semantic validation.")
    parser.add_argument("--map-conf", type=float, default=None, help="Optional confidence override for YOLO native val. Default keeps Ultralytics val default.")
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


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _public_scalar_attrs(obj: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        scalar = _safe_float(value)
        if scalar is not None:
            out[name] = scalar
    return out


def _to_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return list(value.tolist())
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _extract_yolo_map_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", None)
    payload: dict[str, Any] = {
        "map50": None,
        "map50_95": None,
        "precision": None,
        "recall": None,
        "fitness": _safe_float(getattr(metrics, "fitness", None)),
        "speed": dict(speed) if isinstance(speed, dict) else speed,
        "box_metrics": _public_scalar_attrs(box) if box is not None else {},
    }
    if box is not None:
        payload["map50"] = _safe_float(getattr(box, "map50", None))
        payload["map50_95"] = _safe_float(getattr(box, "map", None))
        payload["precision"] = _safe_float(getattr(box, "mp", None))
        payload["recall"] = _safe_float(getattr(box, "mr", None))
        names = getattr(metrics, "names", {}) or {}
        ap_class_index = _to_list(getattr(box, "ap_class_index", None))
        ap50_per_class = _to_list(getattr(box, "ap50", None))
        ap_per_class = _to_list(getattr(box, "ap", None))
        p_per_class = _to_list(getattr(box, "p", None))
        r_per_class = _to_list(getattr(box, "r", None))
        if ap_class_index is not None and ap50_per_class is not None:
            payload["per_class"] = [
                {
                    "class": names.get(int(class_id), str(class_id)),
                    "map50": float(ap50_per_class[index]),
                    "map50_95": float(ap_per_class[index]) if ap_per_class is not None else None,
                    "precision": float(p_per_class[index]) if p_per_class is not None else None,
                    "recall": float(r_per_class[index]) if r_per_class is not None else None,
                }
                for index, class_id in enumerate(ap_class_index)
                if index < len(ap50_per_class)
            ]
    if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
        payload["results_dict"] = {str(k): v for k, v in metrics.results_dict.items()}
    elif is_dataclass(metrics):
        payload["results_dict"] = asdict(metrics)
    return payload


def run_native_map(args: argparse.Namespace, *, data_yaml: str, output_dir: Path) -> dict[str, Any]:
    YOLO = load_yolo_class()
    model = YOLO(str(Path(args.model).expanduser().resolve()))
    device = resolve_device(args.device, num_gpus=int(args.num_gpus or 0))
    kwargs: dict[str, Any] = {
        "data": data_yaml,
        "split": str(args.split),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "device": device,
        "workers": int(args.workers),
        "project": str(output_dir),
        "name": "native_map",
        "verbose": False,
        "save_json": False,
        "plots": False,
    }
    if args.map_conf is not None:
        kwargs["conf"] = float(args.map_conf)
    metrics = model.val(**kwargs)
    report = {
        "kind": "pseudo_map",
        "model": str(Path(args.model).expanduser()),
        "data_yaml": data_yaml,
        "split": str(args.split),
        "device": device,
        "metrics": _extract_yolo_map_metrics(metrics),
        "save_dir": str(getattr(metrics, "save_dir", output_dir / "native_map")),
    }
    write_json(output_dir / "native_map_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = discover_coco_dataset(args.coco_root, splits=(str(args.split),))
    yolo_dataset = ensure_yolo_dataset(args.coco_root)
    map_report = run_native_map(args, data_yaml=str(yolo_dataset["data_yaml"]), output_dir=out_dir) if bool(args.map) else None
    source = dataset.splits[0].image_dir
    tail = [
        "--model", args.model,
        "--source", str(source),
        "--output-dir", str(out_dir),
        "--coco-root", args.coco_root,
        "--split", str(args.split),
        "--imgsz", str(args.imgsz),
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--max-det", str(args.max_det),
        "--device", args.device,
        "--num-gpus", str(args.num_gpus),
        "--semantic-threshold", str(args.semantic_threshold),
        "--semantic-reject-threshold", str(args.semantic_reject_threshold),
        "--semantic-margin-threshold", str(args.semantic_margin_threshold),
        "--semantic-mode", str(args.semantic_mode),
        "--semantic-batch-size", str(args.semantic_batch_size),
        "--expand-ratio", str(args.expand_ratio),
        "--preview-limit", str(args.preview_limit),
    ]
    if not args.previews:
        tail.append("--no-previews")
    if args.prototype_cache:
        tail.extend(["--prototype-cache", args.prototype_cache])
    if args.prototype_dir:
        tail.extend(["--prototype-dir", args.prototype_dir])
    if args.dinov2_checkpoint:
        tail.extend(["--dinov2-checkpoint", args.dinov2_checkpoint])
    code = infer_main(tail)
    semantic_report_path = out_dir / "semantic_validation" / "semantic_validation_report.json"
    semantic_report = load_json(semantic_report_path) if semantic_report_path.is_file() else None
    unified = {
        "kind": "unified_validation",
        "framing": {
            "pseudo_map": "YOLO native validation on semi-label annotations; use as annotation-fit monitor, not ground-truth benchmark.",
            "semantic_validation": "DINOv2 prototype consistency and image-level coverage; prioritize this when pseudo-mAP and semantic metrics disagree.",
        },
        "model": str(Path(args.model).expanduser()),
        "coco_root": str(Path(args.coco_root).expanduser().resolve()),
        "split": str(args.split),
        "native_map": map_report,
        "semantic_validation": semantic_report,
        "selection_hint": {
            "primary": "Prefer checkpoints with high semantic pass/coverage and acceptable pseudo-mAP.",
            "tie_breaker": "If pseudo-mAP and semantic validation disagree, inspect review_queue.csv and prefer semantic coverage for damage-recall use cases.",
        },
    }
    write_json(out_dir / "unified_validation_report.json", unified)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
