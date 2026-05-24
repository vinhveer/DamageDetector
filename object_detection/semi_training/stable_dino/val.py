from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from object_detection.semi_training.common.coco import list_split_image_paths, list_split_image_paths_by_class, write_json
from object_detection.semi_training.common.predictions import coco_detection_results_as_predictions
from object_detection.semi_training.common.semantic_validation import semantic_validate_predictions
from object_detection.semi_training.common.stable_format import ensure_stable_dino_dataset_yaml
from object_detection.stable_dino.train import main as stable_train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StableDINO eval and optional semantic validation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--dataset-yaml", default=None)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("tail", nargs=argparse.REMAINDER, help="Extra StableDINO eval args")
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_per_class_bbox_metrics(bbox: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(bbox, dict):
        return []
    by_label: dict[str, dict[str, Any]] = {}
    for key, value in bbox.items():
        if key.startswith("AP50-"):
            label = key[len("AP50-"):]
            by_label.setdefault(label, {"class": label, "map50": None, "map50_95": None, "precision": None, "recall": None})["map50"] = _safe_float(value)
        elif key.startswith("AP-"):
            label = key[len("AP-"):]
            by_label.setdefault(label, {"class": label, "map50": None, "map50_95": None, "precision": None, "recall": None})["map50_95"] = _safe_float(value)
    return list(by_label.values())


def _extract_coco_bbox_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    bbox = metrics.get("bbox") if isinstance(metrics.get("bbox"), dict) else None
    bbox_wnms = metrics.get("bbox_wnms") if isinstance(metrics.get("bbox_wnms"), dict) else None
    payload: dict[str, Any] = {
        "map50": _safe_float(bbox.get("AP50")) if bbox else None,
        "map50_95": _safe_float(bbox.get("AP")) if bbox else None,
        "bbox_metrics": bbox or {},
    }
    per_class = _extract_per_class_bbox_metrics(bbox)
    if per_class:
        payload["per_class"] = per_class
    if bbox_wnms:
        payload["with_nms"] = {
            "map50": _safe_float(bbox_wnms.get("AP50")),
            "map50_95": _safe_float(bbox_wnms.get("AP")),
            "bbox_metrics": bbox_wnms,
        }
        per_class_wnms = _extract_per_class_bbox_metrics(bbox_wnms)
        if per_class_wnms:
            payload["with_nms"]["per_class"] = per_class_wnms
    return payload


def _find_prediction_path(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "inference" / "coco_instances_results.json",
        output_dir / "coco_instances_results.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _write_native_map_report(
    *,
    output_dir: Path,
    checkpoint: str,
    dataset_yaml: Path,
    split: str,
    device: str,
    extra_args: list[str],
    prediction_path: Path | None,
) -> dict[str, Any]:
    metrics_path = output_dir / "eval_metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.is_file() else None
    prediction_count = None
    if prediction_path is not None:
        try:
            predictions = _read_json(prediction_path)
            prediction_count = len(predictions) if isinstance(predictions, list) else None
        except Exception:
            prediction_count = None
    report = {
        "kind": "pseudo_map",
        "model_family": "StableDINO",
        "checkpoint": str(Path(checkpoint).expanduser()),
        "data_yaml": str(dataset_yaml),
        "split": str(split),
        "device": str(device),
        "extra_args": list(extra_args),
        "metrics": _extract_coco_bbox_metrics(metrics),
        "raw_metrics": metrics,
        "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
        "prediction_path": str(prediction_path) if prediction_path is not None else None,
        "prediction_count": prediction_count,
    }
    write_json(output_dir / "native_map_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = ensure_stable_dino_dataset_yaml(args.coco_root, args.dataset_yaml)
    tail = list(args.tail or [])
    if tail and tail[0] == "--":
        tail = tail[1:]
    eval_args = [
        "--dataset", str(prepared["dataset_yaml"]),
        "--init-checkpoint", str(Path(args.checkpoint).expanduser().resolve()),
        "--output-dir", str(out_dir),
        "--eval-only",
        "--eval-split", str(args.split),
        "--device", str(args.device),
        *tail,
    ]
    stable_train_main(eval_args)

    prediction_path = _find_prediction_path(out_dir)
    map_report = _write_native_map_report(
        output_dir=out_dir,
        checkpoint=args.checkpoint,
        dataset_yaml=Path(prepared["dataset_yaml"]),
        split=str(args.split),
        device=str(args.device),
        extra_args=tail,
        prediction_path=prediction_path,
    )
    semantic_report = None
    if prediction_path is not None and args.prototype_dir and args.dinov2_checkpoint:
        rows = coco_detection_results_as_predictions(coco_root=args.coco_root, split_name=args.split, results_json=prediction_path)
        semantic_report = semantic_validate_predictions(
            predictions=rows,
            output_dir=out_dir / "semantic_validation",
            prototype_dir=args.prototype_dir,
            dinov2_checkpoint=args.dinov2_checkpoint,
            device=args.device,
            batch_size=int(args.semantic_batch_size),
            threshold=float(args.semantic_threshold),
            reject_threshold=float(args.semantic_reject_threshold),
            margin_threshold=float(args.semantic_margin_threshold),
            decision_mode=str(args.semantic_mode),
            expand_ratio=float(args.expand_ratio),
            prototype_cache_path=args.prototype_cache or None,
            expected_image_paths=list_split_image_paths(args.coco_root, args.split),
            expected_per_class=list_split_image_paths_by_class(args.coco_root, args.split),
            save_previews=bool(args.previews),
            preview_limit=int(args.preview_limit),
        )
    unified = {
        "kind": "unified_validation",
        "framing": {
            "pseudo_map": "StableDINO COCO validation on semi-label annotations; use as annotation-fit monitor, not ground-truth benchmark.",
            "semantic_validation": "DINOv2 prototype consistency and image-level coverage; prioritize this when pseudo-mAP and semantic metrics disagree.",
        },
        "model_family": "StableDINO",
        "checkpoint": str(Path(args.checkpoint).expanduser()),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
