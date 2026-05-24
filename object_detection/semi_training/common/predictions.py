from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .coco import coco_image_maps, discover_coco_dataset, load_json, resolve_image_path


def coco_annotations_as_predictions(*, coco_root: str | Path, split_name: str = "val", limit: int = 0) -> list[dict[str, Any]]:
    dataset = discover_coco_dataset(coco_root, splits=(split_name,))
    split = dataset.splits[0]
    categories = {category.id: category.name for category in dataset.categories}
    payload = load_json(split.annotation_path)
    images, anns_by_image = coco_image_maps(payload)
    rows: list[dict[str, Any]] = []
    for image_id, image in sorted(images.items()):
        image_path = resolve_image_path(split, str(image.get("file_name", "")))
        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = (float(v) for v in ann.get("bbox", [0, 0, 0, 0]))
            rows.append({
                "image_path": str(image_path),
                "label": categories.get(int(ann.get("category_id", -1)), ""),
                "score": float(ann.get("score", 1.0)),
                "box": [x, y, x + w, y + h],
            })
            if limit and len(rows) >= int(limit):
                return rows
    return rows


def load_predictions_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("predictions"), list):
            return [row for row in payload["predictions"] if isinstance(row, dict)]
        if isinstance(payload.get("results"), list):
            rows: list[dict[str, Any]] = []
            for result in payload["results"]:
                image_path = str(result.get("path") or result.get("image_path") or "")
                for box in result.get("boxes", []) or []:
                    if isinstance(box, dict):
                        rows.append({"image_path": image_path, **box})
            return rows
    raise ValueError(f"Unsupported predictions JSON format: {path}")


def coco_detection_results_as_predictions(*, coco_root: str | Path, split_name: str, results_json: str | Path) -> list[dict[str, Any]]:
    dataset = discover_coco_dataset(coco_root, splits=(split_name,))
    split = dataset.splits[0]
    categories = {category.id: category.name for category in dataset.categories}
    payload = load_json(split.annotation_path)
    images, _ = coco_image_maps(payload)
    result_rows = json.loads(Path(results_json).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(result_rows, list):
        raise ValueError(f"COCO detection results must be a list: {results_json}")
    rows: list[dict[str, Any]] = []
    for item in result_rows:
        if not isinstance(item, dict):
            continue
        image = images.get(int(item.get("image_id", -1)))
        if image is None:
            continue
        x, y, w, h = (float(v) for v in item.get("bbox", [0, 0, 0, 0]))
        rows.append({
            "image_path": str(resolve_image_path(split, str(image.get("file_name", "")))),
            "label": categories.get(int(item.get("category_id", -1)), ""),
            "score": float(item.get("score", 0.0)),
            "box": [x, y, x + w, y + h],
        })
    return rows
