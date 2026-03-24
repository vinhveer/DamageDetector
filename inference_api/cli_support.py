from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def parse_roi(values: list[int] | tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("ROI must have exactly 4 integers: x1 y1 x2 y2")
    return tuple(int(v) for v in values)


def parse_queries(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def parse_label_list(values: list[str] | None) -> list[str]:
    labels: list[str] = []
    for value in values or []:
        for label in str(value or "").split(","):
            label = label.strip()
            if label:
                labels.append(label)
    return labels


def load_boxes_json(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("detections"), list):
            data = data["detections"]
        elif isinstance(data.get("boxes"), list):
            data = data["boxes"]
    if not isinstance(data, list):
        raise ValueError("Boxes JSON must be a list or an object containing 'detections'/'boxes'.")
    boxes: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        boxes.append(
            {
                "label": str(item.get("label") or "object"),
                "score": float(item.get("score") or 0.0),
                "box": [float(v) for v in box],
            }
        )
    return boxes


def log_to_stderr(message: str) -> None:
    print(str(message), file=sys.stderr, flush=True)


def print_json(data: Any, *, pretty: bool) -> None:
    if pretty:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(json.dumps(data, ensure_ascii=False))
