from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# BGR per group_name; fallback white
_PALETTE = {
    "crack": (0, 0, 255),
    "mold":  (0, 200, 0),
    "stain": (0, 255, 255),
}


def _color_for(name: str) -> tuple[int, int, int]:
    return _PALETTE.get(str(name or "").lower(), (255, 255, 255))


def write_overlay(
    *,
    image_path: Path,
    detections: Iterable[dict],
    out_path: Path,
) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    thickness = max(2, int(round(min(h, w) / 400)))
    font_scale = max(0.5, min(h, w) / 1500.0)

    for det in detections:
        box = det.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _color_for(det.get("group_name"))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        # One-word label only (crack / mold / stain) + score.
        name = str(det.get("group_name", "?")).lower()
        text = f"{name} {float(det.get('score', 0)):.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(th + 2, y1)
        cv2.rectangle(img, (x1, ty - th - bl), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
