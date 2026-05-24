from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


_BOX_COLOR = (0, 255, 0)   # BGR — green
_TEXT_COLOR = (255, 255, 255)
_TEXT_BG = (0, 0, 0)


def _safe_name(rel_path: str) -> str:
    return rel_path.replace("/", "__").replace("\\", "__")


def overlay_path_for(overlay_dir: Path, image_id: str, rel_path: str) -> Path:
    stem = Path(rel_path).stem
    return overlay_dir / f"{image_id}__{_safe_name(stem)}.png"


def write_overlay(
    *,
    image_path: Path,
    detections: Iterable[dict],
    out_path: Path,
) -> None:
    """Draw boxes + scores on a copy of the image and save to out_path."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        # Pillow fallback (handles odd encodings)
        from PIL import Image

        with Image.open(image_path) as pil:
            img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    thickness = max(2, int(round(min(h, w) / 400)))
    font_scale = max(0.5, min(h, w) / 1200.0)

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

        cv2.rectangle(img, (x1, y1), (x2, y2), _BOX_COLOR, thickness)

        score = float(det.get("score") or 0.0)
        label = str(det.get("label") or "bridge")
        text = f"{label} {score:.2f}"

        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
        )
        ty = max(th + 2, y1)
        cv2.rectangle(
            img, (x1, ty - th - baseline), (x1 + tw + 4, ty + 2), _TEXT_BG, -1,
        )
        cv2.putText(
            img, text, (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, _TEXT_COLOR, thickness,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
