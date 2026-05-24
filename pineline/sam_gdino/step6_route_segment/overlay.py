from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


_PALETTE = {
    "crack": (0, 0, 255),     # red
    "mold":  (0, 200, 0),     # green
    "stain": (0, 255, 255),   # yellow
}


def _color_for(label: str) -> tuple[int, int, int]:
    return _PALETTE.get(str(label or "").lower(), (255, 255, 255))


def decode_mask_b64(mask_b64: str, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    if not mask_b64:
        return np.zeros((0, 0), dtype=np.uint8)
    raw = base64.b64decode(mask_b64.encode("ascii"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    if expected_shape is not None and mask.shape[:2] != expected_shape:
        mask = cv2.resize(
            mask, (expected_shape[1], expected_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (mask > 0).astype(np.uint8)


def save_mask_png(mask: np.ndarray, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), (mask > 0).astype(np.uint8) * 255)
    return int(np.count_nonzero(mask))


def write_overlay(
    *,
    image_path: Path,
    detections: Iterable[dict],  # each: {box, clip_label, mask, model_used}
    out_path: Path,
    alpha: float = 0.45,
) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    overlay = img.copy()

    for det in detections:
        mask = det.get("mask")
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            continue
        if mask.shape[:2] != (h, w):
            continue
        color = _color_for(det.get("clip_label"))
        color_arr = np.array(color, dtype=np.uint8)
        m3 = np.stack([mask] * 3, axis=-1).astype(bool)
        overlay = np.where(m3, (overlay * (1 - alpha) + color_arr * alpha).astype(np.uint8), overlay)

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
        color = _color_for(det.get("clip_label"))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        text = f"{det.get('clip_label','?')}|{det.get('model_used','?')}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(th + 2, y1)
        cv2.rectangle(overlay, (x1, ty - th - bl), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(overlay, text, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
