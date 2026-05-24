from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_image_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        from PIL import Image

        with Image.open(image_path) as pil:
            rgb = np.array(pil.convert("RGB"))
        return rgb
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict_mask_for_box(
    predictor: Any,
    points_xy: list[tuple[float, float]],
    *,
    multimask_output: bool = True,
) -> np.ndarray:
    """Run predictor on the points (foreground labels) and return best mask (bool)."""
    if not points_xy:
        return np.zeros((0, 0), dtype=bool)
    coords = np.array(points_xy, dtype=np.float32)
    labels = np.ones((coords.shape[0],), dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=bool(multimask_output),
    )
    if multimask_output:
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool)
    return masks[0].astype(bool)


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    masks = [m for m in masks if m.size > 0]
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    out = masks[0].copy()
    for m in masks[1:]:
        if m.shape != out.shape:
            continue
        np.logical_or(out, m, out=out)
    return out


def mask_bbox(mask: np.ndarray, *, pad_px: int = 0) -> tuple[int, int, int, int] | None:
    if mask.size == 0 or not mask.any():
        return None
    ys, xs = np.where(mask)
    h, w = mask.shape[:2]
    x1 = max(0, int(xs.min()) - pad_px)
    y1 = max(0, int(ys.min()) - pad_px)
    x2 = min(w, int(xs.max()) + 1 + pad_px)
    y2 = min(h, int(ys.max()) + 1 + pad_px)
    return x1, y1, x2, y2


def write_crop_and_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    crop_path: Path,
    mask_path: Path,
) -> int:
    """Write a transparent-background PNG cropped to mask bbox.

    The crop is RGBA: pixels outside the mask have alpha=0 (transparent),
    pixels inside keep their original RGB with alpha=255. The mask PNG is
    a separate binary file (uint8, 0/255).
    """
    x1, y1, x2, y2 = bbox
    rgb_crop = image_rgb[y1:y2, x1:x2]
    mask_crop = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)

    rgba = np.zeros((rgb_crop.shape[0], rgb_crop.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb_crop
    rgba[..., 3] = mask_crop * 255

    crop_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2 expects BGRA on disk; convert RGB->BGR while keeping alpha
    bgra = np.dstack([cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR), rgba[..., 3]])
    cv2.imwrite(str(crop_path), bgra)
    cv2.imwrite(str(mask_path), mask_crop * 255)
    return int(mask.sum())


def write_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    points_by_box: list[list[tuple[float, float]]],
    bbox: tuple[int, int, int, int] | None,
    *,
    out_path: Path,
    alpha: float = 0.45,
) -> None:
    overlay = image_rgb.copy()
    if mask.shape == image_rgb.shape[:2]:
        green = np.zeros_like(image_rgb)
        green[..., 1] = 255
        m3 = np.stack([mask] * 3, axis=-1)
        overlay = np.where(m3, (overlay * (1 - alpha) + green * alpha).astype(np.uint8), overlay)

    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    r = max(3, int(round(min(h, w) / 250)))
    for points in points_by_box:
        for x, y in points:
            cv2.circle(bgr, (int(round(x)), int(round(y))), r, (0, 0, 255), -1, cv2.LINE_AA)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), max(2, int(round(min(h, w) / 400))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)
