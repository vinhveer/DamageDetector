from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .geometry import clip_box
from .models import Box, Detection


def _read_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def _component_mask(crop_bgr: np.ndarray, *, prompt_key: str) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    height, width = gray.shape[:2]
    if height <= 2 or width <= 2:
        return np.zeros_like(gray, dtype=np.uint8)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    # Local contrast catches dark stains/cracks even when global lighting changes.
    local_dark = gray.astype(np.int16) < (blur.astype(np.int16) - 10)
    dark_global = gray <= max(0, int(np.percentile(gray, 38)))
    sat_high = sat >= max(24, int(np.percentile(sat, 68)))
    edges = cv2.Canny(gray, 40, 120)

    key = str(prompt_key or "").lower()
    if key == "crack":
        mask = (local_dark | dark_global) & (cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
        return (mask_u8 > 0).astype(np.uint8)

    b, g, r = cv2.split(crop_bgr)
    greenish = (g.astype(np.int16) > r.astype(np.int16) + 6) & (g.astype(np.int16) > b.astype(np.int16) - 8)
    if key == "mold":
        mask = dark_global | local_dark | sat_high | greenish
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        return (mask_u8 > 0).astype(np.uint8)

    # Spall/broken/peeling tends to be a local texture or material-loss component.
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    rough = gradient >= max(16, int(np.percentile(gradient, 78)))
    mask = local_dark | dark_global | rough | (edges > 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (mask_u8 > 0).astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _split_large_component(
    mask: np.ndarray,
    *,
    max_side_px: int,
    min_component_area: int,
) -> list[tuple[int, int, int, int]]:
    tight = _bbox_from_mask(mask)
    if tight is None:
        return []
    x1, y1, x2, y2 = tight
    width = x2 - x1
    height = y2 - y1
    if width <= max_side_px and height <= max_side_px:
        return [tight]

    stride = max(8, int(max_side_px))
    boxes: list[tuple[int, int, int, int]] = []
    for ty in range(y1, y2, stride):
        for tx in range(x1, x2, stride):
            wx2 = min(x2, tx + max_side_px)
            wy2 = min(y2, ty + max_side_px)
            crop = mask[ty:wy2, tx:wx2]
            if int(crop.sum()) < int(min_component_area):
                continue
            local = _bbox_from_mask(crop)
            if local is None:
                continue
            lx1, ly1, lx2, ly2 = local
            boxes.append((tx + lx1, ty + ly1, tx + lx2, ty + ly2))
    return boxes


def refine_detection_by_visual_components(
    *,
    image_path: Path,
    detection: Detection,
    image_width: int,
    image_height: int,
    max_side_ratio: float,
    max_area_ratio: float,
    min_component_area: int,
    pad: int = 3,
) -> list[Detection]:
    box = clip_box(detection.box, width=int(image_width), height=int(image_height))
    if box is None:
        return []
    max_side_px = max(24, int(round(max(int(image_width), int(image_height)) * float(max_side_ratio))))
    max_area_px = max(1.0, float(image_width * image_height) * float(max_area_ratio))
    should_refine = box.width > max_side_px or box.height > max_side_px or box.area > max_area_px
    if not should_refine and detection.stage == "refine":
        # Still tighten refine boxes when a compact visual component is obvious.
        should_refine = True
    if not should_refine:
        return [detection]

    image = _read_bgr(image_path)
    x1, y1, x2, y2 = box.as_int_xyxy()
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    mask = _component_mask(crop, prompt_key=detection.prompt_key)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    refined: list[Detection] = []
    for label_idx in range(1, int(labels_count)):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        component = (labels == label_idx).astype(np.uint8)
        for lx1, ly1, lx2, ly2 in _split_large_component(
            component,
            max_side_px=int(max_side_px),
            min_component_area=int(min_component_area),
        ):
            candidate = clip_box(
                Box(
                    x1=float(x1 + lx1 - pad),
                    y1=float(y1 + ly1 - pad),
                    x2=float(x1 + lx2 + pad),
                    y2=float(y1 + ly2 + pad),
                ),
                width=int(image_width),
                height=int(image_height),
            )
            if candidate is None:
                continue
            if candidate.area < int(min_component_area):
                continue
            refined.append(
                replace(
                    detection,
                    box=candidate,
                    score=float(detection.score),
                    raw={
                        **dict(detection.raw or {}),
                        "visual_refined_from": detection.box.as_xyxy(),
                        "visual_component_area": int(area),
                    },
                )
            )
    if refined:
        return refined

    # If a large DINO box cannot be split into visual components, drop it instead of
    # keeping a huge low-precision box.
    if box.width > max_side_px or box.height > max_side_px or box.area > max_area_px:
        return []
    return [detection]
