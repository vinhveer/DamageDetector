from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

from object_detection.dinov2.dinov2_classifier import DinoV2ClassifierRunner, default_dinov2_checkpoint
from object_detection.dinov2.dinov2_prototypes import DinoV2PrototypeRunner, default_dinov2_embedding_checkpoint
from torch_runtime import describe_device_fallback, get_torch, select_device_str

_RECURSIVE_TILE_SIZE = 512
_RECURSIVE_TILE_OVERLAP = 64
_RECURSIVE_MIN_VALID_COVERAGE = 0.30
_RECURSIVE_MEDIUM_TILE_SIZE = 1024
_RECURSIVE_MEDIUM_TILE_OVERLAP = 160
_RECURSIVE_MEDIUM_MIN_VALID_COVERAGE = 0.15
_RECURSIVE_LARGE_TILE_SIZE = 1536
_RECURSIVE_LARGE_TILE_OVERLAP = 256
_RECURSIVE_LARGE_MIN_VALID_COVERAGE = 0.05
_RECURSIVE_MIN_REFINED_COVERAGE = 0.35
_RECURSIVE_TILE_CONTEXT_PAD = 16
_RECURSIVE_MIN_REFINED_WIDTH = 96
_RECURSIVE_MIN_REFINED_HEIGHT = 96
_RECURSIVE_MIN_COMPONENT_AREA = 1024
_DINO_BOX_BLACK_RATIO_REJECT = 0.40
_DINO_BLACK_PIXEL_THRESHOLD = 12


@dataclass(frozen=True)
class DinoParams:
    gdino_checkpoint: str
    gdino_config_id: str = "auto"
    text_queries: Sequence[str] = ("crack",)
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    max_dets: int = 20
    device: str = "auto"
    output_dir: str = "results_dino"
    roi_box: tuple[int, int, int, int] | None = None
    nms_iou_threshold: float = 0.5
    parent_contain_threshold: float = 0.7
    recursive_min_box_px: int = 48
    recursive_max_depth: int = 3
    # Which tile passes to run in recursive/tiled mode.
    # Supported: small, medium, large
    recursive_tile_scales: Sequence[str] = ("small", "medium", "large")
    top_k: int = 1
    crop_dirname: str = "dino_crops"
    dinov2_checkpoint: str = ""
    classifier_batch_size: int = 8
    classifier_top_k_labels: int = 3
    classifier_min_confidence: float = 0.0
    classifier_map_path: str = ""
    classifier_strict: bool = False
    prototype_dir: str = ""
    prototype_batch_size: int = 8
    prototype_top_k_labels: int = 3
    prototype_min_similarity: float = 0.3
    prototype_background_labels: Sequence[str] = ("background", "negative", "other", "none")
    prototype_strict: bool = True
    save_overlay: bool = False
    overlay_filename: str = "overlay_filtered.png"
    overlay_include_rejected: bool = False
    # Optional physical scale: how many millimeters correspond to 1 pixel.
    # If > 0, overlay rendering will include box area in mm^2/cm^2.
    mm_per_px: float = 0.0


@dataclass(frozen=True)
class Det:
    label: str
    box_xyxy: Any
    score: float


def normalize_queries(text_queries: Sequence[str]) -> List[str]:
    queries = [q.strip() for q in text_queries if q.strip()]
    seen = set()
    deduped: List[str] = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped


def _canonical_label(text: str) -> str:
    return " ".join(str(text or "").replace("_", " ").replace("-", " ").strip().lower().split())


def label_matches(label: str, targets: Sequence[str]) -> bool:
    lowered = _canonical_label(label)
    for target in targets:
        token = _canonical_label(target)
        if token and token in lowered:
            return True
    return False


def post_process_gdino(
    processor: Any,
    outputs: Any,
    input_ids: Any,
    box_threshold: float,
    text_threshold: float,
    target_sizes: Any,
) -> Any:
    fn = getattr(processor, "post_process_grounded_object_detection", None)
    if fn is None:
        raise RuntimeError(
            "Processor does not expose post_process_grounded_object_detection; "
            "use a compatible transformers version/model bundle."
        )
    signature = inspect.signature(fn)
    kwargs = {
        "outputs": outputs,
        "input_ids": input_ids,
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in signature.parameters:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
    return fn(**kwargs)


def run_text_boxes(
    *,
    processor: Any,
    gdino: Any,
    device: str,
    pil_image: Any,
    text_queries: Sequence[str],
    box_threshold: float,
    text_threshold: float,
) -> List[Det]:
    import numpy as np

    torch = get_torch()

    queries = normalize_queries(text_queries)
    if not queries:
        return []
    width, height = pil_image.size
    caption = ". ".join(queries)
    with torch.no_grad():
        inputs = processor(images=pil_image, text=caption, return_tensors="pt").to(device)
        outputs = gdino(**inputs)
        target_sizes = torch.tensor([[height, width]], device=device)
        processed = post_process_gdino(
            processor=processor,
            outputs=outputs,
            input_ids=inputs["input_ids"],
            target_sizes=target_sizes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    if not processed:
        return []
    p0 = processed[0]
    boxes = p0["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = p0["scores"].detach().cpu().numpy().astype(np.float32)
    labels = p0.get("labels", [])
    return [Det(label=str(label), box_xyxy=box.astype(np.float32), score=float(score)) for box, score, label in zip(boxes, scores, labels)]


def _pad_rgb_for_grounding_dino(rgb: Any, *, min_side: int = 256, square_if_aspect_over: float = 8.0) -> Any:
    import numpy as np

    height, width = rgb.shape[:2]
    short_side = max(1, min(height, width))
    long_side = max(height, width)

    target_h = int(height)
    target_w = int(width)

    if short_side < int(min_side):
        if height <= width:
            target_h = max(target_h, int(min_side))
        else:
            target_w = max(target_w, int(min_side))

    if float(long_side) / float(short_side) >= float(square_if_aspect_over):
        side = max(long_side, int(min_side))
        target_h = max(target_h, int(side))
        target_w = max(target_w, int(side))

    if target_h == height and target_w == width:
        return rgb

    canvas = np.zeros((target_h, target_w, rgb.shape[2]), dtype=rgb.dtype)
    canvas[:height, :width] = rgb
    return canvas


def _box_iou(a: Any, b: Any) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _nms_boxes(boxes_info: list[tuple[Any, str, float]], iou_threshold: float = 0.5) -> list[tuple[Any, str, float]]:
    if not boxes_info:
        return []
    boxes_info = sorted(boxes_info, key=lambda x: x[2], reverse=True)
    used = [False] * len(boxes_info)
    kept = []
    for i, (box_i, label_i, score_i) in enumerate(boxes_info):
        if used[i]:
            continue
        kept.append((box_i, label_i, score_i))
        for j in range(i + 1, len(boxes_info)):
            if used[j]:
                continue
            if _box_iou(box_i, boxes_info[j][0]) > iou_threshold:
                used[j] = True
    return kept


def _filter_parent_boxes(boxes_info: list[tuple[Any, str, float]], contain_thresh: float = 0.7) -> list[tuple[Any, str, float]]:
    import numpy as np

    count = len(boxes_info)
    if count <= 1:
        return boxes_info

    def _area(box: Any) -> float:
        return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))

    def _intersection(a: Any, b: Any) -> float:
        ix1 = max(float(a[0]), float(b[0]))
        iy1 = max(float(a[1]), float(b[1]))
        ix2 = min(float(a[2]), float(b[2]))
        iy2 = min(float(a[3]), float(b[3]))
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    is_parent = [False] * count
    for i in range(count):
        box_i, _, _ = boxes_info[i]
        area_i = _area(box_i)
        if area_i <= 0:
            continue
        for j in range(count):
            if i == j or is_parent[i]:
                break
            box_j, _, _ = boxes_info[j]
            area_j = _area(box_j)
            if area_j <= 0 or area_j >= area_i:
                continue
            inter = _intersection(box_i, box_j)
            if area_j > 0 and inter / area_j >= contain_thresh:
                is_parent[i] = True
    return [boxes_info[i] for i in range(count) if not is_parent[i]]


def _tile_positions(start: int, end: int, tile_size: int, overlap: int) -> list[int]:
    span = max(0, int(end) - int(start))
    if span <= tile_size:
        return [int(start)]
    step = max(1, int(tile_size) - int(overlap))
    positions: list[int] = []
    pos = int(start)
    last = int(end) - int(tile_size)
    while pos < last:
        positions.append(pos)
        pos += step
    positions.append(last)
    deduped: list[int] = []
    seen = set()
    for value in positions:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _mask_bbox(mask01: Any) -> tuple[int, int, int, int] | None:
    import numpy as np

    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _normalize_band_to_uint8(band: Any, valid_mask: Any) -> Any:
    import numpy as np

    band_f = band.astype(np.float32, copy=False)
    valid_values = band_f[valid_mask > 0]
    if valid_values.size == 0:
        valid_values = band_f.reshape(-1)
    lo = float(np.percentile(valid_values, 2.0))
    hi = float(np.percentile(valid_values, 98.0))
    if hi <= lo:
        lo = float(valid_values.min())
        hi = float(valid_values.max())
    if hi <= lo:
        return np.zeros_like(band_f, dtype=np.uint8)
    scaled = np.clip((band_f - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _load_tiff_with_rasterio(image_path: str, *, log_fn=None) -> tuple[Any, Any] | None:
    import numpy as np

    try:
        import rasterio
    except Exception as exc:
        if log_fn is not None:
            log_fn(f"Valid-mask fallback: rasterio unavailable ({exc}). Using OpenCV heuristic.")
        return None

    try:
        with rasterio.open(image_path) as dataset:
            valid_mask = dataset.dataset_mask() > 0
            bands = dataset.read()
    except Exception as exc:
        if log_fn is not None:
            log_fn(f"Valid-mask fallback: cannot read raster mask ({exc}). Using OpenCV heuristic.")
        return None

    if bands.ndim != 3 or bands.shape[0] <= 0:
        if log_fn is not None:
            log_fn("Valid-mask fallback: TIFF bands shape unsupported. Using OpenCV heuristic.")
        return None

    if bands.shape[0] >= 3:
        chosen = bands[:3]
    else:
        chosen = np.repeat(bands[:1], 3, axis=0)
    rgb = np.stack([_normalize_band_to_uint8(chosen[idx], valid_mask) for idx in range(3)], axis=-1)
    return rgb, valid_mask.astype(bool)


def _build_opencv_valid_mask(image_rgb: Any) -> Any:
    import cv2
    import numpy as np

    dark_mask = (np.max(image_rgb[:, :, :3], axis=2) <= _DINO_BLACK_PIXEL_THRESHOLD).astype(np.uint8)
    if int(dark_mask.sum()) == 0:
        return np.ones(image_rgb.shape[:2], dtype=bool)
    _, labels = cv2.connectedComponents(dark_mask, connectivity=8)
    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    invalid_mask = np.isin(labels, border_labels) & (dark_mask > 0)
    valid_mask = (~invalid_mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    valid_mask = cv2.morphologyEx(valid_mask * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
    return valid_mask > 0


def _build_valid_mask(image_path: str, image_rgb: Any, *, log_fn=None) -> tuple[Any, Any, str]:
    lower = str(image_path or "").lower()
    if lower.endswith((".tif", ".tiff")):
        raster_data = _load_tiff_with_rasterio(image_path, log_fn=log_fn)
        if raster_data is not None:
            rgb, valid_mask = raster_data
            return rgb, valid_mask, "rasterio_dataset_mask"
    return image_rgb, _build_opencv_valid_mask(image_rgb), "opencv_valid_mask"


def _rotate_image_and_mask(image_rgb: Any, valid_mask: Any, angle_deg: float) -> tuple[Any, Any, Any, Any]:
    import cv2
    import numpy as np

    height, width = image_rgb.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_v = abs(float(matrix[0, 0]))
    sin_v = abs(float(matrix[0, 1]))
    bound_w = int(round((height * sin_v) + (width * cos_v)))
    bound_h = int(round((height * cos_v) + (width * sin_v)))
    matrix[0, 2] += (bound_w / 2.0) - center[0]
    matrix[1, 2] += (bound_h / 2.0) - center[1]
    rotated_rgb = cv2.warpAffine(
        image_rgb,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rotated_mask = cv2.warpAffine(
        valid_mask.astype(np.uint8) * 255,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    return rotated_rgb, rotated_mask, matrix, cv2.invertAffineTransform(matrix)


def _compute_roi_occupancy(valid_mask: Any, roi_box: tuple[int, int, int, int] | None) -> float:
    if roi_box is None:
        return 0.0
    x1, y1, x2, y2 = roi_box
    area = max(1, int(x2 - x1) * int(y2 - y1))
    valid_area = int(valid_mask[y1:y2, x1:x2].sum())
    return float(valid_area) / float(area)


def _compute_oriented_roi(image_rgb: Any, valid_mask: Any, *, log_fn=None) -> dict[str, Any]:
    import cv2
    import numpy as np

    original_roi = _mask_bbox(valid_mask)
    if original_roi is None:
        return {
            "rgb": image_rgb,
            "valid_mask": valid_mask,
            "roi_box": None,
            "rotation_angle": 0.0,
            "inverse_matrix": None,
            "rotated": False,
        }

    contours, _ = cv2.findContours((valid_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "rgb": image_rgb,
            "valid_mask": valid_mask,
            "roi_box": original_roi,
            "rotation_angle": 0.0,
            "inverse_matrix": None,
            "rotated": False,
        }

    contour = max(contours, key=cv2.contourArea)
    (_, _), (rect_w, rect_h), angle = cv2.minAreaRect(contour)
    if rect_w <= 1 or rect_h <= 1:
        return {
            "rgb": image_rgb,
            "valid_mask": valid_mask,
            "roi_box": original_roi,
            "rotation_angle": 0.0,
            "inverse_matrix": None,
            "rotated": False,
        }

    deskew_angle = float(angle)
    if rect_w < rect_h:
        deskew_angle += 90.0
    while deskew_angle <= -45.0:
        deskew_angle += 90.0
    while deskew_angle > 45.0:
        deskew_angle -= 90.0

    original_occupancy = _compute_roi_occupancy(valid_mask, original_roi)
    if abs(deskew_angle) < 1.0:
        return {
            "rgb": image_rgb,
            "valid_mask": valid_mask,
            "roi_box": original_roi,
            "rotation_angle": 0.0,
            "inverse_matrix": None,
            "rotated": False,
        }

    rotated_rgb, rotated_mask, _matrix, inverse_matrix = _rotate_image_and_mask(image_rgb, valid_mask, deskew_angle)
    rotated_roi = _mask_bbox(rotated_mask)
    rotated_occupancy = _compute_roi_occupancy(rotated_mask, rotated_roi)

    if rotated_roi is None or rotated_occupancy <= (original_occupancy + 0.05):
        return {
            "rgb": image_rgb,
            "valid_mask": valid_mask,
            "roi_box": original_roi,
            "rotation_angle": 0.0,
            "inverse_matrix": None,
            "rotated": False,
        }

    if log_fn is not None:
        log_fn(
            f"Oriented ROI: angle={deskew_angle:.2f}deg "
            f"occupancy {original_occupancy:.3f}->{rotated_occupancy:.3f}"
        )

    return {
        "rgb": rotated_rgb,
        "valid_mask": rotated_mask,
        "roi_box": rotated_roi,
        "rotation_angle": deskew_angle,
        "inverse_matrix": inverse_matrix,
        "rotated": True,
    }


def _generate_valid_tiles(
    valid_mask: Any,
    roi_box: tuple[int, int, int, int],
    *,
    tile_size: int,
    overlap: int,
    min_valid_coverage: float,
    allow_refine: bool = True,
) -> tuple[list[tuple[int, int, int, int, float, str]], int, int, int]:
    import cv2

    x1, y1, x2, y2 = roi_box
    integral = cv2.integral(valid_mask.astype("uint8"))

    def _coverage(ax1: int, ay1: int, ax2: int, ay2: int) -> float:
        valid_pixels = int(integral[ay2, ax2] - integral[ay1, ax2] - integral[ay2, ax1] + integral[ay1, ax1])
        area = max(1, int(ax2 - ax1) * int(ay2 - ay1))
        return float(valid_pixels) / float(area)

    def _expand_box_within_bounds(
        box: tuple[int, int, int, int],
        *,
        bounds: tuple[int, int, int, int],
        pad: int,
        min_width: int,
        min_height: int,
    ) -> tuple[int, int, int, int]:
        bx1, by1, bx2, by2 = [int(v) for v in box]
        min_x, min_y, max_x, max_y = [int(v) for v in bounds]
        bx1 = max(min_x, bx1 - int(pad))
        by1 = max(min_y, by1 - int(pad))
        bx2 = min(max_x, bx2 + int(pad))
        by2 = min(max_y, by2 + int(pad))
        width = bx2 - bx1
        height = by2 - by1
        if width < int(min_width):
            grow = int(min_width) - width
            left = grow // 2
            right = grow - left
            bx1 = max(min_x, bx1 - left)
            bx2 = min(max_x, bx2 + right)
            if (bx2 - bx1) < int(min_width):
                deficit = int(min_width) - (bx2 - bx1)
                if bx1 == min_x:
                    bx2 = min(max_x, bx2 + deficit)
                else:
                    bx1 = max(min_x, bx1 - deficit)
        if height < int(min_height):
            grow = int(min_height) - height
            top = grow // 2
            bottom = grow - top
            by1 = max(min_y, by1 - top)
            by2 = min(max_y, by2 + bottom)
            if (by2 - by1) < int(min_height):
                deficit = int(min_height) - (by2 - by1)
                if by1 == min_y:
                    by2 = min(max_y, by2 + deficit)
                else:
                    by1 = max(min_y, by1 - deficit)
        return int(bx1), int(by1), int(bx2), int(by2)

    def _fit_crop_to_valid_bbox(tile_box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        tile_x1, tile_y1, tile_x2, tile_y2 = tile_box
        submask = valid_mask[tile_y1:tile_y2, tile_x1:tile_x2]
        if submask.size == 0 or int(submask.sum()) == 0:
            return None
        local_bbox = _mask_bbox(submask)
        if local_bbox is None:
            return None
        local_x1, local_y1, local_x2, local_y2 = local_bbox
        return _expand_box_within_bounds(
            (tile_x1 + local_x1, tile_y1 + local_y1, tile_x1 + local_x2, tile_y1 + local_y2),
            bounds=tile_box,
            pad=_RECURSIVE_TILE_CONTEXT_PAD,
            min_width=_RECURSIVE_MIN_REFINED_WIDTH,
            min_height=_RECURSIVE_MIN_REFINED_HEIGHT,
        )

    def _refine_tile_boxes(tile_box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
        tile_x1, tile_y1, tile_x2, tile_y2 = tile_box
        submask = valid_mask[tile_y1:tile_y2, tile_x1:tile_x2]
        if submask.size == 0 or int(submask.sum()) == 0:
            return []
        tile_h, tile_w = submask.shape[:2]
        component_area_thresh = max(_RECURSIVE_MIN_COMPONENT_AREA, int(tile_h * tile_w * 0.01))
        labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(submask.astype("uint8"), connectivity=8)
        refined: list[tuple[int, int, int, int]] = []
        for label_idx in range(1, int(labels_count)):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            if area < component_area_thresh and max(width, height) < max(_RECURSIVE_MIN_REFINED_WIDTH, _RECURSIVE_MIN_REFINED_HEIGHT):
                continue
            local_x1 = int(stats[label_idx, cv2.CC_STAT_LEFT])
            local_y1 = int(stats[label_idx, cv2.CC_STAT_TOP])
            local_x2 = local_x1 + width
            local_y2 = local_y1 + height
            refined.append(
                _expand_box_within_bounds(
                    (tile_x1 + local_x1, tile_y1 + local_y1, tile_x1 + local_x2, tile_y1 + local_y2),
                    bounds=tile_box,
                    pad=_RECURSIVE_TILE_CONTEXT_PAD,
                    min_width=_RECURSIVE_MIN_REFINED_WIDTH,
                    min_height=_RECURSIVE_MIN_REFINED_HEIGHT,
                )
            )
        if refined:
            deduped: list[tuple[int, int, int, int]] = []
            seen = set()
            for candidate in sorted(refined, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]), reverse=True):
                key = tuple(int(v) for v in candidate)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(candidate)
            return deduped
        fitted = _fit_crop_to_valid_bbox(tile_box)
        return [fitted] if fitted is not None else []

    tiles: list[tuple[int, int, int, int, float, str]] = []
    x_positions = _tile_positions(x1, x2, tile_size, overlap)
    y_positions = _tile_positions(y1, y2, tile_size, overlap)
    total_tiles = len(x_positions) * len(y_positions)
    skipped_tiles = 0
    refined_tiles = 0
    seen_tiles: set[tuple[int, int, int, int]] = set()
    for tile_y1 in y_positions:
        for tile_x1 in x_positions:
            tile_x2 = min(x2, tile_x1 + tile_size)
            tile_y2 = min(y2, tile_y1 + tile_size)
            coverage = _coverage(tile_x1, tile_y1, tile_x2, tile_y2)
            if coverage < float(min_valid_coverage):
                if not allow_refine:
                    skipped_tiles += 1
                    continue
                refined_candidates = _refine_tile_boxes((tile_x1, tile_y1, tile_x2, tile_y2))
                kept_refined = 0
                for refined_x1, refined_y1, refined_x2, refined_y2 in refined_candidates:
                    refined_coverage = _coverage(refined_x1, refined_y1, refined_x2, refined_y2)
                    if refined_coverage < _RECURSIVE_MIN_REFINED_COVERAGE:
                        continue
                    key = (int(refined_x1), int(refined_y1), int(refined_x2), int(refined_y2))
                    if key in seen_tiles:
                        continue
                    seen_tiles.add(key)
                    tiles.append((refined_x1, refined_y1, refined_x2, refined_y2, refined_coverage, "refined"))
                    refined_tiles += 1
                    kept_refined += 1
                if kept_refined == 0:
                    skipped_tiles += 1
                continue
            if allow_refine:
                fitted_full = _fit_crop_to_valid_bbox((tile_x1, tile_y1, tile_x2, tile_y2))
                if fitted_full is not None:
                    fit_x1, fit_y1, fit_x2, fit_y2 = fitted_full
                    fit_coverage = _coverage(fit_x1, fit_y1, fit_x2, fit_y2)
                    tile_kind = "trimmed" if (fit_x1, fit_y1, fit_x2, fit_y2) != (tile_x1, tile_y1, tile_x2, tile_y2) else "full"
                else:
                    fit_x1, fit_y1, fit_x2, fit_y2 = tile_x1, tile_y1, tile_x2, tile_y2
                    fit_coverage = coverage
                    tile_kind = "full"
            else:
                fit_x1, fit_y1, fit_x2, fit_y2 = tile_x1, tile_y1, tile_x2, tile_y2
                fit_coverage = coverage
                tile_kind = "full"
            key = (int(fit_x1), int(fit_y1), int(fit_x2), int(fit_y2))
            if key in seen_tiles:
                continue
            seen_tiles.add(key)
            tiles.append((fit_x1, fit_y1, fit_x2, fit_y2, fit_coverage, tile_kind))
    return tiles, total_tiles, skipped_tiles, refined_tiles


def _map_box_from_rotated_to_original(box_xyxy: Any, inverse_matrix: Any, *, original_width: int, original_height: int) -> Any:
    import cv2
    import numpy as np

    if inverse_matrix is None:
        return box_xyxy.astype(np.float32)
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    points = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    mapped = cv2.transform(points, inverse_matrix)[0]
    mapped[:, 0] = np.clip(mapped[:, 0], 0.0, max(0.0, float(original_width - 1)))
    mapped[:, 1] = np.clip(mapped[:, 1], 0.0, max(0.0, float(original_height - 1)))
    return np.array(
        [
            float(mapped[:, 0].min()),
            float(mapped[:, 1].min()),
            float(mapped[:, 0].max()),
            float(mapped[:, 1].max()),
        ],
        dtype=np.float32,
    )


def _mask_integral(mask01: Any) -> Any:
    import cv2

    return cv2.integral(mask01.astype("uint8"))


def _pure_black_integral(image_rgb: Any) -> Any:
    import numpy as np

    black_mask = np.max(image_rgb[:, :, :3], axis=2) <= _DINO_BLACK_PIXEL_THRESHOLD
    return _mask_integral(black_mask)


def _box_valid_coverage(box_xyxy: Any, mask_integral: Any, *, width: int, height: int) -> float:
    import math

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    ix1 = max(0, min(width, int(math.floor(x1))))
    iy1 = max(0, min(height, int(math.floor(y1))))
    ix2 = max(0, min(width, int(math.ceil(x2))))
    iy2 = max(0, min(height, int(math.ceil(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    valid_pixels = int(mask_integral[iy2, ix2] - mask_integral[iy1, ix2] - mask_integral[iy2, ix1] + mask_integral[iy1, ix1])
    area = max(1, int(ix2 - ix1) * int(iy2 - iy1))
    return float(valid_pixels) / float(area)


def _box_center_is_valid(box_xyxy: Any, valid_mask: Any) -> bool:
    import numpy as np

    height, width = valid_mask.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    cx = int(np.clip(cx, 0, max(0, width - 1)))
    cy = int(np.clip(cy, 0, max(0, height - 1)))
    return bool(valid_mask[cy, cx])


def _box_is_fully_valid(box_xyxy: Any, mask_integral: Any, *, width: int, height: int) -> bool:
    return _box_valid_coverage(box_xyxy, mask_integral, width=width, height=height) >= 1.0


def _box_has_any_pure_black_pixels(box_xyxy: Any, black_integral: Any, *, width: int, height: int) -> bool:
    return _box_valid_coverage(box_xyxy, black_integral, width=width, height=height) > 0.0


def _box_black_ratio(box_xyxy: Any, black_integral: Any, *, width: int, height: int) -> float:
    return _box_valid_coverage(box_xyxy, black_integral, width=width, height=height)


def default_gdino_checkpoint() -> str:
    return str((Path(__file__).resolve().parent / "models" / "grounding-dino-base").resolve())


def _cv2_imread_any_path(image_path: str, flags: int) -> Any:
    """Read image with OpenCV, supporting non-ASCII Windows paths.

    Some OpenCV builds fail to read Unicode paths via cv2.imread on Windows.
    Fallback to imdecode(fromfile) in that case.
    """
    import cv2

    # On Windows, cv2.imread() frequently fails (and emits a warning) when the
    # path contains non-ASCII characters. Avoid calling it in that case.
    if not (os.name == "nt" and any(ord(ch) > 127 for ch in str(image_path))):
        image = cv2.imread(image_path, flags)
        if image is not None:
            return image
    try:
        import numpy as np

        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _image_max_dim_from_path(image_path: str, *, log_fn=None) -> int:
    import cv2

    image = _cv2_imread_any_path(image_path, cv2.IMREAD_COLOR)
    if image is not None:
        height, width = image.shape[:2]
        return max(int(width), int(height))
    raster_fallback = _load_tiff_with_rasterio(image_path, log_fn=log_fn) if str(image_path).lower().endswith((".tif", ".tiff")) else None
    if raster_fallback is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb, _mask = raster_fallback
    height, width = rgb.shape[:2]
    return max(int(width), int(height))


def _sanitize_crop_box(box_xyxy: Any, *, width: int, height: int) -> tuple[int, int, int, int] | None:
    if box_xyxy is None or len(box_xyxy) != 4:
        return None
    x1 = max(0, min(width, int(round(float(box_xyxy[0])))))
    y1 = max(0, min(height, int(round(float(box_xyxy[1])))))
    x2 = max(0, min(width, int(round(float(box_xyxy[2])))))
    y2 = max(0, min(height, int(round(float(box_xyxy[3])))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _load_classifier_rules(path_or_json: str) -> list[tuple[str, str | None]]:
    source = str(path_or_json or "").strip()
    if not source:
        return []
    if os.path.isfile(source):
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        raw = json.loads(source)

    rules: list[tuple[str, str | None]] = []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            items.append((entry.get("match"), entry.get("label")))
    else:
        raise ValueError("classifier_map_path must be a JSON object, JSON list, or a path to one.")

    for match, label in items:
        key = str(match or "").strip().lower()
        if not key:
            continue
        mapped = None if label is None or str(label).strip() == "" else str(label).strip()
        rules.append((key, mapped))
    return rules


def _default_classifier_rules_for_queries(query_labels: Sequence[str], checkpoint_path: str) -> list[tuple[str, str | None]]:
    checkpoint_name = os.path.basename(str(checkpoint_path or "")).lower()
    normalized_queries = [str(query).strip() for query in normalize_queries(query_labels)]
    crack_only = bool(normalized_queries) and all("crack" in str(query).lower() for query in normalized_queries)
    if crack_only and ("surface_crack" in checkpoint_name or "crack" in checkpoint_name):
        primary = normalized_queries[0] if normalized_queries else "crack"
        return [("positive", primary), ("negative", None)]
    return []


def _lookup_classifier_rule(predicted_label: str, rules: Sequence[tuple[str, str | None]]) -> tuple[bool, str | None]:
    lowered = _canonical_label(predicted_label)
    if not lowered:
        return False, None
    for match, mapped in rules:
        normalized_match = _canonical_label(match)
        if normalized_match in lowered or lowered in normalized_match:
            return True, mapped
    return False, None


def _auto_map_classifier_label(predicted_label: str, queries: Sequence[str]) -> str | None:
    lowered = _canonical_label(predicted_label)
    if not lowered:
        return None
    for query in normalize_queries(queries):
        token = _canonical_label(query)
        if token and (token in lowered or lowered in token):
            return query
    return None


def _resolve_classifier_decision(
    *,
    proposal_label: str,
    predictions: Sequence[dict[str, Any]],
    query_labels: Sequence[str],
    rules: Sequence[tuple[str, str | None]],
    min_confidence: float,
    strict: bool,
) -> dict[str, Any]:
    top_prediction = dict(predictions[0] or {}) if predictions else {}
    top_confidence = float(top_prediction.get("confidence") or 0.0)

    if predictions and top_confidence < float(min_confidence):
        return {
            "accepted": False,
            "final_label": str(proposal_label or ""),
            "matched_classifier_label": None,
            "matched_confidence": top_confidence,
            "reason": "low_classifier_confidence",
        }

    for prediction in predictions:
        predicted_label = str(prediction.get("label") or "")
        predicted_confidence = float(prediction.get("confidence") or 0.0)
        matched, mapped_label = _lookup_classifier_rule(predicted_label, rules)
        if matched:
            if mapped_label is None:
                return {
                    "accepted": False,
                    "final_label": str(proposal_label or ""),
                    "matched_classifier_label": predicted_label,
                    "matched_confidence": predicted_confidence,
                    "reason": "classifier_rule_rejected",
                }
            return {
                "accepted": True,
                "final_label": str(mapped_label),
                "matched_classifier_label": predicted_label,
                "matched_confidence": predicted_confidence,
                "reason": "classifier_rule_mapped",
            }
        auto_label = _auto_map_classifier_label(predicted_label, query_labels)
        if auto_label is not None:
            return {
                "accepted": True,
                "final_label": str(auto_label),
                "matched_classifier_label": predicted_label,
                "matched_confidence": predicted_confidence,
                "reason": "classifier_query_mapped",
            }

    if strict:
        return {
            "accepted": False,
            "final_label": str(proposal_label or ""),
            "matched_classifier_label": None,
            "matched_confidence": top_confidence,
            "reason": "classifier_unmapped",
        }

    return {
        "accepted": True,
        "final_label": str(proposal_label or ""),
        "matched_classifier_label": None,
        "matched_confidence": top_confidence,
        "reason": "kept_original_label",
    }


def _resolve_prototype_decision(
    *,
    proposal_label: str,
    predictions: Sequence[dict[str, Any]],
    min_similarity: float,
    background_labels: Sequence[str],
    strict: bool,
) -> dict[str, Any]:
    top_prediction = dict(predictions[0] or {}) if predictions else {}
    top_similarity = float(top_prediction.get("similarity") or 0.0)
    predicted_label = str(top_prediction.get("label") or "")

    if not predictions:
        if strict:
            return {
                "accepted": False,
                "final_label": str(proposal_label or ""),
                "matched_prototype_label": None,
                "matched_similarity": 0.0,
                "reason": "prototype_no_match",
            }
        return {
            "accepted": True,
            "final_label": str(proposal_label or ""),
            "matched_prototype_label": None,
            "matched_similarity": 0.0,
            "reason": "kept_original_label",
        }

    if label_matches(predicted_label, background_labels):
        return {
            "accepted": False,
            "final_label": str(proposal_label or ""),
            "matched_prototype_label": predicted_label,
            "matched_similarity": top_similarity,
            "reason": "prototype_background_rejected",
        }

    if top_similarity < float(min_similarity):
        if strict:
            return {
                "accepted": False,
                "final_label": str(proposal_label or ""),
                "matched_prototype_label": predicted_label,
                "matched_similarity": top_similarity,
                "reason": "low_prototype_similarity",
            }
        return {
            "accepted": True,
            "final_label": str(proposal_label or ""),
            "matched_prototype_label": predicted_label,
            "matched_similarity": top_similarity,
            "reason": "kept_original_label",
        }

    return {
        "accepted": True,
        "final_label": predicted_label or str(proposal_label or ""),
        "matched_prototype_label": predicted_label or None,
        "matched_similarity": top_similarity,
        "reason": "prototype_mapped",
    }


def _rank_dino_payload(
    detections: Sequence[dict[str, Any]],
    crop_paths: Sequence[str],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked: list[dict[str, Any]] = []
    for index, det in enumerate(detections):
        payload = dict(det)
        payload["crop_path"] = str(crop_paths[index])
        ranked.append(payload)
    ranked.sort(key=lambda item: float(item.get("refined_score") or item.get("score") or 0.0), reverse=True)
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    keep = max(1, int(top_k)) if ranked else 0
    return ranked, ranked[:keep]


class DinoRunner:
    def __init__(self) -> None:
        self._device: str | None = None
        self._gdino_checkpoint: str | None = None
        self._gdino_config_id: str | None = None
        self._processor: Any | None = None
        self._gdino: Any | None = None
        self._dinov2_runner = DinoV2ClassifierRunner()
        self._prototype_runner = DinoV2PrototypeRunner()

    def _ensure_import_paths(self) -> None:
        here = Path(__file__).resolve()
        repo_root = here.parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _load_gdino_state_dict(self, checkpoint_path: str) -> dict:
        torch = get_torch()

        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(checkpoint_path)
        try:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            if isinstance(raw.get("state_dict"), dict):
                state = raw["state_dict"]
            elif isinstance(raw.get("model"), dict):
                state = raw["model"]
            else:
                state = raw
        else:
            raise TypeError(f"Unsupported GroundingDINO checkpoint format: {type(raw)} ({checkpoint_path})")
        if not isinstance(state, dict):
            raise TypeError(f"GroundingDINO checkpoint state_dict invalid: {type(state)} ({checkpoint_path})")
        state = self._strip_prefix_if_present(state, "module.")
        state = self._strip_prefix_if_present(state, "model.")
        return state

    def _strip_prefix_if_present(self, state: dict, prefix: str) -> dict:
        keys = list(state.keys())
        if not keys:
            return state
        count = sum(1 for key in keys if str(key).startswith(prefix))
        if count >= int(len(keys) * 0.9):
            return {str(key)[len(prefix) :] if str(key).startswith(prefix) else key: value for key, value in state.items()}
        return state

    def _resolve_gdino_config_id(self, params: DinoParams) -> str:
        config_id = str(params.gdino_config_id or "").strip()
        if not config_id or config_id.lower() == "auto":
            name = os.path.basename(str(params.gdino_checkpoint or "")).lower()
            if "swint" in name or "tiny" in name:
                return "IDEA-Research/grounding-dino-tiny"
            return "IDEA-Research/grounding-dino-base"
        return config_id

    def ensure_model_loaded(self, params: DinoParams, *, log_fn=None) -> tuple[Any, Any, str]:
        self._ensure_import_paths()
        import faulthandler
        import sys
        import types

        def _dump_on_hang(label: str, seconds: float) -> threading.Event:
            stop = threading.Event()

            def _arm() -> None:
                if stop.wait(seconds):
                    return
                try:
                    if log_fn is not None:
                        log_fn(f"WARN: '{label}' still running after {int(seconds)}s. Dumping stack traces...")
                except Exception:
                    pass
                try:
                    faulthandler.dump_traceback(all_threads=True)
                except Exception:
                    pass

            threading.Thread(target=_arm, name=f"hang-dump:{label}", daemon=True).start()
            return stop

        def _start_log_tick(label: str, *, every_s: float = 10.0) -> threading.Event:
            stop = threading.Event()

            def _tick() -> None:
                if log_fn is None:
                    return
                start = time.time()
                while not stop.wait(every_s):
                    elapsed = int(time.time() - start)
                    log_fn(f"Still {label}... ({elapsed}s)")

            if log_fn is not None:
                threading.Thread(target=_tick, name=f"tick:{label}", daemon=True).start()
            return stop

        # Emit a log line before importing heavy deps so users aren't left with a "hang"
        # when transformers/torch import is slow on Windows.
        if log_fn is not None:
            log_fn("Step: import transformers + grounding_dino...")
        tick = _start_log_tick("importing transformers", every_s=10.0)
        hang = _dump_on_hang("import transformers", 180.0)
        import transformers  # type: ignore
        import transformers.models.grounding_dino.image_processing_grounding_dino  # noqa: F401
        from transformers import AutoProcessor, GroundingDinoConfig, GroundingDinoForObjectDetection  # type: ignore
        hang.set()
        tick.set()

        device = select_device_str(params.device)
        fallback = describe_device_fallback(params.device, device)
        if fallback is not None and log_fn is not None:
            log_fn(fallback)

        checkpoint_path = str(params.gdino_checkpoint or "").strip()
        if not checkpoint_path:
            raise FileNotFoundError("GroundingDINO checkpoint path is required.")

        checkpoint_lower = checkpoint_path.lower()
        checkpoint_is_dir = os.path.isdir(checkpoint_path)
        checkpoint_is_file = os.path.isfile(checkpoint_path)
        checkpoint_is_explicit_file = checkpoint_lower.endswith((".pth", ".pt", ".safetensors", ".bin"))
        use_hf_id = (not checkpoint_is_dir) and (not checkpoint_is_file) and (not checkpoint_is_explicit_file)

        if checkpoint_is_file and not use_hf_id:
            parent_dir = os.path.dirname(checkpoint_path)
            if os.path.isfile(os.path.join(parent_dir, "config.json")):
                base = os.path.basename(checkpoint_path).lower()
                if base in {"model.safetensors", "pytorch_model.bin", "tf_model.h5"}:
                    if log_fn is not None:
                        log_fn(f"Detected valid HF model folder ({parent_dir}). Using folder mode instead of single file.")
                    checkpoint_path = parent_dir
                    checkpoint_is_dir = True
                    checkpoint_is_file = False
                    use_hf_id = False

        if use_hf_id or checkpoint_is_dir:
            config_id = checkpoint_path
        else:
            parent_dir = os.path.dirname(checkpoint_path)
            if os.path.isfile(os.path.join(parent_dir, "config.json")):
                config_id = parent_dir
            else:
                config_id = self._resolve_gdino_config_id(params)

        needs_reload = (
            self._gdino is None
            or self._processor is None
            or self._gdino_checkpoint != checkpoint_path
            or self._gdino_config_id != config_id
            or self._device != device
        )
        if not needs_reload:
            return self._processor, self._gdino, device

        if log_fn is not None:
            if checkpoint_is_dir:
                log_fn("Loading GroundingDINO from local folder...")
            elif use_hf_id:
                log_fn("Loading GroundingDINO (offline/cache)...")
            else:
                log_fn("Loading GroundingDINO (offline config/cache + local .pth weights)...")

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        def _is_truthy_env(name: str) -> bool:
            value = str(os.environ.get(name, "")).strip().lower()
            return value in {"1", "true", "yes", "on"}

        # Offline-first by default, but allow online downloads when the user explicitly sets:
        # `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` in the parent shell.
        local_files_only = _is_truthy_env("HF_HUB_OFFLINE") or _is_truthy_env("TRANSFORMERS_OFFLINE")

        tick_stop = threading.Event()

        def _tick() -> None:
            if log_fn is None:
                return
            start = time.time()
            while not tick_stop.wait(8.0):
                elapsed = int(time.time() - start)
                log_fn(f"Still loading GroundingDINO... ({elapsed}s)")

        tick_thread = None
        if log_fn is not None:
            tick_thread = threading.Thread(target=_tick, name="gdino-load-tick", daemon=True)
            tick_thread.start()

        try:
            if log_fn is not None:
                log_fn(f"GroundingDINO: device={device}")
                log_fn(f"GroundingDINO: checkpoint={checkpoint_path}")
                log_fn(f"GroundingDINO: config={config_id}")
                log_fn(f"GroundingDINO: local_files_only={local_files_only} (HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE','')}, TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE','')})")
                if checkpoint_is_file:
                    try:
                        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                        log_fn(f"GroundingDINO: weights_size={size_mb:.1f} MB")
                    except Exception:
                        pass
            if checkpoint_is_dir:
                if log_fn is not None:
                    log_fn("Step: load processor (AutoProcessor.from_pretrained folder)...")
                start = time.time()
                hang = _dump_on_hang("AutoProcessor.from_pretrained(folder)", 30.0)
                processor = AutoProcessor.from_pretrained(checkpoint_path, local_files_only=local_files_only)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded processor ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: load model weights (from_pretrained folder)...")
                start = time.time()
                use_safetensors = os.path.isfile(os.path.join(checkpoint_path, "model.safetensors"))
                hang = _dump_on_hang("GroundingDinoForObjectDetection.from_pretrained(folder)", 60.0)
                try:
                    gdino = GroundingDinoForObjectDetection.from_pretrained(
                        checkpoint_path,
                        local_files_only=local_files_only,
                        use_safetensors=use_safetensors,
                    )
                except TypeError:
                    gdino = GroundingDinoForObjectDetection.from_pretrained(checkpoint_path, local_files_only=local_files_only)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded model from folder ({time.time() - start:.1f}s)")
            elif use_hf_id:
                if log_fn is not None:
                    log_fn("Step: load processor (AutoProcessor.from_pretrained cache)...")
                start = time.time()
                hang = _dump_on_hang("AutoProcessor.from_pretrained(cache)", 30.0)
                processor = AutoProcessor.from_pretrained(checkpoint_path, local_files_only=local_files_only)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded processor ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: load model weights (from_pretrained cache)...")
                start = time.time()
                hang = _dump_on_hang("GroundingDinoForObjectDetection.from_pretrained(cache)", 60.0)
                try:
                    gdino = GroundingDinoForObjectDetection.from_pretrained(checkpoint_path, local_files_only=local_files_only)
                except TypeError:
                    gdino = GroundingDinoForObjectDetection.from_pretrained(checkpoint_path, local_files_only=local_files_only)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded model from cache ({time.time() - start:.1f}s)")
            else:
                if not checkpoint_is_file:
                    raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")
                if log_fn is not None:
                    log_fn("Step: load .pth state_dict (torch.load)...")
                start = time.time()
                state_dict = self._load_gdino_state_dict(checkpoint_path)
                if log_fn is not None:
                    log_fn(f"Loaded .pth state_dict ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: load processor (AutoProcessor.from_pretrained)...")
                start = time.time()
                hang = _dump_on_hang("AutoProcessor.from_pretrained(config_id)", 30.0)
                processor = AutoProcessor.from_pretrained(config_id, local_files_only=local_files_only)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded processor ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: load config (GroundingDinoConfig.from_pretrained)...")
                start = time.time()
                config = GroundingDinoConfig.from_pretrained(config_id, local_files_only=local_files_only)
                if log_fn is not None:
                    log_fn(f"Loaded config ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: build model (GroundingDinoForObjectDetection(config))...")
                start = time.time()
                gdino = GroundingDinoForObjectDetection(config)
                if log_fn is not None:
                    log_fn(f"Built model ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: apply weights (load_state_dict)...")
                start = time.time()
                missing, unexpected = gdino.load_state_dict(state_dict, strict=False)
                if log_fn is not None:
                    log_fn(f"Applied weights ({time.time() - start:.1f}s)")
                    if missing or unexpected:
                        log_fn(f"WARN: GroundingDINO missing={len(missing)} unexpected={len(unexpected)} keys.")
        except Exception as exc:
            hint = (
                "Cannot load GroundingDINO config/processor locally.\n\n"
                "This app runs in offline mode (no internet), so HuggingFace downloads will not work.\n\n"
                "Fix options:\n"
                "1) Set DINO 'Checkpoint' to a local HF model folder (contains config.json + tokenizer/preprocessor files).\n"
                "2) Or keep a .pth checkpoint but point 'Config ID' to a local HF model folder or cached model id.\n"
                "3) Or pre-download the HuggingFace repo on another machine, then copy it here.\n\n"
                f"ckpt={checkpoint_path}\nconfig={config_id}"
            )
            raise RuntimeError(f"{exc}\n\n{hint}") from exc
        finally:
            tick_stop.set()

        if log_fn is not None:
            log_fn("Step: move model to device (gdino.to)...")
        start = time.time()
        gdino.to(device)
        if log_fn is not None:
            log_fn(f"Moved model to device ({time.time() - start:.1f}s)")
        gdino.eval()
        self._processor = processor
        self._gdino = gdino
        self._gdino_checkpoint = checkpoint_path
        self._gdino_config_id = config_id
        self._device = device
        if log_fn is not None:
            log_fn(f"GroundingDINO ready (ckpt={checkpoint_path}, config={config_id}).")
        return self._processor, self._gdino, device

    def _run_with_roi(self, func_name: str, image_path: str, params: DinoParams, **kwargs) -> dict:
        import cv2

        image = _cv2_imread_any_path(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        roi_x1, roi_y1, roi_x2, roi_y2 = params.roi_box
        roi_x1 = max(0, min(image.shape[1] - 1, int(roi_x1)))
        roi_y1 = max(0, min(image.shape[0] - 1, int(roi_y1)))
        roi_x2 = max(0, min(image.shape[1], int(roi_x2)))
        roi_y2 = max(0, min(image.shape[0], int(roi_y2)))
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            raise ValueError("Invalid ROI box")
        crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
        ext = os.path.splitext(image_path)[1] or ".png"
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        cv2.imwrite(tmp_path, crop)
        try:
            sub_params = DinoParams(
                gdino_checkpoint=params.gdino_checkpoint,
                gdino_config_id=params.gdino_config_id,
                text_queries=params.text_queries,
                box_threshold=params.box_threshold,
                text_threshold=params.text_threshold,
                max_dets=params.max_dets,
                device=params.device,
                output_dir=params.output_dir,
                roi_box=None,
                nms_iou_threshold=params.nms_iou_threshold,
                parent_contain_threshold=params.parent_contain_threshold,
                recursive_min_box_px=params.recursive_min_box_px,
                recursive_max_depth=params.recursive_max_depth,
            )
            func = getattr(self, func_name)
            result = dict(func(tmp_path, sub_params, **kwargs) or {})

            def _shift_boxes(items: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
                shifted: list[dict[str, Any]] = []
                for det in items or []:
                    item = dict(det)
                    box = item.get("box")
                    if isinstance(box, list) and len(box) == 4:
                        item["box"] = [
                            float(box[0]) + roi_x1,
                            float(box[1]) + roi_y1,
                            float(box[2]) + roi_x1,
                            float(box[3]) + roi_y1,
                        ]
                    shifted.append(item)
                return shifted

            detections = _shift_boxes(result.get("detections"))
            display_detections = _shift_boxes(result.get("display_detections"))
            result["image_path"] = str(image_path)
            result["detections"] = detections
            if "display_detections" in result or display_detections:
                result["display_detections"] = display_detections
                result["display_dets"] = len(display_detections)
            result["dets"] = len(detections)
            return result
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def predict(self, image_path: str, params: DinoParams, *, stop_checker=None, log_fn=None) -> dict:
        import cv2
        from PIL import Image

        if params.roi_box is not None:
            return self._run_with_roi("predict", image_path, params, stop_checker=stop_checker, log_fn=log_fn)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        os.makedirs(params.output_dir, exist_ok=True)
        image = _cv2_imread_any_path(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            base_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raster_fallback = _load_tiff_with_rasterio(image_path, log_fn=log_fn) if str(image_path).lower().endswith((".tif", ".tiff")) else None
            if raster_fallback is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            base_rgb, _fallback_valid_mask = raster_fallback
        rgb, valid_mask, strategy_name = _build_valid_mask(image_path, base_rgb, log_fn=log_fn)
        valid_integral = _mask_integral(valid_mask)
        black_integral = _pure_black_integral(rgb)
        height, width = rgb.shape[:2]
        processor, gdino, device = self.ensure_model_loaded(params, log_fn=log_fn)
        if stop_checker is not None and stop_checker():
            raise RuntimeError("Stopped")
        if log_fn is not None:
            log_fn(f"Running DINO detect-only... valid-mask strategy={strategy_name}")
        pil_img = Image.fromarray(rgb)
        try:
            detections = run_text_boxes(
                processor=processor,
                gdino=gdino,
                device=device,
                pil_image=pil_img,
                text_queries=list(params.text_queries),
                box_threshold=float(params.box_threshold),
                text_threshold=float(params.text_threshold),
            )
        except RuntimeError as exc:
            if "selected index k out of range" not in str(exc):
                raise
            padded_rgb = _pad_rgb_for_grounding_dino(rgb)
            if padded_rgb.shape[:2] == rgb.shape[:2]:
                raise
            if log_fn is not None:
                log_fn(
                    "GroundingDINO retry: padding extreme-aspect image "
                    f"from {width}x{height} to {padded_rgb.shape[1]}x{padded_rgb.shape[0]}."
                )
            detections = run_text_boxes(
                processor=processor,
                gdino=gdino,
                device=device,
                pil_image=Image.fromarray(padded_rgb),
                text_queries=list(params.text_queries),
                box_threshold=float(params.box_threshold),
                text_threshold=float(params.text_threshold),
            )
        before_invalid_filter = len(detections)
        detections = [
            det
            for det in detections
            if _box_is_fully_valid(det.box_xyxy, valid_integral, width=width, height=height)
            and _box_black_ratio(det.box_xyxy, black_integral, width=width, height=height) < _DINO_BOX_BLACK_RATIO_REJECT
        ]
        if log_fn is not None and len(detections) < before_invalid_filter:
            log_fn(
                f"DINO invalid-region filter: removed {before_invalid_filter - len(detections)} box(es) "
                f"with invalid overlap or near_black_ratio>={_DINO_BOX_BLACK_RATIO_REJECT:.2f} "
                f"(pixel<={_DINO_BLACK_PIXEL_THRESHOLD})."
            )
        detections = detections[: max(0, int(params.max_dets))] if int(params.max_dets) > 0 else detections
        payload = [
            {
                "label": str(det.label),
                "score": float(det.score),
                "box": [float(det.box_xyxy[0]), float(det.box_xyxy[1]), float(det.box_xyxy[2]), float(det.box_xyxy[3])],
                "model_name": "Dino",
            }
            for det in detections
        ]
        if log_fn is not None:
            log_fn(f"DINO done. dets={len(payload)}")
        return {
            "image_path": str(image_path),
            "output_dir": params.output_dir,
            "dets": len(payload),
            "detections": payload,
        }

    def rank_boxes(self, image_path: str, params: DinoParams, *, stop_checker=None, log_fn=None) -> dict:
        import cv2
        from PIL import Image

        if params.roi_box is not None:
            raise ValueError("rank_boxes does not support ROI mode yet.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_max_dim = _image_max_dim_from_path(image_path, log_fn=log_fn)
        use_tiled_dino = image_max_dim > 512
        if use_tiled_dino:
            if log_fn is not None:
                log_fn(f"Large image detected ({image_max_dim}px). Using recursive/tiled DINO for proposal boxes.")
            detection_result = self.predict_recursive(
                image_path,
                params,
                target_labels=list(params.text_queries),
                max_depth=int(params.recursive_max_depth),
                min_box_px=int(params.recursive_min_box_px),
                stop_checker=stop_checker,
                log_fn=log_fn,
            )
        else:
            detection_result = self.predict(image_path, params, stop_checker=stop_checker, log_fn=log_fn)
        detections = list(detection_result.get("detections") or [])
        if not detections:
            return {
                **detection_result,
                "ranked_detections": [],
                "selected_detections": [],
                "rejected_detections": [],
                "selected_count": 0,
                "rejected_count": 0,
                "top_k": max(1, int(params.top_k)),
                "used_tiled_dino": use_tiled_dino,
            }

        image = _cv2_imread_any_path(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raster_fallback = _load_tiff_with_rasterio(image_path, log_fn=log_fn) if str(image_path).lower().endswith((".tif", ".tiff")) else None
            if raster_fallback is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            image_rgb, _mask = raster_fallback
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        crop_dir = Path(params.output_dir) / str(params.crop_dirname or "dino_crops").strip()
        crop_dir.mkdir(parents=True, exist_ok=True)
        query_labels = normalize_queries(params.text_queries)
        prototype_dir = str(params.prototype_dir or "").strip()

        crop_paths: list[str] = []
        crop_images: list[Any] = []
        cropped_detections: list[dict[str, Any]] = []
        stem = Path(image_path).stem
        for index, det in enumerate(detections, start=1):
            crop_box = _sanitize_crop_box(det.get("box"), width=width, height=height)
            if crop_box is None:
                continue
            x1, y1, x2, y2 = crop_box
            crop_rgb = image_rgb[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                continue
            crop_path = crop_dir / f"{stem}_crop_{index:03d}.png"
            cv2.imwrite(str(crop_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            crop_images.append(Image.fromarray(crop_rgb))
            crop_paths.append(str(crop_path))
            payload = dict(det)
            payload["crop_box"] = [int(x1), int(y1), int(x2), int(y2)]
            cropped_detections.append(payload)

        if not cropped_detections:
            return {
                **detection_result,
                "ranked_detections": [],
                "selected_detections": [],
                "rejected_detections": [],
                "selected_count": 0,
                "rejected_count": 0,
                "top_k": max(1, int(params.top_k)),
                "used_tiled_dino": use_tiled_dino,
            }

        if stop_checker is not None and stop_checker():
            raise RuntimeError("Stopped")

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        if prototype_dir:
            prototype_checkpoint = str(params.dinov2_checkpoint or "").strip() or default_dinov2_embedding_checkpoint()
            background_labels = normalize_queries(params.prototype_background_labels)
            include_labels = normalize_queries(list(query_labels) + list(background_labels))
            prototype_outputs = self._prototype_runner.classify_crops(
                checkpoint_path=prototype_checkpoint,
                prototype_dir=prototype_dir,
                images=crop_images,
                include_labels=include_labels,
                device_preference=params.device,
                batch_size=params.prototype_batch_size,
                top_k=params.prototype_top_k_labels,
                log_fn=log_fn,
            )
            for det, crop_path, prototype_output in zip(cropped_detections, crop_paths, prototype_outputs):
                item = dict(det)
                item["proposal_label"] = str(det.get("label") or "")
                item["proposal_score"] = float(det.get("score") or 0.0)
                item["crop_path"] = str(crop_path)
                item["prototype_label"] = str(prototype_output.get("label") or "")
                item["prototype_similarity"] = float(prototype_output.get("similarity") or 0.0)
                item["prototype_support_count"] = int(prototype_output.get("support_count") or 0)
                item["prototype_top_predictions"] = list(prototype_output.get("top_predictions") or [])
                decision = _resolve_prototype_decision(
                    proposal_label=item["proposal_label"],
                    predictions=item["prototype_top_predictions"],
                    min_similarity=float(params.prototype_min_similarity),
                    background_labels=background_labels,
                    strict=bool(params.prototype_strict),
                )
                item["prototype_decision"] = str(decision.get("reason") or "")
                item["matched_prototype_label"] = decision.get("matched_prototype_label")
                item["matched_prototype_similarity"] = float(decision.get("matched_similarity") or 0.0)
                if decision.get("final_label"):
                    item["label"] = str(decision["final_label"])
                matched_similarity = max(0.0, float(decision.get("matched_similarity") or 0.0))
                if item["prototype_decision"] == "prototype_mapped" and matched_similarity > 0.0:
                    item["refined_score"] = float(item["proposal_score"]) * matched_similarity
                else:
                    item["refined_score"] = float(item["proposal_score"])
                if bool(decision.get("accepted")):
                    accepted.append(item)
                else:
                    item["rejection_reason"] = item["prototype_decision"]
                    rejected.append(item)
            relabel_mode = "dinov2_prototypes"
            relabel_model = prototype_checkpoint
        else:
            classifier_checkpoint = str(params.dinov2_checkpoint or "").strip() or default_dinov2_checkpoint()
            classifier_rules = _load_classifier_rules(params.classifier_map_path)
            if not classifier_rules:
                classifier_rules = _default_classifier_rules_for_queries(params.text_queries, classifier_checkpoint)
            classifier_outputs = self._dinov2_runner.classify_crops(
                checkpoint_path=classifier_checkpoint,
                images=crop_images,
                device_preference=params.device,
                batch_size=params.classifier_batch_size,
                top_k=params.classifier_top_k_labels,
                log_fn=log_fn,
            )
            for det, crop_path, classifier_output in zip(cropped_detections, crop_paths, classifier_outputs):
                item = dict(det)
                item["proposal_label"] = str(det.get("label") or "")
                item["proposal_score"] = float(det.get("score") or 0.0)
                item["crop_path"] = str(crop_path)
                item["classifier_label"] = str(classifier_output.get("label") or "")
                item["classifier_confidence"] = float(classifier_output.get("confidence") or 0.0)
                item["classifier_top_predictions"] = list(classifier_output.get("top_predictions") or [])
                decision = _resolve_classifier_decision(
                    proposal_label=item["proposal_label"],
                    predictions=item["classifier_top_predictions"],
                    query_labels=query_labels,
                    rules=classifier_rules,
                    min_confidence=float(params.classifier_min_confidence),
                    strict=bool(params.classifier_strict),
                )
                item["classifier_decision"] = str(decision.get("reason") or "")
                item["matched_classifier_label"] = decision.get("matched_classifier_label")
                item["matched_classifier_confidence"] = float(decision.get("matched_confidence") or 0.0)
                if decision.get("final_label"):
                    item["label"] = str(decision["final_label"])
                matched_confidence = float(decision.get("matched_confidence") or 0.0)
                if matched_confidence > 0.0 and item["classifier_decision"] in {"classifier_rule_mapped", "classifier_query_mapped"}:
                    item["refined_score"] = float(item["proposal_score"]) * matched_confidence
                else:
                    item["refined_score"] = float(item["proposal_score"])
                if bool(decision.get("accepted")):
                    accepted.append(item)
                else:
                    item["rejection_reason"] = item["classifier_decision"]
                    rejected.append(item)
            relabel_mode = "dinov2_classifier"
            relabel_model = classifier_checkpoint

        ranked, selected = _rank_dino_payload(
            accepted,
            [str(item.get("crop_path") or "") for item in accepted],
            top_k=params.top_k,
        )

        if bool(params.save_overlay):
            overlay = image_rgb.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            mm_per_px = float(getattr(params, "mm_per_px", 0.0) or 0.0)

            def _format_area(x1: int, y1: int, x2: int, y2: int) -> str:
                if mm_per_px <= 0.0:
                    return ""
                w = max(0, int(x2) - int(x1))
                h = max(0, int(y2) - int(y1))
                area_px2 = float(w * h)
                area_mm2 = area_px2 * (mm_per_px * mm_per_px)
                area_cm2 = area_mm2 / 100.0
                if area_cm2 >= 1.0:
                    return f" A={area_cm2:.2f}cm2"
                return f" A={area_mm2:.0f}mm2"

            def _draw(items: list[dict[str, Any]], *, color: tuple[int, int, int]) -> None:
                for entry in items:
                    box = entry.get("crop_box") or entry.get("box")
                    sanitized = _sanitize_crop_box(box, width=width, height=height)
                    if sanitized is None:
                        continue
                    x1, y1, x2, y2 = sanitized
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                    label = str(entry.get("label") or entry.get("proposal_label") or "")
                    score = float(entry.get("refined_score") or entry.get("score") or 0.0)
                    sim = entry.get("matched_prototype_similarity")
                    sim_text = f" sim={float(sim):.2f}" if sim is not None else ""
                    area_text = _format_area(x1, y1, x2, y2)
                    text = f"{label} {score:.2f}{sim_text}{area_text}".strip()
                    if text:
                        cv2.putText(
                            overlay,
                            text,
                            (x1, max(14, y1 - 6)),
                            font,
                            0.55,
                            (0, 0, 0),
                            4,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            overlay,
                            text,
                            (x1, max(14, y1 - 6)),
                            font,
                            0.55,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

            _draw(ranked, color=(40, 220, 60))
            if bool(params.overlay_include_rejected):
                _draw(rejected, color=(30, 80, 240))

            overlay_path = Path(params.output_dir) / str(params.overlay_filename or "overlay_filtered.png")
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if log_fn is not None:
                log_fn(f"Saved overlay: {overlay_path}")

        if log_fn is not None:
            log_fn(
                f"DINOv2 refine done. mode={relabel_mode} proposals={len(cropped_detections)} kept={len(accepted)} "
                f"rejected={len(rejected)} selected={len(selected)}"
            )
        return {
            **detection_result,
            "classifier_model": relabel_model,
            "relabel_model": relabel_model,
            "relabel_mode": relabel_mode,
            "prototype_dir": prototype_dir or None,
            "overlay_path": str((Path(params.output_dir) / str(params.overlay_filename or "overlay_filtered.png")).resolve())
            if bool(params.save_overlay)
            else None,
            "ranked_detections": ranked,
            "selected_detections": selected,
            "rejected_detections": rejected,
            "selected_count": len(selected),
            "rejected_count": len(rejected),
            "top_k": max(1, int(params.top_k)),
            "used_tiled_dino": use_tiled_dino,
        }

    def predict_recursive(
        self,
        image_path: str,
        params: DinoParams,
        *,
        target_labels: Sequence[str] = ("crack",),
        max_depth: int = 3,
        min_box_px: int = 48,
        nonblack_thresh: int = 10,
        stop_checker=None,
        log_fn=None,
    ) -> dict:
        import cv2
        import numpy as np
        from PIL import Image

        if params.roi_box is not None:
            return self._run_with_roi(
                "predict_recursive",
                image_path,
                params,
                target_labels=target_labels,
                max_depth=max_depth,
                min_box_px=min_box_px,
                nonblack_thresh=nonblack_thresh,
                stop_checker=stop_checker,
                log_fn=log_fn,
            )
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        os.makedirs(params.output_dir, exist_ok=True)
        image = _cv2_imread_any_path(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            base_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raster_fallback = _load_tiff_with_rasterio(image_path, log_fn=log_fn) if str(image_path).lower().endswith((".tif", ".tiff")) else None
            if raster_fallback is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            base_rgb, _fallback_valid_mask = raster_fallback

        original_height, original_width = base_rgb.shape[:2]
        base_rgb, valid_mask, strategy_name = _build_valid_mask(image_path, base_rgb, log_fn=log_fn)
        original_roi = _mask_bbox(valid_mask)
        if log_fn is not None:
            log_fn(f"Tiled DINO valid-mask strategy: {strategy_name}")

        if original_roi is None or int(np.count_nonzero(valid_mask)) == 0:
            if log_fn is not None:
                log_fn("WARN: No valid pixels found in image. Nothing to scan.")
            return {
                "image_path": str(image_path),
                "output_dir": params.output_dir,
                "display_detections": [],
                "display_dets": 0,
                "dets": 0,
                "detections": [],
            }

        oriented = _compute_oriented_roi(base_rgb, valid_mask, log_fn=log_fn)
        work_rgb = oriented["rgb"]
        work_valid_mask = oriented["valid_mask"]
        work_roi = oriented["roi_box"]
        rotation_angle = float(oriented.get("rotation_angle") or 0.0)
        inverse_matrix = oriented.get("inverse_matrix")
        rotated = bool(oriented.get("rotated"))

        if log_fn is not None:
            orig_x1, orig_y1, orig_x2, orig_y2 = original_roi
            if work_roi is None:
                log_fn(
                    f"Valid ROI: original=({orig_x1},{orig_y1})-({orig_x2},{orig_y2}) "
                    f"size={orig_x2-orig_x1}x{orig_y2-orig_y1}. rotation={rotation_angle:.2f}deg"
                )
            else:
                roi_x1, roi_y1, roi_x2, roi_y2 = work_roi
                log_fn(
                    f"Valid ROI: original=({orig_x1},{orig_y1})-({orig_x2},{orig_y2}) "
                    f"rotated=({roi_x1},{roi_y1})-({roi_x2},{roi_y2}) "
                    f"size={roi_x2-roi_x1}x{roi_y2-roi_y1}. rotation={rotation_angle:.2f}deg"
                )

        if work_roi is None:
            if log_fn is not None:
                log_fn("WARN: Valid ROI is empty after preprocessing. Nothing to scan.")
            return {
                "image_path": str(image_path),
                "output_dir": params.output_dir,
                "display_detections": [],
                "display_dets": 0,
                "dets": 0,
                "detections": [],
            }

        effective_min_box_px = int(min_box_px if min_box_px is not None else params.recursive_min_box_px)
        processor, gdino, device = self.ensure_model_loaded(params, log_fn=log_fn)
        original_valid_integral = _mask_integral(valid_mask)
        original_black_integral = _pure_black_integral(base_rgb)
        tile_passes = [
            ("small", _RECURSIVE_TILE_SIZE, _RECURSIVE_TILE_OVERLAP, _RECURSIVE_MIN_VALID_COVERAGE),
            ("medium", _RECURSIVE_MEDIUM_TILE_SIZE, _RECURSIVE_MEDIUM_TILE_OVERLAP, _RECURSIVE_MEDIUM_MIN_VALID_COVERAGE),
            ("large", _RECURSIVE_LARGE_TILE_SIZE, _RECURSIVE_LARGE_TILE_OVERLAP, _RECURSIVE_LARGE_MIN_VALID_COVERAGE),
        ]
        allowed_scales = {str(s).strip().lower() for s in (params.recursive_tile_scales or []) if str(s).strip()}
        if allowed_scales:
            tile_passes = [entry for entry in tile_passes if entry[0] in allowed_scales]

        detections: list[tuple[Any, str, float]] = []
        total_tile_dets = 0
        total_tiles_seen = 0
        total_tiles_kept = 0
        total_tiles_skipped = 0
        total_refined_tiles = 0
        for pass_name, tile_size, tile_overlap, pass_min_valid_coverage in tile_passes:
            tiles, total_tiles, skipped_tiles, refined_tiles = _generate_valid_tiles(
                work_valid_mask,
                work_roi,
                tile_size=tile_size,
                overlap=tile_overlap,
                min_valid_coverage=pass_min_valid_coverage,
                allow_refine=False,
            )
            kept_tiles = len(tiles)
            total_tiles_seen += total_tiles
            total_tiles_kept += kept_tiles
            total_tiles_skipped += skipped_tiles
            total_refined_tiles += refined_tiles
            if log_fn is not None:
                log_fn(
                    f"Tiled DINO {pass_name}: total={total_tiles} kept={kept_tiles} refined={refined_tiles} skipped={skipped_tiles} "
                    f"tile={tile_size}px overlap={tile_overlap}px min_valid_coverage={pass_min_valid_coverage:.2f}"
                )
            if kept_tiles == 0:
                continue

            for tile_index, (tile_x1, tile_y1, tile_x2, tile_y2, coverage, tile_kind) in enumerate(tiles, start=1):
                if stop_checker is not None and stop_checker():
                    raise RuntimeError("Stopped")
                patch_rgb = work_rgb[tile_y1:tile_y2, tile_x1:tile_x2]
                if patch_rgb.size == 0:
                    continue
                tile_dets = run_text_boxes(
                    processor=processor,
                    gdino=gdino,
                    device=device,
                    pil_image=Image.fromarray(patch_rgb),
                    text_queries=list(params.text_queries),
                    box_threshold=float(params.box_threshold),
                    text_threshold=float(params.text_threshold),
                )
                total_tile_dets += len(tile_dets)
                if log_fn is not None:
                    log_fn(
                        f"[{pass_name} {tile_index}/{kept_tiles}] {tile_kind} crop ({tile_x1},{tile_y1})-({tile_x2},{tile_y2}) "
                        f"coverage={coverage:.3f} -> {len(tile_dets)} dets"
                    )
                for det in tile_dets:
                    box = det.box_xyxy.astype(np.float32).copy()
                    box[0] += float(tile_x1)
                    box[1] += float(tile_y1)
                    box[2] += float(tile_x1)
                    box[3] += float(tile_y1)
                    mapped_box = _map_box_from_rotated_to_original(
                        box,
                        inverse_matrix,
                        original_width=original_width,
                        original_height=original_height,
                    )
                    width_px = float(mapped_box[2] - mapped_box[0])
                    height_px = float(mapped_box[3] - mapped_box[1])
                    if width_px < float(effective_min_box_px) or height_px < float(effective_min_box_px):
                        continue
                    box_valid_coverage = _box_valid_coverage(
                        mapped_box,
                        original_valid_integral,
                        width=original_width,
                        height=original_height,
                    )
                    if box_valid_coverage < 1.0:
                        continue
                    if (
                        _box_black_ratio(
                            mapped_box,
                            original_black_integral,
                            width=original_width,
                            height=original_height,
                        )
                        >= _DINO_BOX_BLACK_RATIO_REJECT
                    ):
                        continue
                    detections.append((mapped_box, str(det.label), float(det.score)))

        if total_tiles_kept == 0:
            if log_fn is not None:
                log_fn("No valid small/large tiles passed coverage threshold. Skipping DINO.")
            return {
                "image_path": str(image_path),
                "output_dir": params.output_dir,
                "display_detections": [],
                "display_dets": 0,
                "dets": 0,
                "detections": [],
            }

        if stop_checker is not None and stop_checker():
            raise RuntimeError("Stopped")
        if log_fn is not None:
            log_fn(
                f"Tiled DINO raw detections: total={total_tile_dets} "
                f"tiles_total={total_tiles_seen} tiles_kept={total_tiles_kept} refined={total_refined_tiles} skipped={total_tiles_skipped} "
                f"kept_after_size={len(detections)} rotated={str(rotated).lower()}"
            )
        targets = normalize_queries(target_labels) if target_labels else []
        kept = []
        for box, label, score in detections:
            if targets and not label_matches(label, targets):
                continue
            kept.append((box, label, score))
        display_payload = [
            {
                "label": str(label),
                "score": float(score),
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "model_name": "DinoRecursive",
            }
            for box, label, score in kept
        ]
        before_cf = len(kept)
        kept = _filter_parent_boxes(kept, contain_thresh=float(params.parent_contain_threshold))
        if log_fn is not None and len(kept) < before_cf:
            log_fn(f"Containment filter: removed {before_cf - len(kept)} parent box(es), {len(kept)} boxes remain.")
        kept = _nms_boxes(kept, iou_threshold=float(params.nms_iou_threshold))
        if int(params.max_dets) > 0:
            kept = kept[: int(params.max_dets)]
        payload = [
            {
                "label": str(label),
                "score": float(score),
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "model_name": "DinoRecursive",
            }
            for box, label, score in kept
        ]
        if log_fn is not None:
            log_fn(f"Recursive DINO done. display_dets={len(display_payload)} dets={len(payload)}")
        return {
            "image_path": str(image_path),
            "output_dir": params.output_dir,
            "display_detections": display_payload,
            "display_dets": len(display_payload),
            "dets": len(payload),
            "detections": payload,
        }
