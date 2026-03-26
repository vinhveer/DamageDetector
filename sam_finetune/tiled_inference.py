from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


TileMaskScoreFn = Callable[[np.ndarray], tuple[np.ndarray, float]]


def resolve_tile_size(tile_size: int | None, fallback: int = 512) -> int:
    size = int(tile_size) if tile_size is not None else int(fallback)
    if size <= 0:
        size = int(fallback)
    return max(32, size)


def resolve_tile_overlap(tile_size: int, tile_overlap: int | None, fallback: int | None = None) -> int:
    if fallback is None:
        fallback = tile_size // 2
    overlap = int(tile_overlap) if tile_overlap is not None else int(fallback)
    if overlap < 0:
        overlap = int(fallback)
    if overlap >= tile_size:
        overlap = tile_size - 1
    return max(0, overlap)


def _tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    tail = length - tile_size
    if positions[-1] != tail:
        positions.append(tail)
    return positions


def make_center_weight(tile_size: int, edge_floor: float = 0.05) -> np.ndarray:
    tile_size = resolve_tile_size(tile_size)
    win = np.hanning(tile_size).astype(np.float32)
    if tile_size == 1:
        win = np.ones((1,), dtype=np.float32)
    window = np.outer(win, win).astype(np.float32)
    if float(window.max()) <= 0.0:
        window = np.ones((tile_size, tile_size), dtype=np.float32)
    else:
        window /= float(window.max())
    window = np.maximum(window, float(edge_floor))
    return window.astype(np.float32)


def predictor_tile_mask_score(predictor, tile_hwc: np.ndarray) -> tuple[np.ndarray, float]:
    tile = np.asarray(tile_hwc)
    if tile.dtype != np.uint8:
        if tile.max() <= 1.0:
            tile = np.clip(tile * 255.0, 0, 255).astype(np.uint8)
        else:
            tile = np.clip(tile, 0, 255).astype(np.uint8)
    h_img, w_img = tile.shape[:2]
    predictor.set_image(tile)
    full_box = np.array([[0.0, 0.0, float(w_img - 1), float(h_img - 1)]], dtype=np.float32)
    masks, scores, _ = predictor.predict(box=full_box, multimask_output=True)
    if masks is None or len(masks) == 0:
        return np.zeros((h_img, w_img), dtype=np.float32), 0.0
    idx = int(np.argmax(scores))
    mask = masks[idx].astype(np.float32)
    score = float(scores[idx]) if scores is not None else 1.0
    return mask, score


def tiled_score_map(
    image_hwc: np.ndarray,
    tile_size: int,
    tile_overlap: int,
    predict_tile_mask_score: TileMaskScoreFn,
) -> np.ndarray:
    image = np.asarray(image_hwc)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {tuple(image.shape)}")

    tile_size = resolve_tile_size(tile_size)
    tile_overlap = resolve_tile_overlap(tile_size, tile_overlap)
    stride = max(1, tile_size - tile_overlap)
    h_img, w_img = image.shape[:2]
    value_sum = np.zeros((h_img, w_img), dtype=np.float32)
    weight_sum = np.zeros((h_img, w_img), dtype=np.float32)
    center_weight = make_center_weight(tile_size)

    y_positions = _tile_positions(h_img, tile_size, stride)
    x_positions = _tile_positions(w_img, tile_size, stride)

    for y in y_positions:
        for x in x_positions:
            tile = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
            y2 = min(h_img, y + tile_size)
            x2 = min(w_img, x + tile_size)
            valid_h = y2 - y
            valid_w = x2 - x
            tile[:valid_h, :valid_w] = image[y:y2, x:x2]
            mask_tile, tile_score = predict_tile_mask_score(tile)
            mask_tile = np.asarray(mask_tile, dtype=np.float32)
            if mask_tile.ndim == 3:
                mask_tile = np.squeeze(mask_tile, axis=0)
            if mask_tile.shape != (tile_size, tile_size):
                raise ValueError(
                    f"Tile predictor must return shape {(tile_size, tile_size)}, got {tuple(mask_tile.shape)}"
                )
            score_weight = max(float(tile_score), 1e-6)
            local_weight = center_weight[:valid_h, :valid_w] * score_weight
            local_value = mask_tile[:valid_h, :valid_w] * local_weight
            value_sum[y:y2, x:x2] += local_value
            weight_sum[y:y2, x:x2] += local_weight

    weight_sum = np.clip(weight_sum, 1e-6, None)
    return value_sum / weight_sum


def binary_mask_from_score_map(score_map: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(score_map, dtype=np.float32) >= float(threshold)).astype(np.uint8)


def metric_per_case(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float]:
    pred_mask = np.asarray(pred) > 0
    gt_mask = np.asarray(gt) > 0

    pred_area = int(pred_mask.sum())
    gt_area = int(gt_mask.sum())

    if pred_area > 0 and gt_area > 0:
        inter = int(np.logical_and(pred_mask, gt_mask).sum())
        union = int(np.logical_or(pred_mask, gt_mask).sum())
        dice = (2.0 * inter) / float(pred_area + gt_area)
        iou = inter / float(union) if union > 0 else 0.0
        precision = inter / float(pred_area)
        recall = inter / float(gt_area)
        return float(precision), float(recall), float(dice), float(iou)
    if pred_area > 0 and gt_area == 0:
        return 0.0, 0.0, 0.0, 0.0
    if pred_area == 0 and gt_area == 0:
        return 1.0, 1.0, 1.0, 1.0
    return 0.0, 0.0, 0.0, 0.0


def threshold_sweep(prob_map: np.ndarray, gt_mask: np.ndarray, thresholds: list[float]) -> dict[float, tuple[float, float, float, float]]:
    results: dict[float, tuple[float, float, float, float]] = {}
    for thr in thresholds:
        thr_value = float(thr)
        pred = binary_mask_from_score_map(prob_map, thr_value)
        results[thr_value] = metric_per_case(pred, gt_mask)
    return results


def best_threshold_result(
    sweep: dict[float, tuple[float, float, float, float]]
) -> tuple[float, tuple[float, float, float, float]]:
    if not sweep:
        raise ValueError("threshold sweep is empty")
    best_thr = max(sweep.keys(), key=lambda thr: (sweep[thr][3], sweep[thr][2], -abs(thr - 0.5)))
    return float(best_thr), sweep[best_thr]


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return binary
    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()
    while True:
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = eroded
        if int(cv2.countNonZero(work)) == 0:
            break
    return (skeleton > 0).astype(np.uint8)


def continuity_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_mask = (np.asarray(pred) > 0).astype(np.uint8)
    gt_mask = (np.asarray(gt) > 0).astype(np.uint8)

    pred_skel = skeletonize_mask(pred_mask)
    gt_skel = skeletonize_mask(gt_mask)
    skel_precision, skel_recall, skel_dice, _skel_iou = metric_per_case(pred_skel, gt_skel)

    pred_components = int(cv2.connectedComponents(pred_mask, connectivity=8)[0] - 1) if int(pred_mask.sum()) > 0 else 0
    gt_components = int(cv2.connectedComponents(gt_mask, connectivity=8)[0] - 1) if int(gt_mask.sum()) > 0 else 0
    if gt_components == 0 and pred_components == 0:
        fragmentation = 1.0
    elif gt_components == 0:
        fragmentation = 10.0
    else:
        fragmentation = min(float(pred_components) / float(max(gt_components, 1)), 10.0)

    return {
        "skeleton_dice": float(skel_dice),
        "centerline_precision": float(skel_precision),
        "centerline_recall": float(skel_recall),
        "component_fragmentation": float(fragmentation),
    }
