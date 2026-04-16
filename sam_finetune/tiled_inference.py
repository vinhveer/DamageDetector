from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


TileMaskScoreFn = Callable[[np.ndarray], tuple[np.ndarray, float]]
TileProbFn = Callable[[np.ndarray], np.ndarray]


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
    return make_center_weight_2d(tile_size, tile_size, edge_floor=edge_floor)


def make_center_weight_2d(height: int, width: int, edge_floor: float = 0.05) -> np.ndarray:
    height = max(1, int(height))
    width = max(1, int(width))
    win_h = np.hanning(height).astype(np.float32) if height > 1 else np.ones((1,), dtype=np.float32)
    win_w = np.hanning(width).astype(np.float32) if width > 1 else np.ones((1,), dtype=np.float32)
    window = np.outer(win_h, win_w).astype(np.float32)
    if float(window.max()) <= 0.0:
        window = np.ones((height, width), dtype=np.float32)
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


def _select_binary_logits(outputs) -> "torch.Tensor":
    masks = outputs["masks"]
    if masks.shape[1] == 1:
        return masks[:, :1]
    scores = outputs.get("iou_predictions")
    if scores is None:
        return masks[:, :1]
    from torch_runtime import torch

    best_idx = torch.argmax(scores, dim=1)
    return masks[torch.arange(masks.shape[0], device=masks.device), best_idx].unsqueeze(1)


def model_tile_prob_map(
    model,
    tile_hwc: np.ndarray,
    *,
    image_size: int,
    multimask_output: bool,
    use_amp: bool = False,
) -> np.ndarray:
    from torch_runtime import torch

    tile = np.asarray(tile_hwc)
    if tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError(f"Expected tile shape (H, W, 3), got {tuple(tile.shape)}")

    if tile.dtype != np.float32:
        if tile.max() <= 1.0:
            tile = tile.astype(np.float32)
        else:
            tile = tile.astype(np.float32) / 255.0
    tile = np.clip(tile, 0.0, 1.0)

    h_img, w_img = tile.shape[:2]
    inputs = torch.from_numpy(np.transpose(tile, (2, 0, 1))).unsqueeze(0).float()
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Model has no parameters to infer device from.") from exc
    inputs = inputs.to(device=device)
    boxes = torch.tensor([[0.0, 0.0, float(w_img), float(h_img)]], dtype=torch.float32, device=device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(use_amp) and device.type == "cuda"):
            outputs = model(inputs, multimask_output, image_size, boxes=boxes, points=None)
            logits = _select_binary_logits(outputs)
            prob_map = torch.sigmoid(logits)[0, 0, :h_img, :w_img]
    return prob_map.detach().float().cpu().numpy()


def tiled_model_score_map(
    image_hwc: np.ndarray,
    tile_size: int,
    tile_overlap: int,
    *,
    model,
    image_size: int,
    multimask_output: bool,
    use_amp: bool = False,
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

    y_positions = _tile_positions(h_img, tile_size, stride)
    x_positions = _tile_positions(w_img, tile_size, stride)

    for y in y_positions:
        for x in x_positions:
            y2 = min(h_img, y + tile_size)
            x2 = min(w_img, x + tile_size)
            tile = image[y:y2, x:x2]
            prob_tile = np.asarray(
                model_tile_prob_map(
                    model,
                    tile,
                    image_size=image_size,
                    multimask_output=multimask_output,
                    use_amp=use_amp,
                ),
                dtype=np.float32,
            )
            expected_shape = (y2 - y, x2 - x)
            if prob_tile.shape != expected_shape:
                raise ValueError(
                    f"Model tile predictor must return shape {expected_shape}, got {tuple(prob_tile.shape)}"
                )
            local_weight = make_center_weight_2d(expected_shape[0], expected_shape[1])
            value_sum[y:y2, x:x2] += prob_tile * local_weight
            weight_sum[y:y2, x:x2] += local_weight

    weight_sum = np.clip(weight_sum, 1e-6, None)
    return value_sum / weight_sum


def _clip_box_to_image(box: tuple[int, int, int, int], image_h: int, image_w: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, image_w))
    y1 = max(0, min(y1, image_h))
    x2 = max(0, min(x2, image_w))
    y2 = max(0, min(y2, image_h))
    if x2 <= x1:
        x2 = min(image_w, x1 + 1)
    if y2 <= y1:
        y2 = min(image_h, y1 + 1)
    return x1, y1, x2, y2


def _fixed_size_box(
    center_x: int,
    center_y: int,
    *,
    roi_size: int,
    image_h: int,
    image_w: int,
) -> tuple[int, int, int, int]:
    half = int(max(1, roi_size)) // 2
    x1 = center_x - half
    y1 = center_y - half
    x2 = x1 + int(roi_size)
    y2 = y1 + int(roi_size)
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image_w:
        shift = x2 - image_w
        x1 -= shift
        x2 = image_w
    if y2 > image_h:
        shift = y2 - image_h
        y1 -= shift
        y2 = image_h
    return _clip_box_to_image((x1, y1, x2, y2), image_h, image_w)


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = float(inter_w * inter_h)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = max(1e-6, area_a + area_b - inter)
    return inter / union


def nms_boxes(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    *,
    iou_threshold: float = 0.3,
    max_keep: int | None = None,
) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda idx: float(scores[idx]), reverse=True)
    keep: list[int] = []
    for idx in order:
        if any(_box_iou(boxes[idx], boxes[kept]) > float(iou_threshold) for kept in keep):
            continue
        keep.append(idx)
        if max_keep is not None and len(keep) >= int(max_keep):
            break
    return keep


def score_uncertainty_map(score_map: np.ndarray, threshold: float) -> np.ndarray:
    prob = np.asarray(score_map, dtype=np.float32)
    thr = float(threshold)
    uncertainty = 1.0 - (np.abs(prob - thr) / 0.5)
    return np.clip(uncertainty, 0.0, 1.0).astype(np.float32)


def mine_refine_rois(
    score_map: np.ndarray,
    *,
    threshold: float,
    roi_size: int = 768,
    max_rois: int = 16,
    roi_padding: int = 64,
    positive_band_low: float = 0.20,
    positive_band_high: float = 0.90,
    score_threshold: float = 0.15,
    nms_iou_threshold: float = 0.3,
) -> list[dict[str, float | tuple[int, int, int, int]]]:
    prob = np.asarray(score_map, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"Expected score map shape (H, W), got {tuple(prob.shape)}")

    image_h, image_w = prob.shape
    uncertainty = score_uncertainty_map(prob, threshold)
    candidate_mask = np.logical_and(prob >= float(positive_band_low), prob <= float(positive_band_high))
    candidate_score = ((0.6 * uncertainty) + (0.4 * prob)) * candidate_mask.astype(np.float32)
    heat = np.clip(candidate_score, 0.0, 1.0)

    binary = (heat >= float(score_threshold)).astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates: list[dict[str, float | tuple[int, int, int, int]]] = []
    for label_idx in range(1, int(num_labels)):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        x1 = max(0, x - int(roi_padding))
        y1 = max(0, y - int(roi_padding))
        x2 = min(image_w, x + w + int(roi_padding))
        y2 = min(image_h, y + h + int(roi_padding))
        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        box = _fixed_size_box(center_x, center_y, roi_size=int(roi_size), image_h=image_h, image_w=image_w)
        bx1, by1, bx2, by2 = box
        local = heat[by1:by2, bx1:bx2]
        score = float(local.mean()) if local.size > 0 else 0.0
        candidates.append({"box": box, "score": score})

    if len(candidates) < int(max_rois):
        flat = heat.reshape(-1)
        if flat.size > 0:
            top_k = min(int(max_rois) * 8, flat.size)
            top_indices = np.argpartition(flat, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
            for flat_idx in top_indices:
                score = float(flat[flat_idx])
                if score < float(score_threshold):
                    break
                cy, cx = np.unravel_index(int(flat_idx), heat.shape)
                box = _fixed_size_box(int(cx), int(cy), roi_size=int(roi_size), image_h=image_h, image_w=image_w)
                candidates.append({"box": box, "score": score})
                if len(candidates) >= int(max_rois) * 4:
                    break

    if not candidates:
        return []

    boxes = [item["box"] for item in candidates]
    scores = [float(item["score"]) for item in candidates]
    keep = nms_boxes(boxes, scores, iou_threshold=float(nms_iou_threshold), max_keep=int(max_rois))
    return [candidates[idx] for idx in keep]


def crop_box_from_image(image_hwc: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    return np.asarray(image_hwc)[y1:y2, x1:x2]


def merge_refined_score_map(
    coarse_score_map: np.ndarray,
    refine_outputs: list[dict[str, object]],
    *,
    merge_mode: str = "weighted_replace",
) -> np.ndarray:
    merged = np.asarray(coarse_score_map, dtype=np.float32).copy()
    if not refine_outputs:
        return merged
    mode = str(merge_mode or "weighted_replace").strip().lower()
    if mode != "weighted_replace":
        raise ValueError(f"Unsupported refine merge mode: {merge_mode!r}")

    for item in refine_outputs:
        box = item["box"]
        prob_map = np.asarray(item["prob_map"], dtype=np.float32)
        x1, y1, x2, y2 = [int(v) for v in box]
        expected_shape = (max(0, y2 - y1), max(0, x2 - x1))
        if prob_map.shape != expected_shape:
            raise ValueError(f"Refine prob map shape {tuple(prob_map.shape)} does not match ROI {expected_shape}")
        local_weight = make_center_weight_2d(expected_shape[0], expected_shape[1], edge_floor=0.15)
        coarse_local = merged[y1:y2, x1:x2]
        merged[y1:y2, x1:x2] = (coarse_local * (1.0 - local_weight)) + (prob_map * local_weight)
    return merged.astype(np.float32)


def coarse_refine_model_score_map(
    image_hwc: np.ndarray,
    *,
    coarse_model,
    coarse_image_size: int,
    coarse_tile_size: int,
    coarse_tile_overlap: int,
    refine_model,
    refine_image_size: int,
    refine_tile_size: int = 768,
    refine_tile_sizes: list[int] | tuple[int, ...] | None = None,
    refine_max_rois: int = 16,
    refine_roi_padding: int = 64,
    refine_merge_mode: str = "weighted_replace",
    refine_score_threshold: float = 0.15,
    positive_band_low: float = 0.20,
    positive_band_high: float = 0.90,
    threshold: float = 0.5,
    multimask_output: bool = False,
    use_amp: bool = False,
    coarse_score_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    if coarse_score_map is None:
        coarse_score_map = tiled_model_score_map(
            image_hwc,
            coarse_tile_size,
            coarse_tile_overlap,
            model=coarse_model,
            image_size=coarse_image_size,
            multimask_output=multimask_output,
            use_amp=use_amp,
        )
    else:
        coarse_score_map = np.asarray(coarse_score_map, dtype=np.float32)

    scale_sequence = [int(refine_tile_size)]
    if refine_tile_sizes:
        scale_sequence = [int(size) for size in refine_tile_sizes if int(size) > 0]
        if not scale_sequence:
            scale_sequence = [int(refine_tile_size)]

    merged_score_map = coarse_score_map.copy()
    all_refine_outputs: list[dict[str, object]] = []
    for scale in scale_sequence:
        roi_candidates = mine_refine_rois(
            merged_score_map,
            threshold=float(threshold),
            roi_size=int(scale),
            max_rois=int(refine_max_rois),
            roi_padding=int(refine_roi_padding),
            positive_band_low=float(positive_band_low),
            positive_band_high=float(positive_band_high),
            score_threshold=float(refine_score_threshold),
        )
        if not roi_candidates:
            continue

        refine_outputs: list[dict[str, object]] = []
        for item in roi_candidates:
            box = item["box"]
            crop = crop_box_from_image(image_hwc, box)
            prob_map = model_tile_prob_map(
                refine_model,
                crop,
                image_size=int(refine_image_size),
                multimask_output=multimask_output,
                use_amp=use_amp,
            )
            refine_outputs.append({"box": box, "score": float(item["score"]), "prob_map": prob_map, "scale": int(scale)})

        merged_score_map = merge_refined_score_map(
            merged_score_map,
            refine_outputs,
            merge_mode=refine_merge_mode,
        )
        all_refine_outputs.extend(refine_outputs)

    return merged_score_map, coarse_score_map.copy(), all_refine_outputs


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
