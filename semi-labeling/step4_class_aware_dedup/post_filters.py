from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from output_store import DedupDecision
from pair_features import cosine_similarity
from source_store import Detection


Bbox = tuple[float, float, float, float]


def effective_bbox(decision: DedupDecision, detection: Detection) -> Bbox:
    if decision.fused and decision.fused_bbox is not None and len(decision.fused_bbox) == 4:
        return (
            float(decision.fused_bbox[0]),
            float(decision.fused_bbox[1]),
            float(decision.fused_bbox[2]),
            float(decision.fused_bbox[3]),
        )
    return (float(detection.x1), float(detection.y1), float(detection.x2), float(detection.y2))


def bbox_area(box: Bbox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(parent: Bbox, child: Bbox) -> float:
    ix1 = max(parent[0], child[0])
    iy1 = max(parent[1], child[1])
    ix2 = min(parent[2], child[2])
    iy2 = min(parent[3], child[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def containment_of_child(parent: Bbox, child: Bbox) -> float:
    child_area = bbox_area(child)
    if child_area <= 0.0:
        return 0.0
    return intersection_area(parent, child) / child_area


def area_ratio_of(decision: DedupDecision, detection: Detection) -> float:
    bbox = effective_bbox(decision, detection)
    image_area = float(detection.image_area)
    if image_area <= 0.0:
        return 0.0
    return bbox_area(bbox) / image_area


def _clip_bbox(child: Bbox, parent: Bbox) -> Bbox:
    return (
        max(child[0], parent[0]),
        max(child[1], parent[1]),
        min(child[2], parent[2]),
        min(child[3], parent[3]),
    )


def _union_coverage_ratio(parent: Bbox, children: list[Bbox]) -> float:
    parent_area = bbox_area(parent)
    if parent_area <= 0.0 or not children:
        return 0.0
    px1, py1, px2, py2 = parent
    width = int(max(1, round(px2 - px1)))
    height = int(max(1, round(py2 - py1)))
    if width * height > 4_000_000:
        # Downsample for very large parents to avoid blowing up memory.
        scale = (4_000_000 / float(width * height)) ** 0.5
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
    if width <= 0 or height <= 0:
        return 0.0
    sx = width / max(1e-6, px2 - px1)
    sy = height / max(1e-6, py2 - py1)
    mask = np.zeros((height, width), dtype=np.uint8)
    for child in children:
        cx1, cy1, cx2, cy2 = _clip_bbox(child, parent)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        ix1 = max(0, int((cx1 - px1) * sx))
        iy1 = max(0, int((cy1 - py1) * sy))
        ix2 = min(width, int(round((cx2 - px1) * sx)))
        iy2 = min(height, int(round((cy2 - py1) * sy)))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        mask[iy1:iy2, ix1:ix2] = 1
    covered = float(mask.sum())
    return covered / float(width * height)


def _feature_coverage(
    parent_emb: np.ndarray | None,
    child_embs: list[np.ndarray],
) -> float:
    if parent_emb is None or not child_embs:
        return 0.0
    summed = np.zeros_like(np.asarray(parent_emb, dtype=np.float32))
    for emb in child_embs:
        if emb is None:
            continue
        summed = summed + np.asarray(emb, dtype=np.float32).reshape(-1)
    if float(np.linalg.norm(summed)) <= 1e-12:
        return 0.0
    return float(cosine_similarity(parent_emb, summed))


def apply_oversized_drop(
    decisions: list[DedupDecision],
    detections_by_id: dict[int, Detection],
    *,
    oversized_area_ratio: float,
) -> list[DedupDecision]:
    threshold = float(oversized_area_ratio)
    if threshold >= 1.0:
        return decisions
    out: list[DedupDecision] = []
    for decision in decisions:
        if not decision.keep:
            out.append(decision)
            continue
        detection = detections_by_id.get(int(decision.result_id))
        if detection is None:
            out.append(decision)
            continue
        if area_ratio_of(decision, detection) < threshold:
            out.append(decision)
            continue
        out.append(replace(decision, keep=False, fused=False, drop_reason="oversized"))
    return out


def apply_prime_box(
    decisions: list[DedupDecision],
    detections_by_id: dict[int, Detection],
    embedding_map: dict[int, np.ndarray],
    *,
    container_area_ratio: float,
    container_overlap_threshold: float,
    prime_geom_threshold: float,
    prime_feat_threshold: float,
) -> list[DedupDecision]:
    container_threshold = float(container_area_ratio)
    overlap_threshold = float(container_overlap_threshold)
    geom_threshold = float(prime_geom_threshold)
    feat_threshold = float(prime_feat_threshold)

    kept_by_image: dict[str, list[DedupDecision]] = defaultdict(list)
    for decision in decisions:
        if decision.keep:
            kept_by_image[decision.image_rel_path].append(decision)

    force_drop_ids: set[int] = set()
    for image_decisions in kept_by_image.values():
        if len(image_decisions) < 2:
            continue
        bbox_cache: dict[int, Bbox] = {}
        area_cache: dict[int, float] = {}
        ratio_cache: dict[int, float] = {}
        label_cache: dict[int, str] = {}
        for decision in image_decisions:
            detection = detections_by_id.get(int(decision.result_id))
            if detection is None:
                continue
            bbox = effective_bbox(decision, detection)
            bbox_cache[int(decision.result_id)] = bbox
            area_cache[int(decision.result_id)] = bbox_area(bbox)
            ratio_cache[int(decision.result_id)] = area_ratio_of(decision, detection)
            label_cache[int(decision.result_id)] = str(decision.predicted_label)

        for parent in image_decisions:
            parent_id = int(parent.result_id)
            if parent_id in force_drop_ids:
                continue
            if ratio_cache.get(parent_id, 0.0) < container_threshold:
                continue
            parent_bbox = bbox_cache.get(parent_id)
            parent_area = area_cache.get(parent_id, 0.0)
            if parent_bbox is None or parent_area <= 0.0:
                continue
            parent_label = label_cache.get(parent_id, "")
            child_bboxes: list[Bbox] = []
            child_embs: list[np.ndarray] = []
            for child in image_decisions:
                child_id = int(child.result_id)
                if child_id == parent_id or child_id in force_drop_ids:
                    continue
                if label_cache.get(child_id, "") != parent_label:
                    continue
                child_bbox = bbox_cache.get(child_id)
                child_area = area_cache.get(child_id, 0.0)
                if child_bbox is None or child_area <= 0.0:
                    continue
                if parent_area <= child_area * 1.2:
                    continue
                if containment_of_child(parent_bbox, child_bbox) < overlap_threshold:
                    continue
                child_bboxes.append(child_bbox)
                emb = embedding_map.get(child_id)
                if emb is not None:
                    child_embs.append(emb)
            if not child_bboxes:
                continue
            geom_cov = _union_coverage_ratio(parent_bbox, child_bboxes)
            if geom_cov < geom_threshold:
                continue
            parent_emb = embedding_map.get(parent_id)
            feat_cov = _feature_coverage(parent_emb, child_embs)
            if feat_cov < feat_threshold:
                continue
            force_drop_ids.add(parent_id)

    if not force_drop_ids:
        return decisions

    out: list[DedupDecision] = []
    for decision in decisions:
        if int(decision.result_id) in force_drop_ids:
            out.append(replace(decision, keep=False, fused=False, drop_reason="prime_box_covered"))
        else:
            out.append(decision)
    return out
