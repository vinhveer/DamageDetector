from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .decision_policy import SemanticDecision
from shared.db.source_store import SourceDetection


@dataclass(frozen=True)
class BBoxQualityConfig:
    containment_threshold: float = 0.90
    graph_containment_min: float = 0.50
    graph_iou_min: float = 0.10
    quality_tie_margin: float = 0.05
    composite_child_min: int = 2
    broad_area_ratio_to_image: float = 0.25
    broad_child_coverage_max: float = 0.35
    broad_semantic_confidence_max: float = 0.65
    long_crack_elongation_min: float = 3.0
    long_crack_child_ratio_min: float = 0.75
    long_crack_alignment_min: float = 0.60


@dataclass(frozen=True)
class BoxGeometry:
    result_id: int
    image_rel_path: str
    label: str
    detector_score: float
    semantic_confidence: float
    semantic_margin: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int
    area: float
    area_ratio_to_image: float
    aspect_ratio: float
    elongation: float
    center_x: float
    center_y: float


@dataclass(frozen=True)
class BoxGraphEdge:
    parent_result_id: int
    child_result_id: int
    image_rel_path: str
    iou: float
    intersection_area: float
    containment_small_in_large: float
    child_coverage_of_parent: float
    area_ratio: float
    center_distance_norm: float
    aspect_ratio_similarity: float
    label_agreement: bool
    edge_type: str
    features: dict[str, float | str | bool]

    @property
    def features_json(self) -> str:
        return json.dumps(self.features, ensure_ascii=False, sort_keys=True)


@dataclass
class BoxQualityScore:
    result_id: int
    image_rel_path: str
    label: str
    box_quality_score: float
    detector_score: float
    semantic_confidence: float
    semantic_margin: float
    crop_consistency: float | None
    embedding_core_similarity: float | None
    prototype_similarity: float | None
    area_ratio_to_image: float
    aspect_ratio: float
    elongation: float
    child_count: int
    child_label_diversity: int
    child_alignment_score: float
    background_context_penalty: float
    composite_penalty: float
    components: dict[str, float | str | int | None] = field(default_factory=dict)

    @property
    def components_json(self) -> str:
        return json.dumps(self.components, ensure_ascii=False, sort_keys=True)


@dataclass
class BoxCleanupDecision:
    result_id: int
    image_rel_path: str
    label: str
    decision_type: str
    keep_for_cleaned: bool
    box_quality_score: float
    representative_id: int | None
    reason_codes: list[str]
    features: dict[str, float | str | int | bool | None]

    @property
    def reason_codes_json(self) -> str:
        return json.dumps(sorted(set(self.reason_codes)), ensure_ascii=False, sort_keys=True)

    @property
    def features_json(self) -> str:
        return json.dumps(self.features, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class BBoxQualityResult:
    run_id: str
    box_graph_run_id: str
    options: dict[str, float]
    edges: list[BoxGraphEdge]
    quality_scores: list[BoxQualityScore]
    decisions: list[BoxCleanupDecision]

    @property
    def decisions_by_result_id(self) -> dict[int, BoxCleanupDecision]:
        return {item.result_id: item for item in self.decisions}

    @property
    def review_decisions(self) -> list[BoxCleanupDecision]:
        return [item for item in self.decisions if item.decision_type in {"suspect_composite_box", "suspect_broad_box", "manual_box_review"}]


def run_bbox_quality_filter(
    *,
    run_id: str,
    detections: list[SourceDetection],
    semantic_decisions: list[SemanticDecision],
    config: BBoxQualityConfig | None = None,
) -> BBoxQualityResult:
    cfg = config or BBoxQualityConfig()
    box_graph_run_id = f"{run_id}_box_graph_v1"
    decisions_by_id = {item.result_id: item for item in semantic_decisions}
    geometries = [_build_geometry(item, decisions_by_id[item.result_id]) for item in detections if item.result_id in decisions_by_id]
    by_image: dict[str, list[BoxGeometry]] = defaultdict(list)
    for geometry in geometries:
        by_image[geometry.image_rel_path].append(geometry)

    edges: list[BoxGraphEdge] = []
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    parent_by_child: dict[int, list[int]] = defaultdict(list)
    geometry_by_id = {item.result_id: item for item in geometries}
    for image_boxes in by_image.values():
        group_edges = _build_edges(image_boxes, cfg)
        edges.extend(group_edges)
        for edge in group_edges:
            if edge.containment_small_in_large >= cfg.containment_threshold:
                children_by_parent[edge.parent_result_id].append(edge.child_result_id)
                parent_by_child[edge.child_result_id].append(edge.parent_result_id)

    quality_scores = _compute_quality_scores(geometries, children_by_parent, geometry_by_id, cfg)
    quality_by_id = {item.result_id: item for item in quality_scores}
    cleanup_decisions = _decide_cleanup(geometries, edges, children_by_parent, parent_by_child, quality_by_id, cfg)
    return BBoxQualityResult(
        run_id=run_id,
        box_graph_run_id=box_graph_run_id,
        options={
            "containment_threshold": cfg.containment_threshold,
            "graph_containment_min": cfg.graph_containment_min,
            "graph_iou_min": cfg.graph_iou_min,
            "quality_tie_margin": cfg.quality_tie_margin,
            "composite_child_min": cfg.composite_child_min,
            "broad_area_ratio_to_image": cfg.broad_area_ratio_to_image,
            "broad_child_coverage_max": cfg.broad_child_coverage_max,
            "broad_semantic_confidence_max": cfg.broad_semantic_confidence_max,
            "long_crack_elongation_min": cfg.long_crack_elongation_min,
            "long_crack_child_ratio_min": cfg.long_crack_child_ratio_min,
            "long_crack_alignment_min": cfg.long_crack_alignment_min,
        },
        edges=edges,
        quality_scores=quality_scores,
        decisions=cleanup_decisions,
    )


def _build_geometry(detection: SourceDetection, semantic_decision: SemanticDecision) -> BoxGeometry:
    width = max(0.0, float(detection.x2) - float(detection.x1))
    height = max(0.0, float(detection.y2) - float(detection.y1))
    area = width * height
    image_area = max(1.0, float(detection.image_width) * float(detection.image_height))
    aspect_ratio = width / max(height, 1e-6)
    elongation = max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-6))
    semantic_confidence = float(semantic_decision.score_components.get("semantic_confidence", 0.0))
    return BoxGeometry(
        result_id=detection.result_id,
        image_rel_path=detection.image_rel_path,
        label=semantic_decision.final_label,
        detector_score=max(0.0, min(1.0, float(detection.detector_score))),
        semantic_confidence=max(0.0, min(1.0, semantic_confidence)),
        semantic_margin=max(0.0, min(1.0, float(semantic_decision.top1_top2_margin))),
        x1=float(detection.x1),
        y1=float(detection.y1),
        x2=float(detection.x2),
        y2=float(detection.y2),
        image_width=int(detection.image_width),
        image_height=int(detection.image_height),
        area=area,
        area_ratio_to_image=area / image_area,
        aspect_ratio=aspect_ratio,
        elongation=elongation,
        center_x=(float(detection.x1) + float(detection.x2)) / 2.0,
        center_y=(float(detection.y1) + float(detection.y2)) / 2.0,
    )


def _build_edges(image_boxes: list[BoxGeometry], config: BBoxQualityConfig) -> list[BoxGraphEdge]:
    edges: list[BoxGraphEdge] = []
    for index, left in enumerate(image_boxes):
        for right in image_boxes[index + 1 :]:
            intersection = _intersection_area(left, right)
            if intersection <= 0.0:
                continue
            union = max(left.area + right.area - intersection, 1e-6)
            iou = intersection / union
            smaller_area = max(min(left.area, right.area), 1e-6)
            containment = intersection / smaller_area
            if containment < config.graph_containment_min and iou < config.graph_iou_min:
                continue
            parent, child = (left, right) if left.area >= right.area else (right, left)
            parent_area = max(parent.area, 1e-6)
            area_ratio = parent.area / max(child.area, 1e-6)
            center_distance = math.hypot(parent.center_x - child.center_x, parent.center_y - child.center_y)
            image_diag = math.hypot(max(parent.image_width, child.image_width), max(parent.image_height, child.image_height))
            center_distance_norm = center_distance / max(image_diag, 1e-6)
            aspect_ratio_similarity = min(parent.aspect_ratio, child.aspect_ratio) / max(parent.aspect_ratio, child.aspect_ratio, 1e-6)
            label_agreement = parent.label == child.label
            edge_type = "containment" if containment >= config.containment_threshold else "overlap"
            features = {
                "parent_label": parent.label,
                "child_label": child.label,
                "parent_area": parent.area,
                "child_area": child.area,
                "parent_elongation": parent.elongation,
                "child_elongation": child.elongation,
            }
            edges.append(
                BoxGraphEdge(
                    parent_result_id=parent.result_id,
                    child_result_id=child.result_id,
                    image_rel_path=parent.image_rel_path,
                    iou=iou,
                    intersection_area=intersection,
                    containment_small_in_large=containment,
                    child_coverage_of_parent=intersection / parent_area,
                    area_ratio=area_ratio,
                    center_distance_norm=center_distance_norm,
                    aspect_ratio_similarity=aspect_ratio_similarity,
                    label_agreement=label_agreement,
                    edge_type=edge_type,
                    features=features,
                )
            )
    return edges


def _compute_quality_scores(
    geometries: list[BoxGeometry],
    children_by_parent: dict[int, list[int]],
    geometry_by_id: dict[int, BoxGeometry],
    config: BBoxQualityConfig,
) -> list[BoxQualityScore]:
    scores: list[BoxQualityScore] = []
    for geometry in geometries:
        children = [geometry_by_id[item] for item in children_by_parent.get(geometry.result_id, []) if item in geometry_by_id]
        child_labels = {item.label for item in children}
        child_label_diversity = len(child_labels)
        child_alignment_score = _child_alignment_score(children)
        composite_penalty = 1.0 if len(children) >= config.composite_child_min and child_label_diversity > 1 else 0.0
        background_context_penalty = 1.0 if (
            geometry.area_ratio_to_image >= config.broad_area_ratio_to_image
            and _child_coverage(children, geometry) <= config.broad_child_coverage_max
            and geometry.semantic_confidence <= config.broad_semantic_confidence_max
        ) else 0.0
        box_quality = (
            0.25 * geometry.detector_score
            + 0.25 * geometry.semantic_confidence
            + 0.15 * min(1.0, geometry.semantic_margin / 0.10)
            - 0.20 * background_context_penalty
            - 0.20 * composite_penalty
        )
        box_quality = max(0.0, min(1.0, box_quality))
        components = {
            "detector_score": geometry.detector_score,
            "semantic_confidence": geometry.semantic_confidence,
            "semantic_margin": geometry.semantic_margin,
            "crop_consistency": None,
            "embedding_core_similarity": None,
            "prototype_similarity": None,
            "background_context_penalty": background_context_penalty,
            "composite_penalty": composite_penalty,
            "missing_components_note": "crop_consistency, embedding_core_similarity, and prototype_similarity are not simulated.",
        }
        scores.append(
            BoxQualityScore(
                result_id=geometry.result_id,
                image_rel_path=geometry.image_rel_path,
                label=geometry.label,
                box_quality_score=box_quality,
                detector_score=geometry.detector_score,
                semantic_confidence=geometry.semantic_confidence,
                semantic_margin=geometry.semantic_margin,
                crop_consistency=None,
                embedding_core_similarity=None,
                prototype_similarity=None,
                area_ratio_to_image=geometry.area_ratio_to_image,
                aspect_ratio=geometry.aspect_ratio,
                elongation=geometry.elongation,
                child_count=len(children),
                child_label_diversity=child_label_diversity,
                child_alignment_score=child_alignment_score,
                background_context_penalty=background_context_penalty,
                composite_penalty=composite_penalty,
                components=components,
            )
        )
    return scores


def _decide_cleanup(
    geometries: list[BoxGeometry],
    edges: list[BoxGraphEdge],
    children_by_parent: dict[int, list[int]],
    parent_by_child: dict[int, list[int]],
    quality_by_id: dict[int, BoxQualityScore],
    config: BBoxQualityConfig,
) -> list[BoxCleanupDecision]:
    geometry_by_id = {item.result_id: item for item in geometries}
    decisions = {
        item.result_id: BoxCleanupDecision(
            result_id=item.result_id,
            image_rel_path=item.image_rel_path,
            label=item.label,
            decision_type="keep_representative",
            keep_for_cleaned=True,
            box_quality_score=quality_by_id[item.result_id].box_quality_score,
            representative_id=item.result_id,
            reason_codes=["no_box_conflict"],
            features=_base_decision_features(item, quality_by_id[item.result_id]),
        )
        for item in geometries
        if item.result_id in quality_by_id
    }

    for geometry in geometries:
        quality = quality_by_id[geometry.result_id]
        child_ids = children_by_parent.get(geometry.result_id, [])
        children = [geometry_by_id[item] for item in child_ids if item in geometry_by_id]
        if _is_long_crack_parent(geometry, children, quality, config):
            decisions[geometry.result_id] = _make_decision(geometry, quality, "keep_long_crack_parent", True, geometry.result_id, ["long_crack_geometry", "crack_children_aligned"])
        elif quality.child_count >= config.composite_child_min and quality.child_label_diversity > 1:
            decisions[geometry.result_id] = _make_decision(geometry, quality, "suspect_composite_box", False, None, ["parent_contains_many_children", "child_label_diversity_high"])
        elif quality.background_context_penalty > 0.0:
            decisions[geometry.result_id] = _make_decision(geometry, quality, "suspect_broad_box", False, None, ["parent_too_broad"])

    containment_edges = [edge for edge in edges if edge.containment_small_in_large >= config.containment_threshold and edge.label_agreement]
    for edge in sorted(containment_edges, key=lambda item: item.area_ratio, reverse=True):
        parent = geometry_by_id.get(edge.parent_result_id)
        child = geometry_by_id.get(edge.child_result_id)
        if parent is None or child is None:
            continue
        parent_quality = quality_by_id[parent.result_id]
        child_quality = quality_by_id[child.result_id]
        parent_decision = decisions[parent.result_id]
        if parent_decision.decision_type in {"suspect_composite_box", "suspect_broad_box", "manual_box_review"}:
            continue
        if parent_decision.decision_type == "keep_long_crack_parent":
            decisions[child.result_id] = _make_decision(child, child_quality, "drop_nested_duplicate", False, parent.result_id, ["high_containment_same_label", "contained_by_long_crack_parent"])
            continue
        quality_delta = parent_quality.box_quality_score - child_quality.box_quality_score
        if abs(quality_delta) < config.quality_tie_margin:
            keep_parent = False
            reason = ["high_containment_same_label", "quality_margin_small"]
        else:
            keep_parent = quality_delta > 0.0
            reason = ["high_containment_same_label"]
        if keep_parent:
            decisions[parent.result_id] = _make_decision(parent, parent_quality, "keep_representative", True, parent.result_id, reason)
            decisions[child.result_id] = _make_decision(child, child_quality, "drop_nested_duplicate", False, parent.result_id, reason)
        else:
            decisions[child.result_id] = _make_decision(child, child_quality, "keep_representative", True, child.result_id, reason)
            decisions[parent.result_id] = _make_decision(parent, parent_quality, "drop_nested_duplicate", False, child.result_id, reason)

    for geometry in geometries:
        if geometry.result_id not in parent_by_child and geometry.result_id not in children_by_parent:
            continue
        decision = decisions[geometry.result_id]
        if decision.decision_type == "keep_representative" and decision.reason_codes == ["no_box_conflict"]:
            quality = quality_by_id[geometry.result_id]
            decisions[geometry.result_id] = _make_decision(geometry, quality, "keep_representative", True, geometry.result_id, ["box_graph_checked"])

    return [decisions[item.result_id] for item in geometries if item.result_id in decisions]


def _make_decision(
    geometry: BoxGeometry,
    quality: BoxQualityScore,
    decision_type: str,
    keep_for_cleaned: bool,
    representative_id: int | None,
    reason_codes: list[str],
) -> BoxCleanupDecision:
    return BoxCleanupDecision(
        result_id=geometry.result_id,
        image_rel_path=geometry.image_rel_path,
        label=geometry.label,
        decision_type=decision_type,
        keep_for_cleaned=keep_for_cleaned,
        box_quality_score=quality.box_quality_score,
        representative_id=representative_id,
        reason_codes=reason_codes,
        features=_base_decision_features(geometry, quality),
    )


def _base_decision_features(geometry: BoxGeometry, quality: BoxQualityScore) -> dict[str, float | str | int | bool | None]:
    return {
        "area_ratio_to_image": geometry.area_ratio_to_image,
        "aspect_ratio": geometry.aspect_ratio,
        "elongation": geometry.elongation,
        "child_count": quality.child_count,
        "child_label_diversity": quality.child_label_diversity,
        "child_alignment_score": quality.child_alignment_score,
        "background_context_penalty": quality.background_context_penalty,
        "composite_penalty": quality.composite_penalty,
    }


def _is_long_crack_parent(geometry: BoxGeometry, children: list[BoxGeometry], quality: BoxQualityScore, config: BBoxQualityConfig) -> bool:
    if geometry.label != "crack" or geometry.elongation < config.long_crack_elongation_min:
        return False
    if not children:
        return geometry.semantic_confidence >= 0.75 and geometry.semantic_margin >= 0.05
    crack_ratio = sum(1 for item in children if item.label == "crack") / max(len(children), 1)
    return crack_ratio >= config.long_crack_child_ratio_min and quality.child_alignment_score >= config.long_crack_alignment_min


def _intersection_area(left: BoxGeometry, right: BoxGeometry) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _child_coverage(children: list[BoxGeometry], parent: BoxGeometry) -> float:
    if not children:
        return 0.0
    child_area = sum(_intersection_area(parent, child) for child in children)
    return min(1.0, child_area / max(parent.area, 1e-6))


def _child_alignment_score(children: list[BoxGeometry]) -> float:
    if len(children) < 2:
        return 1.0 if children else 0.0
    centers = [(item.center_x, item.center_y) for item in children]
    mean_x = sum(x for x, _ in centers) / len(centers)
    mean_y = sum(y for _, y in centers) / len(centers)
    xx = sum((x - mean_x) ** 2 for x, _ in centers) / len(centers)
    yy = sum((y - mean_y) ** 2 for _, y in centers) / len(centers)
    xy = sum((x - mean_x) * (y - mean_y) for x, y in centers) / len(centers)
    trace = xx + yy
    det = xx * yy - xy * xy
    disc = max(0.0, trace * trace - 4.0 * det)
    lambda1 = (trace + math.sqrt(disc)) / 2.0
    lambda2 = (trace - math.sqrt(disc)) / 2.0
    if lambda1 <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (lambda2 / lambda1)))
