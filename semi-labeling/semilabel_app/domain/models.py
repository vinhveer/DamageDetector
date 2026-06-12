from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BBox = tuple[float, float, float, float] | None


@dataclass(frozen=True)
class QueueItem:
    result_id: int
    image_rel_path: str
    crop_path: str
    image_uri: str
    crop_uri: str
    initial_label: str
    suggested_label: str
    queue_type: str
    reliability_score: float
    reasons: tuple[str, ...]
    box: BBox
    decided_action: str
    decided_label: str


@dataclass(frozen=True)
class CleanedItem:
    result_id: int
    image_rel_path: str
    crop_path: str
    final_label: str
    export_label: str
    decision_type: str
    reliability_score: float
    box: BBox
    crop_uri: str
    image_uri: str
    reasons: tuple[str, ...] = ()
    decision_policy_run_id: str = ""


@dataclass(frozen=True)
class Candidate:
    result_id: int
    label: str
    predicted_label: str
    reliability_score: float
    crop_uri: str
    image_uri: str
    box: BBox
    cluster_id: str
    domain_index: int | None
    cluster_size: int
    centroid_similarity: float | None
    model_agreement: float = 0.0
    image_rel_path: str = ""


@dataclass(frozen=True)
class Group:
    """A DINOv2 visual domain / core cluster produced in the prepare step.

    Each group bundles near-duplicate crops of one label.  ``rep_*`` describes
    the representative member (highest centroid similarity) used as the group
    thumbnail.
    """

    core_cluster_id: str
    label: str
    size: int
    member_count: int
    domain_index: int
    status: str
    rep_result_id: int | None
    rep_similarity: float
    rep_image_rel_path: str
    rep_box: BBox
    rep_image_uri: str


@dataclass(frozen=True)
class ClassDist:
    total: int
    by_label: list[tuple[str, int, float]]
    by_decision_type: list[tuple[str, int, float]]


@dataclass(frozen=True)
class ChainStep:
    key: str
    module: str
    flags: dict[str, Any]
