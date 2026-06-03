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
    pred_label: str | None
    pred_prob: float
    pred_margin: float
    second_label: str
    second_prob: float | None
    disagrees_with_policy: bool
    policy_label: str
    defer_reasons: tuple[str, ...]


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
    self_training_run_id: str = ""
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
