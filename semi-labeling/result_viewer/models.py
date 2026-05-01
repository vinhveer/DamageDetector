from __future__ import annotations

from dataclasses import dataclass


LABELS = ("crack", "mold", "spall")


@dataclass(frozen=True)
class GroupRun:
    grouping_run_id: str
    created_at_utc: str
    source_db_path: str
    filtered_db_path: str
    source_filter_run_id: str
    model_name: str
    device: str
    total_boxes: int
    total_clusters: int
    outlier_boxes: int
    label_suspect_boxes: int


@dataclass(frozen=True)
class ClusterSummary:
    cluster_key: str
    predicted_label_scope: str
    cluster_id: int
    cluster_size: int
    major_label: str
    purity: float
    crack_count: int
    mold_count: int
    spall_count: int
    outlier_count: int
    representative_nearest_result_id: int | None
    representative_farthest_result_id: int | None
    representative_low_confidence_result_id: int | None
    representative_mismatch_result_id: int | None


@dataclass(frozen=True)
class AssignmentRow:
    result_id: int
    image_rel_path: str
    predicted_label: str
    predicted_probability_pct: float
    detector_score: float
    cluster_key: str
    is_outlier: int
    distance_to_center: float
    suggested_label: str
    label_suspect: int
    cluster_purity: float
    cluster_size: int
    image_path: str = ""
    source_input_dir: str = ""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


@dataclass(frozen=True)
class SourceMeta:
    result_id: int
    image_path: str
    source_input_dir: str
    x1: float
    y1: float
    x2: float
    y2: float
