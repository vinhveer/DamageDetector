from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KeptBox:
    result_id: int
    source_detection_id: int
    image_id: int
    image_rel_path: str
    image_path: str
    source_input_dir: str
    predicted_label: str
    predicted_probability_pct: float
    detector_score: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int
    filter_run_id: str

    @property
    def area(self) -> float:
        return max(0.0, float(self.x2) - float(self.x1)) * max(0.0, float(self.y2) - float(self.y1))

    @property
    def image_area(self) -> float:
        return max(1.0, float(self.image_width) * float(self.image_height))

    @property
    def area_ratio(self) -> float:
        return self.area / self.image_area


@dataclass(frozen=True)
class FeatureGroupConfig:
    source_db_path: Path
    filtered_db_path: Path
    image_root: Path | None
    output_db_path: Path
    filter_run_id: str
    model_name: str
    device: str
    batch_size: int
    labels: tuple[str, ...]
    limit: int
    padding_ratio: float
    pca_dim: int
    cluster_method: str
    dbscan_eps: float
    dbscan_min_samples: int
    agglomerative_distance_threshold: float
    cluster_per_label: bool
    label_suspect_purity_threshold: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_db_path": str(self.source_db_path),
            "filtered_db_path": str(self.filtered_db_path),
            "image_root": "" if self.image_root is None else str(self.image_root),
            "filter_run_id": self.filter_run_id,
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "labels": list(self.labels),
            "limit": self.limit,
            "padding_ratio": self.padding_ratio,
            "pca_dim": self.pca_dim,
            "cluster_method": self.cluster_method,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "agglomerative_distance_threshold": self.agglomerative_distance_threshold,
            "cluster_per_label": self.cluster_per_label,
            "label_suspect_purity_threshold": self.label_suspect_purity_threshold,
        }


@dataclass(frozen=True)
class ClusterAssignment:
    result_id: int
    source_detection_id: int
    image_rel_path: str
    predicted_label: str
    predicted_probability_pct: float
    detector_score: float
    label_scope: str
    cluster_id: int
    cluster_key: str
    is_outlier: bool
    distance_to_center: float
    suggested_label: str
    label_suspect: bool
    cluster_purity: float
    cluster_size: int


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
