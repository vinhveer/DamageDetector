from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.exceptions import EfficiencyWarning
from sklearn.neighbors import NearestNeighbors, sort_graph_by_row_values
from sklearn.preprocessing import normalize

from models import ClusterAssignment, ClusterSummary, FeatureGroupConfig, KeptBox


LABELS = ("crack", "mold", "spall")


def to_numpy(embeddings: Any) -> np.ndarray:
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={arr.shape}")
    return normalize(arr, norm="l2")


def reduce_features(embeddings: np.ndarray, *, pca_dim: int) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings
    max_components = min(int(pca_dim), embeddings.shape[0], embeddings.shape[1])
    if max_components < 2 or max_components >= embeddings.shape[1]:
        return normalize(embeddings, norm="l2")
    reduced = PCA(n_components=max_components, random_state=0).fit_transform(embeddings)
    return normalize(reduced, norm="l2")


def cluster_features(features: np.ndarray, config: FeatureGroupConfig) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0,), dtype=np.int32)
    method = str(config.cluster_method).strip().lower()
    if method == "dbscan":
        neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1).fit(features)
        graph = neighbors.radius_neighbors_graph(
            features,
            radius=float(config.dbscan_eps),
            mode="distance",
        )
        graph = sort_graph_by_row_values(graph.maximum(graph.T), warn_when_not_sorted=False)
        model = DBSCAN(
            eps=float(config.dbscan_eps),
            min_samples=int(config.dbscan_min_samples),
            metric="precomputed",
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=EfficiencyWarning, message="Precomputed sparse input was not sorted.*")
            return model.fit_predict(graph).astype(np.int32)
    if method in {"agglomerative", "agglomerativeclustering"}:
        if len(features) == 1:
            return np.zeros((1,), dtype=np.int32)
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=float(config.agglomerative_distance_threshold),
        )
        return model.fit_predict(features).astype(np.int32)
    raise ValueError(f"Unsupported cluster_method={config.cluster_method!r}")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - float(np.dot(a, b)))


def label_counts(boxes: list[KeptBox]) -> Counter[str]:
    return Counter(str(box.predicted_label).lower() for box in boxes)


def major_label_and_purity(boxes: list[KeptBox]) -> tuple[str, float]:
    counts = label_counts(boxes)
    if not counts:
        return "", 0.0
    major_label, major_count = counts.most_common(1)[0]
    return major_label, float(major_count) / max(1, len(boxes))


def cluster_key(label_scope: str, cluster_id: int, boxes: list[KeptBox]) -> str:
    if int(cluster_id) == -1:
        return f"{label_scope}:noise:{boxes[0].result_id}"
    return f"{label_scope}:{int(cluster_id)}"


def representative_ids(
    boxes: list[KeptBox],
    features: np.ndarray,
    distances: list[float],
    major_label: str,
) -> tuple[int | None, int | None, int | None, int | None]:
    if not boxes:
        return None, None, None, None
    nearest = boxes[int(np.argmin(np.asarray(distances, dtype=np.float32)))].result_id
    farthest = boxes[int(np.argmax(np.asarray(distances, dtype=np.float32)))].result_id
    low_conf = min(boxes, key=lambda box: float(box.predicted_probability_pct)).result_id
    mismatches = [box for box in boxes if str(box.predicted_label).lower() != major_label]
    mismatch = min(mismatches, key=lambda box: float(box.predicted_probability_pct)).result_id if mismatches else None
    return nearest, farthest, low_conf, mismatch


def build_assignments_for_scope(
    *,
    boxes: list[KeptBox],
    embeddings: np.ndarray,
    label_scope: str,
    config: FeatureGroupConfig,
) -> tuple[list[ClusterAssignment], list[ClusterSummary]]:
    features = reduce_features(embeddings, pca_dim=int(config.pca_dim))
    labels = cluster_features(features, config)
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(labels):
        key = cluster_key(label_scope, int(cluster_id), [boxes[idx]])
        grouped_indices[key].append(idx)

    assignments: list[ClusterAssignment] = []
    summaries: list[ClusterSummary] = []
    for key, indices in grouped_indices.items():
        cluster_boxes = [boxes[idx] for idx in indices]
        cluster_id = int(labels[indices[0]])
        outlier = cluster_id == -1
        cluster_features_arr = features[indices]
        if outlier:
            distances = [1.0 for _ in cluster_boxes]
        else:
            centroid = normalize(cluster_features_arr.mean(axis=0, keepdims=True), norm="l2")[0]
            distances = [cosine_distance(features[idx], centroid) for idx in indices]
        major_label, purity = major_label_and_purity(cluster_boxes)
        label_suspect = purity < float(config.label_suspect_purity_threshold)
        nearest, farthest, low_conf, mismatch = representative_ids(cluster_boxes, cluster_features_arr, distances, major_label)
        counts = label_counts(cluster_boxes)
        summaries.append(
            ClusterSummary(
                cluster_key=key,
                predicted_label_scope=label_scope,
                cluster_id=cluster_id,
                cluster_size=len(cluster_boxes),
                major_label=major_label,
                purity=purity,
                crack_count=int(counts.get("crack", 0)),
                mold_count=int(counts.get("mold", 0)),
                spall_count=int(counts.get("spall", 0)),
                outlier_count=len(cluster_boxes) if outlier else 0,
                representative_nearest_result_id=nearest,
                representative_farthest_result_id=farthest,
                representative_low_confidence_result_id=low_conf,
                representative_mismatch_result_id=mismatch,
            )
        )
        for box, distance in zip(cluster_boxes, distances):
            assignments.append(
                ClusterAssignment(
                    result_id=box.result_id,
                    source_detection_id=box.source_detection_id,
                    image_rel_path=box.image_rel_path,
                    predicted_label=box.predicted_label,
                    predicted_probability_pct=box.predicted_probability_pct,
                    detector_score=box.detector_score,
                    label_scope=label_scope,
                    cluster_id=cluster_id,
                    cluster_key=key,
                    is_outlier=outlier,
                    distance_to_center=float(distance),
                    suggested_label=major_label if str(box.predicted_label).lower() != major_label else "",
                    label_suspect=label_suspect,
                    cluster_purity=purity,
                    cluster_size=len(cluster_boxes),
                )
            )
    return assignments, summaries


def build_feature_groups(
    boxes: list[KeptBox],
    embeddings: np.ndarray,
    config: FeatureGroupConfig,
) -> tuple[list[ClusterAssignment], list[ClusterSummary]]:
    if len(boxes) != len(embeddings):
        raise ValueError(f"Box count and embedding count differ: boxes={len(boxes)} embeddings={len(embeddings)}")
    scopes: dict[str, list[int]] = defaultdict(list)
    if bool(config.cluster_per_label):
        for idx, box in enumerate(boxes):
            scopes[str(box.predicted_label).lower()].append(idx)
    else:
        scopes["all"].extend(range(len(boxes)))

    assignments: list[ClusterAssignment] = []
    summaries: list[ClusterSummary] = []
    for label_scope, indices in sorted(scopes.items()):
        scope_boxes = [boxes[idx] for idx in indices]
        scope_embeddings = embeddings[indices]
        scope_assignments, scope_summaries = build_assignments_for_scope(
            boxes=scope_boxes,
            embeddings=scope_embeddings,
            label_scope=label_scope,
            config=config,
        )
        assignments.extend(scope_assignments)
        summaries.extend(scope_summaries)
    return assignments, summaries
