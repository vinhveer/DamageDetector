from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (matrix / norms).astype(np.float32, copy=False)


@dataclass(frozen=True)
class PCAResult:
    transformed: np.ndarray          # (N, n_components), L2-normalized
    explained_variance_ratio: float  # cumulative variance preserved


def reduce_pca(embeddings: np.ndarray, *, n_components: int, random_state: int = 42) -> PCAResult:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        normalized = _l2_normalize(matrix.reshape(matrix.shape[0], -1)) if matrix.size > 0 else matrix
        return PCAResult(transformed=normalized.astype(np.float32, copy=False), explained_variance_ratio=0.0)

    if int(n_components) <= 0 or int(n_components) >= matrix.shape[1]:
        normalized = _l2_normalize(matrix)
        return PCAResult(transformed=normalized, explained_variance_ratio=1.0)

    from sklearn.decomposition import PCA

    target_dim = max(1, min(int(n_components), matrix.shape[1], matrix.shape[0]))
    pca = PCA(n_components=target_dim, random_state=int(random_state))
    transformed = pca.fit_transform(matrix).astype(np.float32, copy=False)
    explained = float(np.sum(pca.explained_variance_ratio_))
    transformed = _l2_normalize(transformed)
    return PCAResult(transformed=transformed, explained_variance_ratio=explained)


@dataclass(frozen=True)
class ClusterFit:
    labels: np.ndarray       # (N,) int32 cluster id per box
    centroids: np.ndarray    # (K, D) cluster centroid in PCA space (L2-normalized)
    k: int


def fit_clusters(
    matrix: np.ndarray,
    *,
    k: int,
    random_state: int = 42,
    batch_size: int = 4096,
    max_iter: int = 300,
    n_init: int = 5,
) -> ClusterFit:
    from sklearn.cluster import MiniBatchKMeans

    n = int(matrix.shape[0])
    k_clamped = max(1, min(int(k), max(1, n)))
    if k_clamped == 1 or n <= 1:
        labels = np.zeros((n,), dtype=np.int32)
        centroid = matrix.mean(axis=0, keepdims=True) if n > 0 else matrix
        centroid = _l2_normalize(centroid) if centroid.size else centroid
        return ClusterFit(labels=labels, centroids=centroid, k=1)

    model = MiniBatchKMeans(
        n_clusters=k_clamped,
        random_state=int(random_state),
        batch_size=int(batch_size),
        max_iter=int(max_iter),
        n_init=int(n_init),
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(matrix).astype(np.int32, copy=False)
    centroids = _l2_normalize(np.asarray(model.cluster_centers_, dtype=np.float32))
    return ClusterFit(labels=labels, centroids=centroids, k=k_clamped)


def compute_distances_to_centroid(
    matrix: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Cosine distance (1 - cos_sim) for each box vs its assigned centroid."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    chosen = centroids[labels]                  # (N, D)
    sims = np.sum(matrix * chosen, axis=1)      # cos_sim since both L2-normalized
    sims = np.clip(sims, -1.0, 1.0)
    return (1.0 - sims).astype(np.float32, copy=False)


def rank_within_clusters(
    distances: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """For each box, rank within its cluster: 0 = closest to centroid."""
    ranks = np.full(labels.shape, -1, dtype=np.int32)
    for cluster_id in range(int(k)):
        idx = np.where(labels == cluster_id)[0]
        if idx.size == 0:
            continue
        order = np.argsort(distances[idx], kind="stable")
        ranks[idx[order]] = np.arange(idx.size, dtype=np.int32)
    return ranks
