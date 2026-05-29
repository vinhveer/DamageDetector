from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .embedding_cache import load_embeddings


@dataclass(frozen=True)
class CoreMiningConfig:
    model_name: str
    view_name: str
    embedding_run_id: str = ""
    min_confidence: float = 0.70
    min_agreement: float = 0.75
    core_min_size: int = 10
    max_clusters_per_class: int = 8
    rare_cluster_min_size: int = 3
    random_state: int = 17


@dataclass(frozen=True)
class CoreCluster:
    core_cluster_id: str
    label: str
    centroid: np.ndarray
    member_ids: list[int]
    member_similarities: list[float]
    density_score: float
    agreement_score: float
    status: str


@dataclass(frozen=True)
class CoreOutlier:
    result_id: int
    label: str
    outlier_type: str
    nearest_cluster_id: str | None
    similarity: float | None
    reason_codes: list[str]

    @property
    def reason_codes_json(self) -> str:
        return json.dumps(sorted(set(self.reason_codes)), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class CoreMiningResult:
    core_mining_run_id: str
    embedding_run_id: str
    run_id: str
    model_name: str
    view_name: str
    options: dict[str, float | int | str]
    total_embeddings: int
    clustered_count: int
    clusters: list[CoreCluster]
    outliers: list[CoreOutlier]

    @property
    def rare_count(self) -> int:
        return sum(1 for item in self.outliers if item.outlier_type == "rare_cluster")

    @property
    def noise_count(self) -> int:
        return sum(1 for item in self.outliers if item.outlier_type != "rare_cluster")


def run_core_mining(conn: sqlite3.Connection, *, run_id: str, config: CoreMiningConfig) -> CoreMiningResult:
    embeddings, result_ids, embedding_run = load_embeddings(
        conn,
        model_name=config.model_name,
        view_name=config.view_name,
        run_id=run_id,
        embedding_run_id=config.embedding_run_id or None,
    )
    embedding_run_id = str(embedding_run["embedding_run_id"])
    if embeddings.size == 0:
        raise RuntimeError("No embeddings available for core mining.")
    label_rows = _read_candidate_labels(conn, run_id=run_id, result_ids=result_ids, config=config)
    label_by_id = {int(row["result_id"]): str(row["final_label"] or row["majority_label"] or "unknown") for row in label_rows}
    agreement_by_id = {int(row["result_id"]): float(row["agreement_ratio"] or 0.0) for row in label_rows}
    kept_indices = [idx for idx, result_id in enumerate(result_ids) if int(result_id) in label_by_id]
    if not kept_indices:
        raise RuntimeError("No embeddings matched confidence/agreement filters.")

    ids = [int(result_ids[idx]) for idx in kept_indices]
    matrix = embeddings[kept_indices].astype(np.float32, copy=False)
    labels = [label_by_id[item] for item in ids]
    core_mining_run_id = f"core_{run_id}_{embedding_run_id[:8]}_{config.view_name}"
    clusters: list[CoreCluster] = []
    outliers: list[CoreOutlier] = []
    clustered_count = 0
    for label in sorted(set(labels)):
        class_indices = [idx for idx, item_label in enumerate(labels) if item_label == label]
        class_ids = [ids[idx] for idx in class_indices]
        class_matrix = matrix[class_indices]
        if len(class_ids) < config.core_min_size:
            outliers.extend(
                CoreOutlier(result_id=item, label=label, outlier_type="insufficient_class_samples", nearest_cluster_id=None, similarity=None, reason_codes=["class_below_core_min_size"])
                for item in class_ids
            )
            continue
        n_clusters = max(1, min(config.max_clusters_per_class, len(class_ids) // config.core_min_size))
        assignments = _cluster(class_matrix, n_clusters=n_clusters, random_state=config.random_state)
        for cluster_idx in range(n_clusters):
            member_local = [idx for idx, assigned in enumerate(assignments) if int(assigned) == cluster_idx]
            member_ids = [class_ids[idx] for idx in member_local]
            if not member_ids:
                continue
            member_matrix = class_matrix[member_local]
            centroid = _l2_normalize(member_matrix.mean(axis=0, keepdims=True))[0]
            similarities = np.clip(member_matrix @ centroid, -1.0, 1.0)
            density = float(np.mean(similarities))
            agreement = float(np.mean([agreement_by_id[item] for item in member_ids]))
            cluster_id = f"{core_mining_run_id}_{label}_{cluster_idx:03d}"
            if len(member_ids) >= config.core_min_size and agreement >= config.min_agreement:
                clusters.append(
                    CoreCluster(
                        core_cluster_id=cluster_id,
                        label=label,
                        centroid=centroid.astype(np.float32, copy=False),
                        member_ids=member_ids,
                        member_similarities=[float(item) for item in similarities],
                        density_score=density,
                        agreement_score=agreement,
                        status="core",
                    )
                )
                clustered_count += len(member_ids)
            else:
                outlier_type = "rare_cluster" if len(member_ids) >= config.rare_cluster_min_size else "noise_cluster"
                reason = "cluster_below_core_min_size" if len(member_ids) < config.core_min_size else "cluster_low_agreement"
                outliers.extend(
                    CoreOutlier(
                        result_id=item,
                        label=label,
                        outlier_type=outlier_type,
                        nearest_cluster_id=cluster_id,
                        similarity=float(similarities[pos]),
                        reason_codes=[reason],
                    )
                    for pos, item in enumerate(member_ids)
                )
    return CoreMiningResult(
        core_mining_run_id=core_mining_run_id,
        embedding_run_id=embedding_run_id,
        run_id=run_id,
        model_name=config.model_name,
        view_name=config.view_name,
        options={
            "min_confidence": config.min_confidence,
            "min_agreement": config.min_agreement,
            "core_min_size": config.core_min_size,
            "max_clusters_per_class": config.max_clusters_per_class,
            "rare_cluster_min_size": config.rare_cluster_min_size,
            "random_state": config.random_state,
            "embedding_run_id": config.embedding_run_id,
        },
        total_embeddings=len(result_ids),
        clustered_count=clustered_count,
        clusters=clusters,
        outliers=outliers,
    )


def persist_core_mining_result(conn: sqlite3.Connection, result: CoreMiningResult, *, created_at_utc: str) -> None:
    _delete_existing(conn, result.core_mining_run_id)
    conn.execute(
        """
        INSERT INTO core_mining_runs (
            core_mining_run_id, run_id, embedding_run_id, model_name, view_name,
            created_at_utc, options_json, total_embeddings, clustered_count,
            core_cluster_count, rare_count, noise_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.core_mining_run_id,
            result.run_id,
            result.embedding_run_id,
            result.model_name,
            result.view_name,
            created_at_utc,
            json.dumps(result.options, ensure_ascii=False, sort_keys=True),
            result.total_embeddings,
            result.clustered_count,
            len(result.clusters),
            result.rare_count,
            result.noise_count,
        ),
    )
    for cluster in result.clusters:
        centroid_blob = np.asarray(cluster.centroid, dtype="<f4").tobytes()
        centroid_preview = json.dumps([float(item) for item in cluster.centroid[:16]], ensure_ascii=False)
        conn.execute(
            """
            INSERT OR REPLACE INTO core_clusters (
                run_id, core_cluster_id, label, centroid_json, size, created_at_utc,
                core_mining_run_id, embedding_run_id, view_name, centroid_blob,
                member_count, density_score, agreement_score, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.run_id,
                cluster.core_cluster_id,
                cluster.label,
                centroid_preview,
                len(cluster.member_ids),
                created_at_utc,
                result.core_mining_run_id,
                result.embedding_run_id,
                result.view_name,
                centroid_blob,
                len(cluster.member_ids),
                cluster.density_score,
                cluster.agreement_score,
                cluster.status,
            ),
        )
        member_rows = [
            (
                result.run_id,
                cluster.core_cluster_id,
                int(member_id),
                float(similarity),
                result.core_mining_run_id,
                float(1.0 - similarity),
                1,
            )
            for member_id, similarity in zip(cluster.member_ids, cluster.member_similarities, strict=True)
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO core_cluster_members (
                run_id, core_cluster_id, result_id, similarity,
                core_mining_run_id, distance_to_centroid, is_core_member
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            member_rows,
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO core_outliers (
            core_mining_run_id, result_id, label, outlier_type,
            nearest_cluster_id, similarity, reason_codes_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                result.core_mining_run_id,
                outlier.result_id,
                outlier.label,
                outlier.outlier_type,
                outlier.nearest_cluster_id,
                outlier.similarity,
                outlier.reason_codes_json,
            )
            for outlier in result.outliers
        ],
    )
    conn.commit()


def _read_candidate_labels(conn: sqlite3.Connection, *, run_id: str, result_ids: list[int], config: CoreMiningConfig) -> list[sqlite3.Row]:
    if not result_ids:
        return []
    rows: list[sqlite3.Row] = []
    chunk = 900
    for offset in range(0, len(result_ids), chunk):
        ids = [int(item) for item in result_ids[offset : offset + chunk]]
        placeholders = ", ".join("?" for _ in ids)
        rows.extend(
            conn.execute(
                f"""
                SELECT d.result_id, d.final_label, d.reliability_score,
                       COALESCE(a.majority_label, d.final_label) AS majority_label,
                       COALESCE(a.agreement_ratio, d.model_agreement) AS agreement_ratio
                FROM semantic_decisions d
                LEFT JOIN semantic_agreements a ON a.run_id = d.run_id AND a.result_id = d.result_id
                LEFT JOIN box_cleanup_decisions b ON b.result_id = d.result_id
                    AND b.box_graph_run_id = d.run_id || '_box_graph_v1'
                WHERE d.run_id = ?
                  AND d.result_id IN ({placeholders})
                  AND d.reliability_score >= ?
                  AND COALESCE(a.agreement_ratio, d.model_agreement) >= ?
                  AND COALESCE(b.keep_for_cleaned, 1) = 1
                  AND d.final_label NOT IN ('reject', 'unknown', 'background', 'shadow', 'edge', 'object')
                """,
                [run_id, *ids, float(config.min_confidence), float(config.min_agreement)],
            ).fetchall()
        )
    return rows


def _cluster(matrix: np.ndarray, *, n_clusters: int, random_state: int) -> np.ndarray:
    if n_clusters <= 1:
        return np.zeros((matrix.shape[0],), dtype=np.int32)
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=max(32, n_clusters * 8), n_init="auto")
    return model.fit_predict(matrix).astype(np.int32, copy=False)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (matrix / norms).astype(np.float32, copy=False)


def _delete_existing(conn: sqlite3.Connection, core_mining_run_id: str) -> None:
    row = conn.execute("SELECT run_id FROM core_mining_runs WHERE core_mining_run_id = ?", (core_mining_run_id,)).fetchone()
    if row is not None:
        run_id = str(row["run_id"])
        cluster_rows = conn.execute("SELECT core_cluster_id FROM core_clusters WHERE core_mining_run_id = ?", (core_mining_run_id,)).fetchall()
        cluster_ids = [str(item["core_cluster_id"]) for item in cluster_rows]
        for cluster_id in cluster_ids:
            conn.execute("DELETE FROM core_cluster_members WHERE run_id = ? AND core_cluster_id = ?", (run_id, cluster_id))
        conn.execute("DELETE FROM core_clusters WHERE core_mining_run_id = ?", (core_mining_run_id,))
    conn.execute("DELETE FROM core_outliers WHERE core_mining_run_id = ?", (core_mining_run_id,))
    conn.execute("DELETE FROM core_mining_runs WHERE core_mining_run_id = ?", (core_mining_run_id,))
    conn.commit()
