from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ClusterAssignment:
    result_id: int
    image_rel_path: str
    predicted_label: str
    cluster_id: int
    distance_to_centroid: float
    rank_in_cluster: int
    is_representative: bool


def connect_output(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cluster_runs (
            cluster_run_id       TEXT PRIMARY KEY,
            created_at_utc       TEXT NOT NULL,
            dedup_db_path        TEXT NOT NULL,
            dedup_run_id         TEXT NOT NULL,
            embedding_db_path    TEXT NOT NULL,
            embedding_run_id     TEXT NOT NULL,
            algorithm            TEXT NOT NULL,
            options_json         TEXT NOT NULL,
            total_boxes          INTEGER NOT NULL,
            total_clusters       INTEGER NOT NULL,
            pca_dim              INTEGER NOT NULL,
            pca_explained_ratio  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cluster_results (
            cluster_run_id          TEXT NOT NULL,
            result_id               INTEGER NOT NULL,
            image_rel_path          TEXT NOT NULL,
            predicted_label         TEXT NOT NULL,
            cluster_id              INTEGER NOT NULL,
            distance_to_centroid    REAL NOT NULL,
            is_representative       INTEGER NOT NULL,
            rank_in_cluster         INTEGER NOT NULL,
            PRIMARY KEY (cluster_run_id, result_id)
        );

        CREATE INDEX IF NOT EXISTS idx_cluster_id
        ON cluster_results (cluster_run_id, cluster_id, rank_in_cluster);

        CREATE INDEX IF NOT EXISTS idx_cluster_image
        ON cluster_results (cluster_run_id, image_rel_path);

        CREATE TABLE IF NOT EXISTS cluster_summary (
            cluster_run_id            TEXT NOT NULL,
            cluster_id                INTEGER NOT NULL,
            size                      INTEGER NOT NULL,
            representative_result_id  INTEGER NOT NULL,
            label_distribution_json   TEXT NOT NULL,
            dominant_label            TEXT NOT NULL,
            avg_intra_distance        REAL NOT NULL,
            centroid_blob             BLOB,
            PRIMARY KEY (cluster_run_id, cluster_id)
        );
        """
    )
    conn.commit()


def insert_run_metadata(
    conn: sqlite3.Connection,
    *,
    cluster_run_id: str,
    dedup_db_path: Path,
    dedup_run_id: str,
    embedding_db_path: Path,
    embedding_run_id: str,
    algorithm: str,
    options: dict[str, Any],
    total_boxes: int,
    total_clusters: int,
    pca_dim: int,
    pca_explained_ratio: float,
) -> None:
    conn.execute(
        """
        INSERT INTO cluster_runs (
            cluster_run_id, created_at_utc, dedup_db_path, dedup_run_id,
            embedding_db_path, embedding_run_id, algorithm, options_json,
            total_boxes, total_clusters, pca_dim, pca_explained_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cluster_run_id,
            utc_now(),
            str(Path(dedup_db_path).expanduser().resolve()),
            str(dedup_run_id),
            str(Path(embedding_db_path).expanduser().resolve()),
            str(embedding_run_id),
            str(algorithm),
            json.dumps(options, ensure_ascii=False, sort_keys=True),
            int(total_boxes),
            int(total_clusters),
            int(pca_dim),
            float(pca_explained_ratio),
        ),
    )
    conn.commit()


def write_assignments(
    conn: sqlite3.Connection,
    *,
    cluster_run_id: str,
    assignments: Iterable[ClusterAssignment],
) -> int:
    rows = [
        (
            cluster_run_id,
            int(item.result_id),
            item.image_rel_path,
            item.predicted_label,
            int(item.cluster_id),
            float(item.distance_to_centroid),
            1 if item.is_representative else 0,
            int(item.rank_in_cluster),
        )
        for item in assignments
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO cluster_results (
            cluster_run_id, result_id, image_rel_path, predicted_label,
            cluster_id, distance_to_centroid, is_representative, rank_in_cluster
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def write_summary(
    conn: sqlite3.Connection,
    *,
    cluster_run_id: str,
    assignments: list[ClusterAssignment],
    centroids: np.ndarray,
) -> int:
    by_cluster: dict[int, list[ClusterAssignment]] = {}
    for item in assignments:
        by_cluster.setdefault(int(item.cluster_id), []).append(item)

    rows = []
    for cluster_id, members in by_cluster.items():
        members_sorted = sorted(members, key=lambda m: int(m.rank_in_cluster))
        rep_id = int(members_sorted[0].result_id) if members_sorted else -1
        label_counter = Counter(m.predicted_label for m in members)
        dominant = label_counter.most_common(1)[0][0] if label_counter else ""
        distances = [float(m.distance_to_centroid) for m in members]
        avg_distance = float(sum(distances) / max(1, len(distances)))
        centroid_blob = None
        if 0 <= cluster_id < int(centroids.shape[0]):
            centroid_vec = np.asarray(centroids[cluster_id], dtype="<f4")
            centroid_blob = sqlite3.Binary(centroid_vec.tobytes())
        rows.append(
            (
                cluster_run_id,
                int(cluster_id),
                len(members),
                rep_id,
                json.dumps(dict(label_counter), ensure_ascii=False, sort_keys=True),
                dominant,
                avg_distance,
                centroid_blob,
            )
        )
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO cluster_summary (
            cluster_run_id, cluster_id, size, representative_result_id,
            label_distribution_json, dominant_label, avg_intra_distance, centroid_blob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)
