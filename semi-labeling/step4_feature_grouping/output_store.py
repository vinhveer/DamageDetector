from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import ClusterAssignment, ClusterSummary, FeatureGroupConfig


def connect_output(db_path: Path) -> sqlite3.Connection:
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
        CREATE TABLE IF NOT EXISTS feature_group_runs (
            grouping_run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            source_db_path TEXT NOT NULL,
            filtered_db_path TEXT NOT NULL,
            source_filter_run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            device TEXT NOT NULL,
            options_json TEXT NOT NULL,
            total_boxes INTEGER NOT NULL,
            total_clusters INTEGER NOT NULL,
            outlier_boxes INTEGER NOT NULL,
            label_suspect_boxes INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feature_group_clusters (
            grouping_run_id TEXT NOT NULL,
            cluster_key TEXT NOT NULL,
            predicted_label_scope TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            cluster_size INTEGER NOT NULL,
            major_label TEXT NOT NULL,
            purity REAL NOT NULL,
            crack_count INTEGER NOT NULL,
            mold_count INTEGER NOT NULL,
            spall_count INTEGER NOT NULL,
            outlier_count INTEGER NOT NULL,
            representative_nearest_result_id INTEGER,
            representative_farthest_result_id INTEGER,
            representative_low_confidence_result_id INTEGER,
            representative_mismatch_result_id INTEGER,
            PRIMARY KEY (grouping_run_id, cluster_key)
        );
        CREATE TABLE IF NOT EXISTS feature_group_assignments (
            grouping_run_id TEXT NOT NULL,
            result_id INTEGER NOT NULL,
            source_detection_id INTEGER NOT NULL,
            image_rel_path TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            predicted_probability_pct REAL NOT NULL,
            detector_score REAL NOT NULL,
            label_scope TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            cluster_key TEXT NOT NULL,
            is_outlier INTEGER NOT NULL,
            distance_to_center REAL NOT NULL,
            suggested_label TEXT NOT NULL,
            label_suspect INTEGER NOT NULL,
            cluster_purity REAL NOT NULL,
            cluster_size INTEGER NOT NULL,
            PRIMARY KEY (grouping_run_id, result_id)
        );
        CREATE INDEX IF NOT EXISTS idx_feature_group_clusters_run ON feature_group_clusters (grouping_run_id, predicted_label_scope, cluster_size DESC);
        CREATE INDEX IF NOT EXISTS idx_feature_group_assignments_cluster ON feature_group_assignments (grouping_run_id, cluster_key, distance_to_center);
        CREATE INDEX IF NOT EXISTS idx_feature_group_assignments_image ON feature_group_assignments (grouping_run_id, image_rel_path);
        """
    )
    conn.commit()


def write_feature_groups(
    db_path: Path,
    *,
    grouping_run_id: str,
    config: FeatureGroupConfig,
    device: str,
    assignments: list[ClusterAssignment],
    summaries: list[ClusterSummary],
) -> None:
    conn = connect_output(db_path)
    try:
        ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO feature_group_runs (
                grouping_run_id, created_at_utc, source_db_path, filtered_db_path,
                source_filter_run_id, model_name, device, options_json, total_boxes,
                total_clusters, outlier_boxes, label_suspect_boxes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                grouping_run_id,
                datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                str(config.source_db_path),
                str(config.filtered_db_path),
                config.filter_run_id,
                config.model_name,
                device,
                json.dumps(config.to_json_dict(), ensure_ascii=False, sort_keys=True),
                len(assignments),
                len(summaries),
                sum(1 for item in assignments if item.is_outlier),
                sum(1 for item in assignments if item.label_suspect),
            ),
        )
        conn.executemany(
            """
            INSERT INTO feature_group_clusters (
                grouping_run_id, cluster_key, predicted_label_scope, cluster_id,
                cluster_size, major_label, purity, crack_count, mold_count, spall_count,
                outlier_count, representative_nearest_result_id,
                representative_farthest_result_id, representative_low_confidence_result_id,
                representative_mismatch_result_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    grouping_run_id,
                    item.cluster_key,
                    item.predicted_label_scope,
                    item.cluster_id,
                    item.cluster_size,
                    item.major_label,
                    item.purity,
                    item.crack_count,
                    item.mold_count,
                    item.spall_count,
                    item.outlier_count,
                    item.representative_nearest_result_id,
                    item.representative_farthest_result_id,
                    item.representative_low_confidence_result_id,
                    item.representative_mismatch_result_id,
                )
                for item in summaries
            ],
        )
        conn.executemany(
            """
            INSERT INTO feature_group_assignments (
                grouping_run_id, result_id, source_detection_id, image_rel_path,
                predicted_label, predicted_probability_pct, detector_score, label_scope,
                cluster_id, cluster_key, is_outlier, distance_to_center, suggested_label,
                label_suspect, cluster_purity, cluster_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    grouping_run_id,
                    item.result_id,
                    item.source_detection_id,
                    item.image_rel_path,
                    item.predicted_label,
                    item.predicted_probability_pct,
                    item.detector_score,
                    item.label_scope,
                    item.cluster_id,
                    item.cluster_key,
                    1 if item.is_outlier else 0,
                    item.distance_to_center,
                    item.suggested_label,
                    1 if item.label_suspect else 0,
                    item.cluster_purity,
                    item.cluster_size,
                )
                for item in assignments
            ],
        )
        conn.commit()
    finally:
        conn.close()
