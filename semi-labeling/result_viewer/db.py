from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from models import AssignmentRow, ClusterSummary, GroupRun, SourceMeta


def connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.expanduser().resolve()}?mode=ro", uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def connect_rw(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.expanduser().resolve(), timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


class FeatureGroupStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().resolve()

    def list_runs(self) -> list[GroupRun]:
        conn = connect_ro(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT grouping_run_id, created_at_utc, source_db_path, filtered_db_path,
                       source_filter_run_id, model_name, device, total_boxes, total_clusters,
                       outlier_boxes, label_suspect_boxes
                FROM feature_group_runs
                ORDER BY created_at_utc DESC
                """
            ).fetchall()
        finally:
            conn.close()
        return [GroupRun(**dict(row)) for row in rows]

    def list_clusters(self, run_id: str, label_scope: str, mode: str) -> list[ClusterSummary]:
        clauses = ["grouping_run_id = ?", "predicted_label_scope = ?"]
        params: list[Any] = [run_id, label_scope]
        if mode == "non_outlier":
            clauses.append("outlier_count = 0")
        elif mode == "outlier":
            clauses.append("outlier_count > 0")
        elif mode == "label_suspect":
            clauses.append(
                """
                EXISTS (
                    SELECT 1
                    FROM feature_group_assignments a
                    WHERE a.grouping_run_id = feature_group_clusters.grouping_run_id
                      AND a.cluster_key = feature_group_clusters.cluster_key
                      AND a.label_suspect != 0
                )
                """
            )
        conn = connect_ro(self.db_path)
        try:
            rows = conn.execute(
                f"""
                SELECT cluster_key, predicted_label_scope, cluster_id, cluster_size,
                       major_label, purity, crack_count, mold_count, spall_count,
                       outlier_count, representative_nearest_result_id,
                       representative_farthest_result_id,
                       representative_low_confidence_result_id,
                       representative_mismatch_result_id
                FROM feature_group_clusters
                WHERE {' AND '.join(clauses)}
                ORDER BY cluster_size DESC, purity ASC, cluster_key
                """,
                params,
            ).fetchall()
        finally:
            conn.close()
        return [ClusterSummary(**dict(row)) for row in rows]

    def list_assignments(self, run_id: str, cluster_key: str) -> list[AssignmentRow]:
        conn = connect_ro(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT result_id, image_rel_path, predicted_label, predicted_probability_pct,
                       detector_score, cluster_key, is_outlier, distance_to_center,
                       suggested_label, label_suspect, cluster_purity, cluster_size
                FROM feature_group_assignments
                WHERE grouping_run_id = ? AND cluster_key = ?
                ORDER BY distance_to_center ASC, predicted_probability_pct ASC, result_id
                """,
                (run_id, cluster_key),
            ).fetchall()
        finally:
            conn.close()
        return [AssignmentRow(**dict(row)) for row in rows]

    def clear_flags_for_results(self, run_id: str, result_ids: list[int]) -> int:
        if not result_ids:
            return 0
        placeholders = ",".join("?" for _ in result_ids)
        conn = connect_rw(self.db_path)
        try:
            before = conn.execute(
                f"""
                SELECT cluster_key, SUM(is_outlier) AS outliers
                FROM feature_group_assignments
                WHERE grouping_run_id = ? AND result_id IN ({placeholders})
                GROUP BY cluster_key
                """,
                [run_id, *result_ids],
            ).fetchall()
            changed = conn.execute(
                f"""
                UPDATE feature_group_assignments
                SET is_outlier = 0, label_suspect = 0
                WHERE grouping_run_id = ?
                  AND result_id IN ({placeholders})
                  AND (is_outlier != 0 OR label_suspect != 0)
                """,
                [run_id, *result_ids],
            ).rowcount
            for row in before:
                outliers = int(row["outliers"] or 0)
                if outliers > 0:
                    conn.execute(
                        """
                        UPDATE feature_group_clusters
                        SET outlier_count = MAX(0, outlier_count - ?)
                        WHERE grouping_run_id = ? AND cluster_key = ?
                        """,
                        (outliers, run_id, row["cluster_key"]),
                    )
            self._refresh_run_flag_counts(conn, run_id)
            conn.commit()
        finally:
            conn.close()
        return int(changed)

    def clear_flags_for_cluster(self, run_id: str, cluster_key: str) -> int:
        conn = connect_rw(self.db_path)
        try:
            changed = conn.execute(
                """
                UPDATE feature_group_assignments
                SET is_outlier = 0, label_suspect = 0
                WHERE grouping_run_id = ? AND cluster_key = ?
                  AND (is_outlier != 0 OR label_suspect != 0)
                """,
                (run_id, cluster_key),
            ).rowcount
            conn.execute(
                """
                UPDATE feature_group_clusters
                SET outlier_count = 0
                WHERE grouping_run_id = ? AND cluster_key = ?
                """,
                (run_id, cluster_key),
            )
            self._refresh_run_flag_counts(conn, run_id)
            conn.commit()
        finally:
            conn.close()
        return int(changed)

    def _refresh_run_flag_counts(self, conn: sqlite3.Connection, run_id: str) -> None:
        totals = conn.execute(
            """
            SELECT SUM(is_outlier) AS outliers, SUM(label_suspect) AS suspects
            FROM feature_group_assignments
            WHERE grouping_run_id = ?
            """,
            (run_id,),
        ).fetchone()
        conn.execute(
            """
            UPDATE feature_group_runs
            SET outlier_boxes = ?, label_suspect_boxes = ?
            WHERE grouping_run_id = ?
            """,
            (int(totals["outliers"] or 0), int(totals["suspects"] or 0), run_id),
        )


class SourceStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().resolve()

    def source_meta(self, result_ids: list[int]) -> dict[int, SourceMeta]:
        if not result_ids:
            return {}
        placeholders = ",".join("?" for _ in result_ids)
        conn = connect_ro(self.db_path)
        try:
            rows = conn.execute(
                f"""
                SELECT res.result_id, res.image_path, src_run.input_dir AS source_input_dir,
                       res.x1, res.y1, res.x2, res.y2
                FROM openclip_semantic_results res
                JOIN runs src_run ON src_run.run_id = res.source_run_id
                WHERE res.result_id IN ({placeholders})
                """,
                result_ids,
            ).fetchall()
        finally:
            conn.close()
        return {int(row["result_id"]): SourceMeta(**dict(row)) for row in rows}


def merge_source_meta(assignments: list[AssignmentRow], meta_by_id: dict[int, SourceMeta]) -> list[AssignmentRow]:
    out: list[AssignmentRow] = []
    for row in assignments:
        meta = meta_by_id.get(int(row.result_id))
        if meta is None:
            out.append(row)
            continue
        out.append(
            AssignmentRow(
                result_id=row.result_id,
                image_rel_path=row.image_rel_path,
                predicted_label=row.predicted_label,
                predicted_probability_pct=row.predicted_probability_pct,
                detector_score=row.detector_score,
                cluster_key=row.cluster_key,
                is_outlier=row.is_outlier,
                distance_to_center=row.distance_to_center,
                suggested_label=row.suggested_label,
                label_suspect=row.label_suspect,
                cluster_purity=row.cluster_purity,
                cluster_size=row.cluster_size,
                image_path=meta.image_path,
                source_input_dir=meta.source_input_dir,
                x1=meta.x1,
                y1=meta.y1,
                x2=meta.x2,
                y2=meta.y2,
            )
        )
    return out
