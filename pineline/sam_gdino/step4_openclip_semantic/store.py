from __future__ import annotations

import sqlite3
from typing import Iterable


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS step4_run_info (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            step3_db TEXT NOT NULL,
            step3_run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            pretrained TEXT NOT NULL,
            device TEXT NOT NULL,
            batch_size INTEGER NOT NULL,
            detection_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS semantic_labels (
            run_id TEXT NOT NULL,
            parent_image_id TEXT NOT NULL,
            det_idx INTEGER NOT NULL,
            predicted_label TEXT NOT NULL,
            predicted_probability REAL NOT NULL,
            class_scores_json TEXT NOT NULL,
            PRIMARY KEY (run_id, parent_image_id, det_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_semantic_parent
            ON semantic_labels (run_id, parent_image_id);
        CREATE INDEX IF NOT EXISTS idx_semantic_label
            ON semantic_labels (run_id, predicted_label);
        """
    )
    conn.commit()


def insert_run_info(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO step4_run_info
            (run_id, created_at_utc, step3_db, step3_run_id,
             model_name, pretrained, device, batch_size, detection_count)
        VALUES
            (:run_id, :created_at_utc, :step3_db, :step3_run_id,
             :model_name, :pretrained, :device, :batch_size, :detection_count)
        """,
        row,
    )
    conn.commit()


def insert_labels(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO semantic_labels
            (run_id, parent_image_id, det_idx,
             predicted_label, predicted_probability, class_scores_json)
        VALUES
            (:run_id, :parent_image_id, :det_idx,
             :predicted_label, :predicted_probability, :class_scores_json)
        """,
        rows,
    )


def load_step3_detections(
    step3_conn: sqlite3.Connection,
    *,
    source_run_id: str | None = None,
) -> tuple[str, list[dict]]:
    if source_run_id is None or str(source_run_id).strip().lower() == "latest":
        cur = step3_conn.execute(
            "SELECT run_id FROM step3_run_info ORDER BY created_at_utc DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return "", []
        source_run_id = row[0]
    cur = step3_conn.execute(
        """
        SELECT parent_image_id, det_idx, group_name, label,
               x1, y1, x2, y2, score
        FROM damage_detections
        WHERE run_id = ?
        ORDER BY parent_image_id, det_idx
        """,
        (source_run_id,),
    )
    rows = [
        {
            "parent_image_id": r[0],
            "det_idx": int(r[1]),
            "group_name": r[2],
            "label": r[3],
            "box": [float(r[4]), float(r[5]), float(r[6]), float(r[7])],
            "score": float(r[8]),
        }
        for r in cur.fetchall()
    ]
    return source_run_id, rows
