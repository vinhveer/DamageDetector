from __future__ import annotations

import sqlite3
from typing import Iterable


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS step5_run_info (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            step3_db TEXT NOT NULL,
            step4_db TEXT NOT NULL,
            step3_run_id TEXT NOT NULL,
            step4_run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            device TEXT NOT NULL,
            iou_threshold REAL NOT NULL,
            cosine_threshold REAL NOT NULL,
            input_count INTEGER NOT NULL,
            kept_count INTEGER NOT NULL,
            merged_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dedup_decisions (
            run_id TEXT NOT NULL,
            parent_image_id TEXT NOT NULL,
            det_idx INTEGER NOT NULL,
            kept INTEGER NOT NULL,
            merged_into_det_idx INTEGER,
            reason TEXT NOT NULL,
            PRIMARY KEY (run_id, parent_image_id, det_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_dedup_parent
            ON dedup_decisions (run_id, parent_image_id);
        """
    )
    conn.commit()


def insert_run_info(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO step5_run_info
            (run_id, created_at_utc, step3_db, step4_db,
             step3_run_id, step4_run_id, model_name, device,
             iou_threshold, cosine_threshold,
             input_count, kept_count, merged_count)
        VALUES
            (:run_id, :created_at_utc, :step3_db, :step4_db,
             :step3_run_id, :step4_run_id, :model_name, :device,
             :iou_threshold, :cosine_threshold,
             :input_count, :kept_count, :merged_count)
        """,
        row,
    )
    conn.commit()


def insert_decisions(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO dedup_decisions
            (run_id, parent_image_id, det_idx, kept, merged_into_det_idx, reason)
        VALUES
            (:run_id, :parent_image_id, :det_idx, :kept, :merged_into_det_idx, :reason)
        """,
        rows,
    )


def load_step3_with_step4(
    step3_conn: sqlite3.Connection,
    step4_conn: sqlite3.Connection,
    *,
    step3_run_id: str | None = "latest",
    step4_run_id: str | None = "latest",
) -> tuple[str, str, list[dict]]:
    if step3_run_id is None or str(step3_run_id).lower() == "latest":
        cur = step3_conn.execute(
            "SELECT run_id FROM step3_run_info ORDER BY created_at_utc DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return "", "", []
        step3_run_id = row[0]
    if step4_run_id is None or str(step4_run_id).lower() == "latest":
        cur = step4_conn.execute(
            "SELECT run_id FROM step4_run_info ORDER BY created_at_utc DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return "", "", []
        step4_run_id = row[0]

    cur = step3_conn.execute(
        """
        SELECT parent_image_id, det_idx, group_name,
               x1, y1, x2, y2, score
        FROM damage_detections
        WHERE run_id = ?
        ORDER BY parent_image_id, det_idx
        """,
        (step3_run_id,),
    )
    dets = {
        (r[0], int(r[1])): {
            "parent_image_id": r[0],
            "det_idx": int(r[1]),
            "gdino_group": r[2],
            "box": [float(r[3]), float(r[4]), float(r[5]), float(r[6])],
            "gdino_score": float(r[7]),
        }
        for r in cur.fetchall()
    }

    cur = step4_conn.execute(
        """
        SELECT parent_image_id, det_idx, predicted_label, predicted_probability
        FROM semantic_labels
        WHERE run_id = ?
        """,
        (step4_run_id,),
    )
    rows: list[dict] = []
    for r in cur.fetchall():
        key = (r[0], int(r[1]))
        det = dets.get(key)
        if det is None:
            continue
        det["clip_label"] = r[2]
        det["clip_prob"] = float(r[3])
        rows.append(det)
    rows.sort(key=lambda d: (d["parent_image_id"], d["det_idx"]))
    return step3_run_id, step4_run_id, rows
