from __future__ import annotations

import sqlite3
from typing import Iterable


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS step6_run_info (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            step3_db TEXT NOT NULL,
            step4_db TEXT NOT NULL,
            step5_db TEXT NOT NULL,
            step3_run_id TEXT NOT NULL,
            step4_run_id TEXT NOT NULL,
            step5_run_id TEXT NOT NULL,
            sam_checkpoint TEXT NOT NULL,
            sam_model_type TEXT NOT NULL,
            sam_lora_base TEXT NOT NULL,
            sam_lora_delta TEXT NOT NULL,
            device TEXT NOT NULL,
            input_count INTEGER NOT NULL,
            crack_count INTEGER NOT NULL,
            non_crack_count INTEGER NOT NULL,
            mask_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS segmentation_masks (
            run_id TEXT NOT NULL,
            parent_image_id TEXT NOT NULL,
            det_idx INTEGER NOT NULL,
            clip_label TEXT NOT NULL,
            model_used TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            mask_area_px INTEGER NOT NULL,
            PRIMARY KEY (run_id, parent_image_id, det_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_step6_parent
            ON segmentation_masks (run_id, parent_image_id);
        CREATE INDEX IF NOT EXISTS idx_step6_model
            ON segmentation_masks (run_id, model_used);
        """
    )
    conn.commit()


def insert_run_info(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO step6_run_info VALUES
            (:run_id, :created_at_utc, :step3_db, :step4_db, :step5_db,
             :step3_run_id, :step4_run_id, :step5_run_id,
             :sam_checkpoint, :sam_model_type, :sam_lora_base, :sam_lora_delta,
             :device, :input_count, :crack_count, :non_crack_count, :mask_count)
        """,
        row,
    )
    conn.commit()


def insert_masks(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO segmentation_masks
            (run_id, parent_image_id, det_idx, clip_label, model_used,
             mask_path, x1, y1, x2, y2, mask_area_px)
        VALUES
            (:run_id, :parent_image_id, :det_idx, :clip_label, :model_used,
             :mask_path, :x1, :y1, :x2, :y2, :mask_area_px)
        """,
        rows,
    )


def load_routed_detections(
    step3_conn: sqlite3.Connection,
    step4_conn: sqlite3.Connection,
    step5_conn: sqlite3.Connection,
    *,
    step3_run_id: str | None = "latest",
    step4_run_id: str | None = "latest",
    step5_run_id: str | None = "latest",
) -> tuple[str, str, str, list[dict]]:
    """Return (step3_run, step4_run, step5_run, list of detections)
    where each detection includes clip_label and is filtered by step5 kept=1.
    """
    def _resolve(conn, table):
        cur = conn.execute(
            f"SELECT run_id FROM {table} ORDER BY created_at_utc DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None

    if step3_run_id is None or str(step3_run_id).lower() == "latest":
        step3_run_id = _resolve(step3_conn, "step3_run_info")
    if step4_run_id is None or str(step4_run_id).lower() == "latest":
        step4_run_id = _resolve(step4_conn, "step4_run_info")
    if step5_run_id is None or str(step5_run_id).lower() == "latest":
        step5_run_id = _resolve(step5_conn, "step5_run_info")

    if not (step3_run_id and step4_run_id and step5_run_id):
        return step3_run_id or "", step4_run_id or "", step5_run_id or "", []

    kept = step5_conn.execute(
        """
        SELECT parent_image_id, det_idx
        FROM dedup_decisions
        WHERE run_id = ? AND kept = 1
        """,
        (step5_run_id,),
    ).fetchall()
    kept_set = {(r[0], int(r[1])) for r in kept}

    labels = {}
    for r in step4_conn.execute(
        """
        SELECT parent_image_id, det_idx, predicted_label, predicted_probability
        FROM semantic_labels
        WHERE run_id = ?
        """,
        (step4_run_id,),
    ).fetchall():
        labels[(r[0], int(r[1]))] = (r[2], float(r[3]))

    detections: list[dict] = []
    for r in step3_conn.execute(
        """
        SELECT parent_image_id, det_idx, group_name,
               x1, y1, x2, y2, score
        FROM damage_detections
        WHERE run_id = ?
        ORDER BY parent_image_id, det_idx
        """,
        (step3_run_id,),
    ).fetchall():
        key = (r[0], int(r[1]))
        if key not in kept_set or key not in labels:
            continue
        clip_label, clip_prob = labels[key]
        detections.append(
            {
                "parent_image_id": r[0],
                "det_idx": int(r[1]),
                "gdino_group": r[2],
                "box": [float(r[3]), float(r[4]), float(r[5]), float(r[6])],
                "gdino_score": float(r[7]),
                "clip_label": clip_label,
                "clip_prob": clip_prob,
            }
        )
    return step3_run_id, step4_run_id, step5_run_id, detections
