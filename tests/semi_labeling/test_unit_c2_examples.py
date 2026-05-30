# C2B example/edge unit tests (Requirement 2.4 schema + deterministic label fallback).
from __future__ import annotations

import os
import sqlite3
import tempfile

import object_detection.damage_scan.sqlite_store as store

_BASE_DDL = """
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, image_id INTEGER, prompt_key TEXT, label TEXT, score REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL
)
"""


def test_gdino_columns_and_label_fallback():  # Req 2.4
    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(os.path.join(d, "t.db"))
        conn.row_factory = sqlite3.Row
        conn.execute(_BASE_DDL)
        conn.execute(
            "INSERT INTO detections (run_id, image_id, prompt_key, label, score, x1, y1, x2, y2)"
            " VALUES ('r', 1, 'crack_prompt', 'crack', 0.8, 0, 0, 10, 10)"
        )
        conn.commit()

        store.ensure_gdino_columns(conn)
        store.ensure_gdino_columns(conn)  # idempotent
        cols = store._existing_columns(conn, "detections")
        assert {"gdino_label", "gdino_score", "gdino_prompt_key", "predicted_label"} <= cols

        # Forward backfill populates gdino_* from legacy columns (idempotent).
        store.backfill_gdino_columns(conn)
        store.backfill_gdino_columns(conn)
        row = conn.execute("SELECT * FROM detections WHERE detection_id = 1").fetchone()
        assert row["gdino_label"] == "crack"
        assert row["gdino_score"] == 0.8
        assert row["gdino_prompt_key"] == "crack_prompt"

        # Deterministic fallback: predicted_label is NULL -> falls back to gdino_label.
        assert store.read_predicted_label(row) == "crack"
        conn.execute("UPDATE detections SET predicted_label = 'mold' WHERE detection_id = 1")
        conn.commit()
        row2 = conn.execute("SELECT * FROM detections WHERE detection_id = 1").fetchone()
        assert store.read_predicted_label(row2) == "mold"  # committed label wins
        conn.close()


def test_pipeline_order_default_is_label_first():
    assert store.PipelineOrder.LABEL_FIRST.value == "label_first"
    assert store.PipelineOrder.DEDUP_FIRST.value == "dedup_first"
