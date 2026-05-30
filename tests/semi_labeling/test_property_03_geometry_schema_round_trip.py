# Feature: semi-labeling-pipeline-improvements, Property 3: Geometry schema round-trip and backward compatibility
# Written geometry preserves all pre-existing column values and returns the geometry
# values; a pre-feature table (no geometry columns) reads through the new path with
# defaults applied (0 for count columns, 0.0 for max_iou_same_label / max_containment).
from __future__ import annotations

import os
import sqlite3
import tempfile

from hypothesis import given
from hypothesis import strategies as st

import object_detection.damage_scan.geometry as geo
import object_detection.damage_scan.sqlite_store as store

# Minimal "pre-feature" detections table (subset of the canonical schema, no geometry cols).
_BASE_DDL = """
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    image_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL
)
"""

_coord = st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False)


@given(
    box=st.tuples(_coord, _coord, _coord, _coord),
    label=st.sampled_from(["crack", "mold", "spall"]),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_geometry_schema_round_trip_and_backward_compat(box, label, score):
    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(os.path.join(d, "t.db"))
        conn.row_factory = sqlite3.Row
        conn.execute(_BASE_DDL)
        conn.execute(
            "INSERT INTO detections (run_id, image_id, label, score, x1, y1, x2, y2)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("run", 1, label, score, *box),
        )
        conn.commit()
        det_id = int(conn.execute("SELECT detection_id FROM detections").fetchone()["detection_id"])

        # Pre-feature read path applies defaults (Requirement 1.11).
        defaults = store.read_detection_geometry(conn, det_id)
        assert defaults["contains_count"] == 0
        assert defaults["contained_by_count"] == 0
        assert defaults["max_iou_same_label"] == 0.0
        assert defaults["max_containment"] == 0.0

        # Idempotent migration.
        store.ensure_geometry_columns(conn)
        store.ensure_geometry_columns(conn)

        (bg,) = geo.compute_box_geometry(
            [geo.GeoInput(det_id, box[0], box[1], box[2], box[3], label)], 256, 256
        )
        store.update_geometry(conn, [bg])

        got = store.read_detection_geometry(conn, det_id)
        for name in store._GEOMETRY_COLUMNS:
            assert got[name] == getattr(bg, name)

        # Pre-existing columns preserved (Requirement 1.10).
        row = conn.execute(
            "SELECT label, score, x1, y1, x2, y2 FROM detections WHERE detection_id = ?",
            (det_id,),
        ).fetchone()
        assert row["label"] == label
        assert row["score"] == score
        assert (row["x1"], row["y1"], row["x2"], row["y2"]) == box
        conn.close()
