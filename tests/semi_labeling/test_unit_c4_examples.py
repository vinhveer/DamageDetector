# C4 example/edge unit tests (Requirements 4.5, 4.7).
from __future__ import annotations

import os
import sqlite3
import tempfile
from types import SimpleNamespace

import numpy as np

import sl_imports

_os = sl_imports.load_step("step3_embedding", "output_store")
_ed = sl_imports.load_step("step3_embedding", "embed_detections")

# Legacy detection_embeddings schema (pre-C4, PK without view_name).
_LEGACY_DDL = """
CREATE TABLE detection_embeddings (
    embedding_run_id TEXT NOT NULL, result_id INTEGER NOT NULL,
    image_rel_path TEXT NOT NULL, predicted_label TEXT NOT NULL,
    embedding_blob BLOB NOT NULL, PRIMARY KEY (embedding_run_id, result_id)
);
CREATE TABLE skipped_detections (
    embedding_run_id TEXT NOT NULL, result_id INTEGER NOT NULL, reason TEXT NOT NULL,
    PRIMARY KEY (embedding_run_id, result_id)
);
"""


def test_pre_feature_table_defaults_to_tight():  # Req 4.7
    blob = _os.encode_vector(np.arange(4, dtype=np.float32))
    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(os.path.join(d, "e.db"))
        conn.row_factory = sqlite3.Row
        conn.executescript(_LEGACY_DDL)
        conn.execute(
            "INSERT INTO detection_embeddings VALUES (?, ?, ?, ?, ?)",
            ("run", 7, "i.png", "crack", blob),
        )
        conn.commit()

        _os.ensure_view_schema(conn)  # migrate; idempotent
        _os.ensure_view_schema(conn)
        assert _os.embedded_view_keys(conn, embedding_run_id="run") == {(7, "tight")}
        out, missing = _os.load_tight_embeddings(conn, embedding_run_id="run", dim=4, result_ids=[7, 8])
        assert set(out.keys()) == {7} and missing == {8}
        conn.close()


class _MockEmbedder:
    dim = 8

    def embed(self, crops, *, batch_size):
        return np.zeros((len(crops), self.dim), dtype=np.float32)


def test_invalid_crop_skips_only_affected_view():  # Req 4.5
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        Image.new("RGB", (100, 100), (10, 20, 30)).save(os.path.join(d, "img.png"))
        # Zero-area box -> both views invalid.
        det = SimpleNamespace(
            result_id=3, image_rel_path="img.png", image_path="img.png", source_input_dir=d,
            predicted_label="crack", x1=50.0, y1=50.0, x2=50.0, y2=50.0, image_width=100, image_height=100,
        )
        vectors, skips = _ed.embed_detection_multiview(det, None, _MockEmbedder())
        assert vectors == {}
        assert {s.view_name for s in skips} == {"tight", "context"}
        assert all(s.result_id == 3 and s.reason for s in skips)

    # padded_crop_box returns None for an out-of-bounds / degenerate crop.
    assert _ed.padded_crop_box(det, 100, 100, 0.0) is None
