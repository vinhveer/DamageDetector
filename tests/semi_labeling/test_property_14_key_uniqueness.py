# Feature: semi-labeling-pipeline-improvements, Property 14: Embedding key uniqueness
# For all stored embeddings, the tuple (embedding_run_id, result_id, view_name) is unique.
from __future__ import annotations

import os
import sqlite3
import tempfile

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_os = sl_imports.load_step("step3_embedding", "output_store")

_rows = st.lists(
    st.tuples(st.integers(min_value=1, max_value=10), st.sampled_from(["tight", "context", "wide"])),
    min_size=1, max_size=30,
)


@given(keys=_rows)
def test_embedding_key_uniqueness(keys):
    blob = _os.encode_vector(np.zeros(4, dtype=np.float32))
    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(os.path.join(d, "e.db"))
        conn.row_factory = sqlite3.Row
        _os.ensure_view_schema(conn)
        rows = [("run", rid, "i.png", "crack", view, blob) for rid, view in keys]
        _os.bulk_insert_view_embeddings(conn, rows)

        distinct = {(rid, view) for rid, view in keys}
        total = int(conn.execute("SELECT COUNT(*) FROM detection_embeddings").fetchone()[0])
        assert total == len(distinct)
        assert _os.embedded_view_keys(conn, embedding_run_id="run") == distinct
        conn.close()
