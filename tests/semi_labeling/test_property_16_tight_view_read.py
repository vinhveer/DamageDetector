# Feature: semi-labeling-pipeline-improvements, Property 16: Tight-view read selection
# load_tight_embeddings returns the tight vector for exactly the detections with a
# persisted tight view, and reports every other requested detection as unavailable.
from __future__ import annotations

import os
import sqlite3
import tempfile

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_os = sl_imports.load_step("step3_embedding", "output_store")
_DIM = 4

# Per result_id: which views are persisted (subset of tight/context).
_presence = st.dictionaries(
    keys=st.integers(1, 15),
    values=st.lists(st.sampled_from(["tight", "context"]), min_size=0, max_size=2, unique=True),
    min_size=1, max_size=10,
)


@given(presence=_presence, extra_requested=st.lists(st.integers(16, 25), max_size=5))
def test_tight_view_read_selection(presence, extra_requested):
    blob = _os.encode_vector(np.arange(_DIM, dtype=np.float32))
    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(os.path.join(d, "e.db"))
        conn.row_factory = sqlite3.Row
        _os.ensure_view_schema(conn)
        rows = [("run", rid, "i.png", "crack", view, blob) for rid, views in presence.items() for view in views]
        _os.bulk_insert_view_embeddings(conn, rows)

        requested = list(presence.keys()) + list(extra_requested)
        out, missing = _os.load_tight_embeddings(conn, embedding_run_id="run", dim=_DIM, result_ids=requested)

        have_tight = {rid for rid, views in presence.items() if "tight" in views}
        assert set(out.keys()) == have_tight
        assert missing == {rid for rid in requested if rid not in have_tight}
        conn.close()
