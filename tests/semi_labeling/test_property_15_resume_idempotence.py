# Feature: semi-labeling-pipeline-improvements, Property 15: Resume idempotence
# For all interrupted runs, resuming after an interruption point and completing produces
# the same set of (result_id, view_name) embeddings and skip records as one uninterrupted run.
from __future__ import annotations

import os
import sqlite3
import tempfile

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_os = sl_imports.load_step("step3_embedding", "output_store")

# Each target: (result_id, view_name, will_embed). Distinct keys, deterministic outcome.
_targets = st.lists(
    st.tuples(st.integers(1, 12), st.sampled_from(["tight", "context"]), st.booleans()),
    min_size=1, max_size=20,
)


def _dedup(targets):
    seen, out = set(), []
    for rid, view, embed in targets:
        if (rid, view) in seen:
            continue
        seen.add((rid, view))
        out.append((rid, view, embed))
    return out


def _process(conn, items):
    blob = _os.encode_vector(np.zeros(4, dtype=np.float32))
    emb_rows = [("run", rid, "i.png", "crack", view, blob) for rid, view, e in items if e]
    skip_rows = [("run", rid, view, "invalid_bbox") for rid, view, e in items if not e]
    _os.bulk_insert_view_embeddings(conn, emb_rows)
    _os.bulk_insert_view_skipped(conn, skip_rows)


@given(targets=_targets, cut=st.integers(0, 20))
def test_resume_idempotence(targets, cut):
    items = _dedup(targets)
    expected_emb = {(rid, view) for rid, view, e in items if e}
    expected_skip = {(rid, view) for rid, view, e in items if not e}

    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(os.path.join(d, "e.db"))
        conn.row_factory = sqlite3.Row
        _os.ensure_view_schema(conn)

        # Interrupt after `cut` items, then resume on the not-yet-done keys.
        first, rest = items[:cut], items[cut:]
        _process(conn, first)
        done = _os.embedded_view_keys(conn, embedding_run_id="run") | _os.skipped_view_keys(
            conn, embedding_run_id="run"
        )
        remaining = [it for it in rest if (it[0], it[1]) not in done]
        _process(conn, remaining)

        assert _os.embedded_view_keys(conn, embedding_run_id="run") == expected_emb
        assert _os.skipped_view_keys(conn, embedding_run_id="run") == expected_skip
        conn.close()
