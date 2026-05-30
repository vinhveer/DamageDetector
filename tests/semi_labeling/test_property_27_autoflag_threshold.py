# Feature: semi-labeling-pipeline-improvements, Property 27: Auto-flag threshold
# A 'similar_to_known_error:<original_label>' flag is set iff the max cosine similarity to a
# stored correction embedding >= 0.92.
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import sl_imports

_af = sl_imports.load_step_file("step7_label_review", "corrections/autoflag.py", "sl_autoflag")
_DIM = 5
_THRESH = 0.92
_elem = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False, width=32)


def _matrix(n):
    return hnp.arrays(np.float32, (n, _DIM), elements=_elem)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@given(
    new=st.integers(1, 5).flatmap(_matrix),
    corr=st.integers(1, 4).flatmap(_matrix),
    labels=st.lists(st.sampled_from(["crack", "mold", "spall"]), min_size=4, max_size=4),
)
def test_autoflag_threshold(new, corr, labels):
    corr_labels = labels[: corr.shape[0]]
    new_ids = list(range(new.shape[0]))
    flags = _af.find_similar_to_corrections(new, new_ids, corr, corr_labels, similarity_threshold=_THRESH)

    for i, rid in enumerate(new_ids):
        sims = [_cos(new[i], corr[j]) for j in range(corr.shape[0])]
        max_sim = max(sims)
        if max_sim >= _THRESH:
            assert rid in flags
            best = int(np.argmax(sims))
            assert flags[rid] == f"similar_to_known_error:{corr_labels[best]}"
        else:
            assert rid not in flags


def test_autoflag_empty_corrections_is_noop():
    flags = _af.find_similar_to_corrections(
        np.ones((2, _DIM), dtype=np.float32), [1, 2], np.zeros((0, 0), dtype=np.float32), []
    )
    assert flags == {}
