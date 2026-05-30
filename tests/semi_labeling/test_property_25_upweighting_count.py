# Feature: semi-labeling-pipeline-improvements, Property 25: Correction upweighting count
# Added rows == number of confirm/relabel corrections x w, each carrying its corrected_label.
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import sl_imports

load_training_data = sl_imports.load_step("step6_classifier", "train_classifier").load_training_data
_DIM = 6
_elem = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False, width=32)


def _corr(ctype, corrected):
    blob = np.zeros(_DIM, dtype="<f4").tobytes()
    return SimpleNamespace(correction_type=ctype, corrected_label=corrected, embedding_blob=blob)


@given(
    base_n=st.integers(0, 5),
    corrections=st.lists(
        st.tuples(st.sampled_from(["confirm", "relabel", "reject"]), st.sampled_from(["crack", "mold", "spall"])),
        max_size=12,
    ),
    weight=st.integers(1, 5),
)
def test_upweighting_count(base_n, corrections, weight):
    X = np.zeros((base_n, _DIM), dtype=np.float32)
    y = np.array(["crack"] * base_n)
    store = SimpleNamespace(iter_corrections=lambda: [_corr(t, lbl) for t, lbl in corrections])

    new_X, new_y = load_training_data(X, y, store, correction_weight=weight)

    upweightable = [(t, lbl) for t, lbl in corrections if t in ("confirm", "relabel")]
    expected_added = len(upweightable) * weight
    assert new_X.shape[0] - base_n == expected_added
    assert len(new_y) - base_n == expected_added
    # Each added correction carries its corrected_label, replicated `weight` times.
    expected_tail = [lbl for _, lbl in upweightable for _ in range(weight)]
    assert list(new_y[base_n:]) == expected_tail
