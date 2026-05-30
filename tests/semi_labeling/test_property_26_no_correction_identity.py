# Feature: semi-labeling-pipeline-improvements, Property 26: No-correction identity
# load_training_data without a Correction_Store returns the dataset unchanged in size and content.
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import sl_imports

load_training_data = sl_imports.load_step("step6_classifier", "train_classifier").load_training_data


@given(
    X=st.integers(0, 8).flatmap(
        lambda n: hnp.arrays(np.float32, (n, 5), elements=st.floats(-2.0, 2.0, allow_nan=False, width=32))
    ),
    labels=st.lists(st.sampled_from(["crack", "mold", "spall", "reject"]), max_size=8),
)
def test_no_correction_identity(X, labels):
    y = np.array(labels[: X.shape[0]] + ["crack"] * (X.shape[0] - len(labels[: X.shape[0]])))
    new_X, new_y = load_training_data(X, y, None)
    assert new_X.shape == X.shape
    assert np.array_equal(new_X, X)
    assert list(new_y) == list(y)
