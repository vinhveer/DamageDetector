# Feature: semi-labeling-pipeline-improvements, Property 18: Cosine similarity ordering and range
# similarity_to_texts returns one value per text in the same order as provided, each in
# [-1, 1] (tested via the pure cosine layer with mock features).
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import sl_imports

cosine_to_texts = sl_imports.load_step("step2_sematic", "negatives").cosine_to_texts

_elem = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False, width=32)


@given(
    image_feat=hnp.arrays(np.float32, 6, elements=_elem),
    text_feats=st.integers(0, 5).flatmap(lambda n: hnp.arrays(np.float32, (n, 6), elements=_elem)),
)
def test_cosine_order_and_range(image_feat, text_feats):
    sims = cosine_to_texts(image_feat, text_feats)
    assert sims.shape[0] == text_feats.shape[0]  # one per text, in order
    assert np.all(sims >= -1.0 - 1e-6) and np.all(sims <= 1.0 + 1e-6)
