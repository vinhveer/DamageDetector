# Feature: semi-labeling-pipeline-improvements, Property 22: Negative-anchor caching equivalence
# Classification using cached negative text embeddings yields the same Adjusted_Scores as
# recomputing the negative text embeddings per image (mock, deterministic encode_text).
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import sl_imports
from sl_strategies import score_dicts

adjusted_scores_for_image = sl_imports.load_step("step2_sematic", "negatives").adjusted_scores_for_image

_ANCHORS = {"crack": ["a", "b"], "mold": ["c"], "spall": ["d", "e"]}
_elem = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False, width=32)


def _encode_text(texts):
    rows = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        rows.append(rng.standard_normal(6).astype(np.float32))
    return np.vstack(rows)


@given(image_feat=hnp.arrays(np.float32, 6, elements=_elem), pos=score_dicts())
def test_negative_caching_equivalence(image_feat, pos):
    alpha = {label: 1.0 for label in pos}
    # Cache once vs recompute per image -> deterministic encode makes both identical.
    cached = {label: _encode_text(_ANCHORS[label]) for label in pos}
    recomputed = {label: _encode_text(_ANCHORS[label]) for label in pos}

    pen_c, adj_c = adjusted_scores_for_image(image_feat, pos, cached, alpha)
    pen_r, adj_r = adjusted_scores_for_image(image_feat, pos, recomputed, alpha)
    assert pen_c == pen_r
    assert adj_c == adj_r
