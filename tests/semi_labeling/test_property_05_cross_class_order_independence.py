# Feature: semi-labeling-pipeline-improvements, Property 5: Cross-class flag iteration-order independence
# For all cross-class pairs (A, B), the flagging result is identical whether the pair is
# evaluated as (A, B) or (B, A).
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_cc = sl_imports.load_step("step4_class_aware_dedup", "cross_class")
DedupDecision = sl_imports.load_step("step4_class_aware_dedup", "output_store").DedupDecision
Detection = sl_imports.load_step("step4_class_aware_dedup", "source_store").Detection

_pos = st.floats(min_value=0.0, max_value=180.0, allow_nan=False)
_side = st.floats(min_value=1.0, max_value=180.0, allow_nan=False)


def _det(rid, label, x, y, w, h):
    return Detection(
        result_id=rid, source_detection_id=rid, image_id=1, image_rel_path="img.png",
        image_path="img.png", source_input_dir="", predicted_label=label, detector_label="d",
        detector_score=0.9, predicted_probability=0.9, predicted_probability_pct=90.0,
        x1=x, y1=y, x2=x + w, y2=y + h, image_width=200, image_height=200,
    )


def _dec(rid, label):
    return DedupDecision(rid, "img.png", label, True, False, "g", rid, 0.0, 0.0, "")


def _flagged(decisions, by_id, emb):
    out = _cc.apply_cross_class_containment(decisions, by_id, emb)
    return {d.result_id for d in out if d.drop_reason == _cc.SUSPECT_REASON}


@given(
    a=st.tuples(_pos, _pos, _side, _side), b=st.tuples(_pos, _pos, _side, _side),
    la=st.sampled_from(["crack", "mold", "spall"]), lb=st.sampled_from(["crack", "mold", "spall"]),
    seed=st.integers(0, 10_000),
)
def test_cross_class_order_independence(a, b, la, lb, seed):
    rng = np.random.default_rng(seed)
    da, db = _det(0, la, *a), _det(1, lb, *b)
    by_id = {0: da, 1: db}
    emb = {0: rng.standard_normal(4).astype(np.float32), 1: rng.standard_normal(4).astype(np.float32)}

    forward = _flagged([_dec(0, la), _dec(1, lb)], by_id, emb)
    reverse = _flagged([_dec(1, lb), _dec(0, la)], by_id, emb)
    assert forward == reverse
