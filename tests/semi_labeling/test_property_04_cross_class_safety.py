# Feature: semi-labeling-pipeline-improvements, Property 4: Cross-class flag-only safety and threshold gating
# Flagging never changes the number of keep==True nor any keep value; a suspect marker is
# set iff Containment>=0.85 and Cos_Sim>=0.80, on the smaller-area box (larger result_id on tie).
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_cc = sl_imports.load_step("step4_class_aware_dedup", "cross_class")
_pf = sl_imports.load_step("step4_class_aware_dedup", "pair_features")
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


def _decision(rid, label):
    return DedupDecision(rid, "img.png", label, True, False, "g", rid, 0.0, 0.0, "")


def _oracle(dets, emb):
    flagged = set()
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a, b = dets[i], dets[j]
            if a.predicted_label == b.predicted_label:
                continue
            cont = _pf.intersection_area(a, b) / max(min(_pf.area(a), _pf.area(b)), 1e-6)
            if cont < 0.85 or _pf.cosine_similarity(emb[a.result_id], emb[b.result_id]) < 0.80:
                continue
            aa, ab = _pf.area(a), _pf.area(b)
            flagged.add(a.result_id if aa < ab else b.result_id if ab < aa else max(a.result_id, b.result_id))
    return flagged


@given(
    rows=st.lists(
        st.tuples(st.sampled_from(["crack", "mold", "spall"]), _pos, _pos, _side, _side),
        min_size=1, max_size=4,
    ),
    seed=st.integers(0, 10_000),
)
def test_cross_class_flag_only_safety_and_gating(rows, seed):
    rng = np.random.default_rng(seed)
    dets = [_det(i, lbl, x, y, w, h) for i, (lbl, x, y, w, h) in enumerate(rows)]
    decisions = [_decision(d.result_id, d.predicted_label) for d in dets]
    by_id = {d.result_id: d for d in dets}
    emb = {d.result_id: rng.standard_normal(4).astype(np.float32) for d in dets}

    out = _cc.apply_cross_class_containment(decisions, by_id, emb)

    # Flag-only safety: keep count and each keep value unchanged.
    assert sum(d.keep for d in out) == sum(d.keep for d in decisions)
    assert [d.keep for d in out] == [d.keep for d in decisions]

    # Threshold gating (iff) against an independent oracle.
    got = {d.result_id for d in out if d.drop_reason == _cc.SUSPECT_REASON}
    assert got == _oracle(dets, emb)
