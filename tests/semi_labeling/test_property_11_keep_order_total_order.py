# Feature: semi-labeling-pipeline-improvements, Property 11: Keep-ordering total order
# Ordering by (descending P_Good, descending box_area, ascending result_id) is a strict
# total order over distinct detections (no two distinct detections compare equal).
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

import sl_imports

_keep_order_key = sl_imports.load_step("step4_class_aware_dedup", "greedy_dedup")._keep_order_key
Detection = sl_imports.load_step("step4_class_aware_dedup", "source_store").Detection

_pct = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_pos = st.floats(min_value=0.0, max_value=180.0, allow_nan=False)
_side = st.floats(min_value=0.0, max_value=180.0, allow_nan=False)


def _det(result_id, pct, detector, x, y, w, h):
    return Detection(
        result_id=result_id, source_detection_id=result_id, image_id=1,
        image_rel_path="s.png", image_path="s.png", source_input_dir="",
        predicted_label="crack", detector_label="damage", detector_score=detector,
        predicted_probability=pct / 100.0, predicted_probability_pct=pct,
        x1=x, y1=y, x2=x + w, y2=y + h, image_width=200, image_height=200,
    )


@given(rows=st.lists(st.tuples(_pct, _unit, _pos, _pos, _side, _side), min_size=1, max_size=8))
def test_keep_ordering_total_order(rows):
    dets = [_det(i, *row) for i, row in enumerate(rows)]  # result_id = i is unique
    keys = [_keep_order_key(d) for d in dets]
    # Strict total order: distinct detections never share a full sort key.
    assert len(set(keys)) == len(keys)
    # Deterministic ordering.
    assert sorted(range(len(dets)), key=lambda i: keys[i]) == sorted(
        range(len(dets)), key=lambda i: keys[i]
    )
