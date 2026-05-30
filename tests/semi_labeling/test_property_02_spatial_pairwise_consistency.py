# Feature: semi-labeling-pipeline-improvements, Property 2: Spatial pairwise consistency
# For all images: IoU/Containment in [0,1]; Containment and max_iou_same_label symmetric
# across a pair; sum(contains_count) == sum(contained_by_count); a single-detection image
# yields contains_count = contained_by_count = 0 and max_iou_same_label = max_containment = 0.0.
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

import object_detection.damage_scan.geometry as geo

_coord = st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False)
_geo_inputs = st.lists(
    st.tuples(_coord, _coord, _coord, _coord, st.sampled_from(["crack", "mold", "spall"])),
    min_size=1,
    max_size=6,
)


def _build(items):
    return [
        geo.GeoInput(detection_id=i, x1=x1, y1=y1, x2=x2, y2=y2, label=lbl)
        for i, (x1, y1, x2, y2, lbl) in enumerate(items)
    ]


@given(items=_geo_inputs)
def test_spatial_pairwise_consistency(items):
    dets = _build(items)
    results = geo.compute_box_geometry(dets, 256, 256)

    # Pairwise bounds + symmetry of the helpers.
    for i in range(len(dets)):
        for j in range(len(dets)):
            assert 0.0 <= geo.iou(dets[i], dets[j]) <= 1.0
            assert 0.0 <= geo.containment(dets[i], dets[j]) <= 1.0
            assert geo.containment(dets[i], dets[j]) == geo.containment(dets[j], dets[i])
            assert geo.iou(dets[i], dets[j]) == geo.iou(dets[j], dets[i])

    # Count consistency: every containment increments one contains and one contained_by.
    assert sum(r.contains_count for r in results) == sum(r.contained_by_count for r in results)

    if len(dets) == 1:
        only = results[0]
        assert only.contains_count == 0
        assert only.contained_by_count == 0
        assert only.max_iou_same_label == 0.0
        assert only.max_containment == 0.0
