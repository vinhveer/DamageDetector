# Feature: semi-labeling-pipeline-improvements, Property 1: Geometry scalar invariants
# For all boxes (incl. inverted/zero), box_width>=0, box_height>=0, box_area>=0,
# area_ratio_to_image in [0,1] for in-bounds boxes, elongation>=1.0, and each scalar
# equals its defining formula.
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

import object_detection.damage_scan.geometry as geo
from sl_strategies import boxes


@given(
    box=boxes(coord=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False)),
    image_width=st.integers(min_value=1, max_value=4000),
    image_height=st.integers(min_value=1, max_value=4000),
)
def test_geometry_scalar_invariants(box, image_width, image_height):
    x1, y1, x2, y2 = box
    g = geo.GeoInput(detection_id=1, x1=x1, y1=y1, x2=x2, y2=y2, label="crack")
    (bg,) = geo.compute_box_geometry([g], image_width, image_height)

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    assert bg.box_width == w >= 0.0
    assert bg.box_height == h >= 0.0
    assert bg.box_area == w * h >= 0.0
    assert bg.center_x == (x1 + x2) / 2.0
    assert bg.center_y == (y1 + y2) / 2.0
    assert bg.aspect_ratio == w / max(h, 1e-6)
    assert bg.elongation == geo.elongation(w, h) >= 1.0

    # in-bounds (coords within image) => area ratio in [0, 1]
    if x2 <= image_width and y2 <= image_height:
        assert 0.0 <= bg.area_ratio_to_image <= 1.0
    assert bg.area_ratio_to_image == bg.box_area / max(1, image_width * image_height)


@given(box=boxes())
def test_area_ratio_zero_for_nonpositive_image_dims(box):
    # Requirement 1.12: image_width or image_height <= 0 => area_ratio_to_image = 0.0
    x1, y1, x2, y2 = box
    g = geo.GeoInput(detection_id=1, x1=x1, y1=y1, x2=x2, y2=y2, label="crack")
    (bg,) = geo.compute_box_geometry([g], 0, 100)
    assert bg.area_ratio_to_image == 0.0
