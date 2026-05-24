from __future__ import annotations

from typing import Sequence


def sample_points_in_box(
    box: tuple[float, float, float, float],
    *,
    image_w: int,
    image_h: int,
    points_per_box: int = 5,
) -> list[tuple[float, float]]:
    """Sample N points inside a box: 1 center + (N-1) interior anchors.

    Points are clipped to image bounds. Returns list of (x, y) floats.
    """
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(image_w) - 1, float(x1)))
    y1 = max(0.0, min(float(image_h) - 1, float(y1)))
    x2 = max(0.0, min(float(image_w) - 1, float(x2)))
    y2 = max(0.0, min(float(image_h) - 1, float(y2)))
    if x2 <= x1 or y2 <= y1:
        return []
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if int(points_per_box) <= 1:
        return [(cx, cy)]

    # offset points: 0.25 and 0.75 of width/height (quarter anchors)
    pts: list[tuple[float, float]] = [(cx, cy)]
    qx1 = x1 + (x2 - x1) * 0.25
    qx2 = x1 + (x2 - x1) * 0.75
    qy1 = y1 + (y2 - y1) * 0.25
    qy2 = y1 + (y2 - y1) * 0.75
    candidates = [
        (qx1, qy1), (qx2, qy1),
        (qx1, qy2), (qx2, qy2),
        (cx, qy1), (cx, qy2),
        (qx1, cy), (qx2, cy),
    ]
    needed = max(0, int(points_per_box) - 1)
    pts.extend(candidates[:needed])
    return pts
