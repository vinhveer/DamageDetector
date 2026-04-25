from __future__ import annotations

from pathlib import Path

from .models import Detection, ImageInfo


_COLORS_BGR = {
    "full_raw": (120, 120, 120),
    "final": (255, 255, 255),
    "crack": (40, 40, 240),
    "mold": (40, 180, 40),
    "spall": (0, 140, 255),
}


def _read_image(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def _color_for(det: Detection) -> tuple[int, int, int]:
    if det.stage in {"refine", "final"}:
        return _COLORS_BGR.get(det.prompt_key, (255, 255, 255))
    return _COLORS_BGR.get(det.stage, (255, 255, 255))


def _line_width(det: Detection) -> int:
    if det.stage == "full_raw":
        return 1
    return 2


def save_overlay(
    *,
    image: ImageInfo,
    detections: list[Detection],
    output_path: Path,
    include_proposals: bool = False,
    include_proposal_raw: bool = False,
) -> None:
    import cv2

    canvas = _read_image(image.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    draw_items = [
        det
        for det in detections
        if det.stage == "final"
        or (include_proposals and det.stage != "full_raw")
        or (include_proposal_raw and det.stage == "full_raw")
    ]
    order = {"full_raw": 0, "final": 1}
    draw_items.sort(key=lambda det: order.get(det.stage, 99))

    for det in draw_items:
        x1, y1, x2, y2 = det.box.as_int_xyxy()
        color = _color_for(det)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, _line_width(det))
        label = f"{det.prompt_key}:{det.score:.2f}" if det.stage == "final" else f"{det.stage}:{det.score:.2f}"
        y_text = max(14, y1 - 6)
        cv2.putText(canvas, label, (x1, y_text), font, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x1, y_text), font, 0.48, color, 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(output_path.suffix or ".png", canvas)
    if not ok:
        raise RuntimeError(f"Could not encode overlay: {output_path}")
    buf.tofile(str(output_path))
