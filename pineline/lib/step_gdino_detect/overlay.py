from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2


# BGR per detector; label text still carries the damage class.
_MODEL_PALETTE = {
    "gdino": (255, 255, 0),       # cyan
    "yolo": (255, 0, 255),        # magenta
    "stabledino": (0, 165, 255),  # orange
}


def _detector_name(det: dict) -> str:
    raw = str(det.get("detector_name") or "").strip().lower()
    if raw:
        return raw
    label = str(det.get("label") or "")
    if ":" in label:
        return label.split(":", 1)[0].strip().lower()
    return "unknown"


def _color_for_detector(name: str) -> tuple[int, int, int]:
    return _MODEL_PALETTE.get(str(name or "").lower(), (255, 255, 255))


def _draw_legend(img, *, font_scale: float, thickness: int) -> None:
    x, y = 12, 12
    pad = 6
    line_h = max(18, int(24 * font_scale))
    width = 260
    height = line_h * (len(_MODEL_PALETTE) + 1) + pad * 2
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.putText(img, "Detector colors", (x + pad, y + line_h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    for idx, (name, color) in enumerate(_MODEL_PALETTE.items(), start=1):
        yy = y + line_h * (idx + 1)
        cv2.rectangle(img, (x + pad, yy - line_h + 4), (x + pad + 28, yy - 4), color, -1)
        cv2.putText(img, name, (x + pad + 38, yy - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def write_overlay(
    *,
    image_path: Path,
    detections: Iterable[dict],
    out_path: Path,
) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    thickness = max(2, int(round(min(h, w) / 400)))
    font_scale = max(0.5, min(h, w) / 1500.0)
    _draw_legend(img, font_scale=font_scale, thickness=thickness)

    for det in detections:
        box = det.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detector_name = _detector_name(det)
        color = _color_for_detector(detector_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        name = str(det.get("group_name", "?")).lower()
        text = f"{detector_name}:{name} {float(det.get('score', 0)):.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(th + 2, y1)
        cv2.rectangle(img, (x1, ty - th - bl), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def write_detector_overlays(
    *,
    image_path: Path,
    detections: Iterable[dict],
    out_dir: Path,
    image_id: str,
) -> dict[str, str]:
    grouped: dict[str, list[dict]] = {}
    for det in detections:
        grouped.setdefault(_detector_name(det), []).append(dict(det))
    written: dict[str, str] = {}
    for detector_name, rows in grouped.items():
        if not rows:
            continue
        out_path = out_dir / detector_name / f"{image_id}.png"
        write_overlay(image_path=image_path, detections=rows, out_path=out_path)
        written[detector_name] = str(out_path)
    return written
