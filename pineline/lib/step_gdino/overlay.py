from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray | None,
    *,
    house_boxes: list[dict],
    neg_boxes: list[dict],
    positive_points: list[tuple[float, float]],
    negative_points: list[tuple[float, float]],
    crop_bbox: tuple[int, int, int, int] | None,
    out_path: Path,
    alpha: float = 0.40,
) -> None:
    """Vẽ overlay debug cho bước cắt nhà.

    - mask nhà: tô xanh lá mờ
    - house box: xanh lá  |  window/door box: đỏ
    - điểm dương: chấm xanh dương  |  điểm âm: chấm đỏ
    - crop bbox: khung vàng
    """
    overlay = image_rgb.copy()
    if mask is not None and mask.shape == image_rgb.shape[:2] and mask.any():
        green = np.zeros_like(image_rgb)
        green[..., 1] = 255
        m3 = np.stack([mask] * 3, axis=-1)
        overlay = np.where(
            m3, (overlay * (1 - alpha) + green * alpha).astype(np.uint8), overlay
        )

    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    thickness = max(2, int(round(min(h, w) / 500)))
    r = max(3, int(round(min(h, w) / 300)))

    def _draw_boxes(boxes: list[dict], color: tuple[int, int, int]) -> None:
        for det in boxes:
            box = det.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)
            text = f"{det.get('label', '?')} {float(det.get('score', 0)):.2f}"
            cv2.putText(
                bgr, text, (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.5, min(h, w) / 1600.0),
                color, thickness, cv2.LINE_AA,
            )

    _draw_boxes(house_boxes, (0, 200, 0))    # house = xanh lá
    _draw_boxes(neg_boxes, (0, 0, 255))      # window/door = đỏ

    for x, y in positive_points:
        cv2.circle(bgr, (int(round(x)), int(round(y))), r, (255, 0, 0), -1, cv2.LINE_AA)
    for x, y in negative_points:
        cv2.circle(bgr, (int(round(x)), int(round(y))), r, (0, 0, 255), -1, cv2.LINE_AA)

    if crop_bbox is not None:
        x1, y1, x2, y2 = crop_bbox
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255),
                      max(2, int(round(min(h, w) / 400))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)
