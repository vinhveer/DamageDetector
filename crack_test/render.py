from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


BOX_COLOR = (30, 144, 255)
MASK_COLOR = np.array([34, 197, 94], dtype=np.float32)
TEXT_COLOR = (255, 255, 255)


def _line_width(width: int) -> int:
    return max(2, int(round(width / 500)))


def _font_scale(width: int) -> float:
    return max(0.45, min(1.2, width / 1800.0))


def draw_boxes(image_bgr: np.ndarray, boxes: list[dict], *, labels: bool) -> np.ndarray:
    out = image_bgr.copy()
    h_img, w_img = out.shape[:2]
    line_w = _line_width(w_img)
    font_scale = _font_scale(w_img)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
        x1 = max(0, min(w_img - 1, x1))
        x2 = max(0, min(w_img - 1, x2))
        y1 = max(0, min(h_img - 1, y1))
        y2 = max(0, min(h_img - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, line_w)
        if not labels:
            continue
        text = f"{box.get('label', 'crack')} {float(box.get('score', 0.0)):.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_w)
        top = max(0, y1 - th - baseline - 6)
        cv2.rectangle(out, (x1, top), (min(w_img - 1, x1 + tw + 8), y1), BOX_COLOR, -1)
        cv2.putText(out, text, (x1 + 4, max(th + 2, y1 - baseline - 3)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, line_w, cv2.LINE_AA)
    return out


def draw_mask(image_bgr: np.ndarray, mask: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    out = image_bgr.copy()
    mask_bool = np.asarray(mask) > 0
    if not np.any(mask_bool):
        return out
    blended = (out[mask_bool].astype(np.float32) * (1.0 - alpha)) + (MASK_COLOR * alpha)
    out[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def save_four_overlays(image_path: Path, boxes: list[dict], mask: np.ndarray, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    artifacts = {
        "overlay_box_only": output_dir / "overlay_box_only.png",
        "overlay_box_label": output_dir / "overlay_box_label.png",
        "overlay_segmentation": output_dir / "overlay_segmentation.png",
        "overlay_segmentation_box_label": output_dir / "overlay_segmentation_box_label.png",
    }
    cv2.imwrite(str(artifacts["overlay_box_only"]), draw_boxes(image_bgr, boxes, labels=False))
    cv2.imwrite(str(artifacts["overlay_box_label"]), draw_boxes(image_bgr, boxes, labels=True))
    cv2.imwrite(str(artifacts["overlay_segmentation"]), draw_mask(image_bgr, mask))
    combined = draw_boxes(draw_mask(image_bgr, mask), boxes, labels=True)
    cv2.imwrite(str(artifacts["overlay_segmentation_box_label"]), combined)
    return {key: str(path) for key, path in artifacts.items()}


def save_mask(mask: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.asarray(mask) > 0).astype(np.uint8) * 255)
    return str(path)
