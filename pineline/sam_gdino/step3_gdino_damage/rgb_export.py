from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def rgba_to_rgb_on_black(crop_path: Path, out_path: Path) -> tuple[int, int]:
    """Convert RGBA crop -> RGB with alpha=0 pixels painted black.

    Black background lets GroundingDINO's valid-mask logic auto-exclude
    non-bridge regions. Returns (width, height).
    """
    img = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read crop image: {crop_path}")
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr = img[..., :3]
        alpha = img[..., 3]
        m = (alpha > 0).astype(np.uint8)[..., None]
        rgb = bgr * m
    else:
        rgb = img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rgb)
    h, w = rgb.shape[:2]
    return int(w), int(h)
