from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RgbExport:
    """Result of converting an RGBA cutout to an RGB image for GDINO.

    Coordinates produced on the exported image are in *cropped* space; add
    (offset_x, offset_y) to map a detection box back to the original cutout.
    """
    width: int               # width of the exported (cropped) RGB
    height: int              # height of the exported (cropped) RGB
    offset_x: int            # left of valid bbox in the original cutout
    offset_y: int            # top of valid bbox in the original cutout
    orig_width: int          # original cutout width
    orig_height: int         # original cutout height


def rgba_to_rgb_on_black(
    crop_path: Path,
    out_path: Path,
    *,
    pad: int = 8,
) -> RgbExport:
    """Convert an RGBA cutout to RGB and crop away empty (alpha==0) borders.

    Empty regions carry no information; feeding the surrounding black padding
    to GroundingDINO wastes compute and hurts the text-region matching. We
    therefore tightly crop to the valid (alpha>0) bounding box, keeping a
    small ``pad`` margin for context. Pixels that are still transparent inside
    that bbox are painted black so the engine's valid-mask logic can exclude
    them. The returned offsets let callers map boxes back to the original
    cutout coordinate system.
    """
    img = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read crop image: {crop_path}")

    if img.ndim == 2:
        rgb_full = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif img.shape[2] == 4:
        bgr = img[..., :3]
        alpha = img[..., 3]
        m = (alpha > 0).astype(np.uint8)[..., None]
        rgb_full = bgr * m
    else:
        rgb_full = img
        alpha = None

    orig_h, orig_w = rgb_full.shape[:2]

    # Determine the valid bounding box. Prefer the alpha channel; fall back to
    # non-black pixels when there is no alpha channel.
    if alpha is not None:
        valid = alpha > 0
    else:
        valid = np.max(rgb_full[:, :, :3], axis=2) > 0

    ys, xs = np.where(valid)
    if len(xs) == 0 or len(ys) == 0:
        # Fully empty cutout: nothing to feed the model. Write a 1x1 black
        # pixel so downstream file handling stays consistent; caller can skip.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), np.zeros((1, 1, 3), dtype=rgb_full.dtype))
        return RgbExport(
            width=0, height=0, offset_x=0, offset_y=0,
            orig_width=int(orig_w), orig_height=int(orig_h),
        )

    x1 = max(0, int(xs.min()) - int(pad))
    y1 = max(0, int(ys.min()) - int(pad))
    x2 = min(orig_w, int(xs.max()) + 1 + int(pad))
    y2 = min(orig_h, int(ys.max()) + 1 + int(pad))

    rgb = rgb_full[y1:y2, x1:x2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rgb)
    h, w = rgb.shape[:2]
    return RgbExport(
        width=int(w), height=int(h),
        offset_x=int(x1), offset_y=int(y1),
        orig_width=int(orig_w), orig_height=int(orig_h),
    )
