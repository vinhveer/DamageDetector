from __future__ import annotations

from pathlib import Path
from typing import Any


def clamp_xywh_to_xyxy(bbox: list[float] | tuple[float, ...], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) < 4:
        return None
    x, y, w, h = (float(v) for v in bbox[:4])
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None
    x1 = max(0, min(width, int(round(x))))
    y1 = max(0, min(height, int(round(y))))
    x2 = max(0, min(width, int(round(x + w))))
    y2 = max(0, min(height, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def clamp_xyxy(box: list[float] | tuple[float, ...], width: int, height: int, *, expand_ratio: float = 0.0) -> tuple[int, int, int, int] | None:
    if len(box) < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if expand_ratio > 0:
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        pad_x = bw * float(expand_ratio)
        pad_y = bh * float(expand_ratio)
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y
    ix1 = max(0, min(width, int(round(x1))))
    iy1 = max(0, min(height, int(round(y1))))
    ix2 = max(0, min(width, int(round(x2))))
    iy2 = max(0, min(height, int(round(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def save_crop(image: Any, box_xyxy: tuple[int, int, int, int], output_path: str | Path) -> Path:
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box_xyxy).save(out)
    return out
