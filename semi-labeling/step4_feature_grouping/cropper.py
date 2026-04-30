from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from models import KeptBox


def resolve_image_path(box: KeptBox, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(box.image_rel_path or "").strip()
    stored_path = str(box.image_path or "").strip()
    source_input_dir = Path(str(box.source_input_dir or "")).expanduser()
    if image_root is not None:
        root = image_root.expanduser().resolve()
        candidates.append(root / rel_path)
        if stored_path:
            candidates.append(root / Path(stored_path).name)
    if stored_path:
        stored = Path(stored_path).expanduser()
        candidates.append(stored if stored.is_absolute() else source_input_dir / stored_path)
    if rel_path:
        candidates.append(source_input_dir / rel_path)
    if stored_path:
        candidates.append(source_input_dir / Path(stored_path).name)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    if image_root is not None:
        return (image_root.expanduser().resolve() / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()


def crop_box(box: KeptBox, image_root: Path | None, *, padding_ratio: float) -> Image.Image:
    image_path = resolve_image_path(box, image_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pad_x = max(0.0, float(box.x2) - float(box.x1)) * float(padding_ratio)
        pad_y = max(0.0, float(box.y2) - float(box.y1)) * float(padding_ratio)
        x1 = max(0, int(math.floor(float(box.x1) - pad_x)))
        y1 = max(0, int(math.floor(float(box.y1) - pad_y)))
        x2 = min(width, int(math.ceil(float(box.x2) + pad_x)))
        y2 = min(height, int(math.ceil(float(box.y2) + pad_y)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop box result_id={box.result_id}: {(x1, y1, x2, y2)}")
        return rgb.crop((x1, y1, x2, y2))
