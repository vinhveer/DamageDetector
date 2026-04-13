from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import numpy as np
from PIL import Image

from data_split.types import SampleRecord

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def normalize_split_ratios(raw_ratios: Iterable[float]) -> list[float]:
    ratios = [float(value) for value in raw_ratios]
    if len(ratios) != 3:
        raise ValueError("Exactly three split ratios are required.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return [value / total for value in ratios]


def infer_source_id(stem: str) -> str:
    candidate = re.sub(r"_dup\d+$", "", stem)
    parts = candidate.split("_")
    if len(parts) > 4 and all(part.isdigit() for part in parts[-4:]):
        return "_".join(parts[:-4])
    if len(parts) > 2 and all(part.isdigit() for part in parts[-2:]):
        return "_".join(parts[:-2])
    return candidate


def compute_mask_positive_ratio(mask_path: Path | None, threshold: int) -> float:
    if mask_path is None:
        return 0.0
    with Image.open(mask_path) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    if pixels.size == 0:
        return 0.0
    return float((pixels > threshold).mean())


def discover_samples(input_root: Path, mask_threshold: int) -> list[SampleRecord]:
    images_dir = input_root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    masks_dir = input_root / "masks"
    has_masks = masks_dir.is_dir()

    records: list[SampleRecord] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = resolve_mask_path(masks_dir, image_path.relative_to(images_dir)) if has_masks else None
        positive_ratio = compute_mask_positive_ratio(mask_path, mask_threshold) if mask_path is not None else 0.0
        records.append(
            SampleRecord(
                image_path=image_path,
                mask_path=mask_path,
                stem=image_path.stem,
                source_id=infer_source_id(image_path.stem),
                positive_ratio=positive_ratio,
            )
        )
    if not records:
        raise FileNotFoundError(f"No images found under: {images_dir}")
    return records


def resolve_mask_path(masks_dir: Path, relative_image_path: Path) -> Path | None:
    direct = masks_dir / relative_image_path
    if direct.is_file():
        return direct
    parent = direct.parent
    if not parent.is_dir():
        return None
    matches = [path for path in parent.glob(f"{relative_image_path.stem}.*") if path.is_file() and not path.name.startswith(".")]
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.suffix.lower())[0]
