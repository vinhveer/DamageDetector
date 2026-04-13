from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path


def prepare_crack500_dataset(
    source_root: Path,
    target_root: Path,
    *,
    overwrite: bool = True,
) -> None:
    source_root = source_root.expanduser().resolve()
    target_root = target_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source dataset root: {source_root}")

    images_root = target_root / "images"
    masks_root = target_root / "masks"
    if overwrite and target_root.exists():
        shutil.rmtree(target_root)
    images_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    image_name_counts: Counter[str] = Counter()
    ordered_images: list[tuple[str, Path, Path, Path]] = []
    for split in ["train", "val", "test"]:
        split_root = source_root / split
        split_images = split_root / "images"
        split_masks = split_root / "masks"
        if not split_images.is_dir():
            continue
        for image_path in sorted(split_images.iterdir()):
            if not image_path.is_file() or image_path.name.startswith("."):
                continue
            mask_path = split_masks / image_path.name.replace(".jpg", ".png")
            if not mask_path.is_file():
                alt_mask = split_masks / image_path.name
                if alt_mask.is_file():
                    mask_path = alt_mask
            ordered_images.append((split, image_path, mask_path, split_masks))
            image_name_counts[image_path.name] += 1

    copied_images = 0
    copied_masks = 0
    duplicate_index: Counter[str] = Counter()
    for _split, image_path, mask_path, _split_masks in ordered_images:
        duplicate_index[image_path.name] += 1
        if image_name_counts[image_path.name] == 1:
            merged_image_name = image_path.name
        else:
            if duplicate_index[image_path.name] == 1:
                merged_image_name = image_path.name
            else:
                merged_image_name = f"{image_path.stem}_dup{duplicate_index[image_path.name]}{image_path.suffix}"

        shutil.copy2(image_path, images_root / merged_image_name)
        copied_images += 1

        if mask_path.is_file():
            if image_name_counts[image_path.name] == 1 or duplicate_index[image_path.name] == 1:
                merged_mask_name = f"{Path(merged_image_name).stem}{mask_path.suffix}"
            else:
                merged_mask_name = f"{Path(merged_image_name).stem}{mask_path.suffix}"
            shutil.copy2(mask_path, masks_root / merged_mask_name)
            copied_masks += 1

    print(f"Merged crack500 dataset into {target_root}")
    print(f"images={copied_images} masks={copied_masks}")
