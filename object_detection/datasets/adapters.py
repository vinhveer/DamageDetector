from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .augment import build_stable_dino_augmentation, build_yolo_augmentation_overrides
from .manifest import DetectionDatasetManifest, DetectionSplit, iter_split_images


def build_yolo_training_kwargs(manifest: DetectionDatasetManifest, augmentation_profile: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"data": str(manifest.yaml_path)}
    kwargs.update(build_yolo_augmentation_overrides(augmentation_profile))
    return kwargs


def _build_categories(names: list[str]) -> list[dict[str, Any]]:
    return [{"id": index + 1, "name": name, "supercategory": "damage"} for index, name in enumerate(names)]


def _convert_yolo_label_line(parts: list[str], width: int, height: int, names: list[str], annotation_id: int, image_id: int) -> dict[str, Any]:
    class_index = int(parts[0])
    if class_index < 0 or class_index >= len(names):
        raise ValueError(f"Class index {class_index} is out of range for {len(names)} classes")

    x_center = float(parts[1]) * width
    y_center = float(parts[2]) * height
    box_width = float(parts[3]) * width
    box_height = float(parts[4]) * height
    x = max(0.0, x_center - (box_width / 2.0))
    y = max(0.0, y_center - (box_height / 2.0))
    box_width = max(0.0, min(box_width, width - x))
    box_height = max(0.0, min(box_height, height - y))

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_index + 1,
        "bbox": [x, y, box_width, box_height],
        "area": box_width * box_height,
        "iscrowd": 0,
        "segmentation": [[x, y, x + box_width, y, x + box_width, y + box_height, x, y + box_height]],
    }


def _load_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _split_cache_file(cache_root: Path, split_name: str) -> Path:
    return cache_root / f"{split_name}_coco.json"


def _export_split_to_coco(split: DetectionSplit, manifest: DetectionDatasetManifest, cache_root: Path) -> Path:
    if split.annotation_file is not None:
        return split.annotation_file
    if split.label_dir is None:
        raise ValueError(f"Split '{split.name}' does not define labels or COCO annotations")

    cache_root.mkdir(parents=True, exist_ok=True)
    output_path = _split_cache_file(cache_root, split.name)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(iter_split_images(split), start=1):
        width, height = _load_image_size(image_path)
        images.append(
            {
                "id": image_id,
                "file_name": str(image_path.resolve()),
                "width": width,
                "height": height,
            }
        )
        label_path = (split.label_dir / image_path.relative_to(split.image_dir)).with_suffix(".txt")
        if not label_path.exists():
            continue
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid YOLO label line in {label_path}: {raw_line}")
            annotations.append(
                _convert_yolo_label_line(parts, width, height, manifest.names, annotation_id, image_id)
            )
            annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": _build_categories(manifest.names),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def prepare_stable_dino_dataset(
    manifest: DetectionDatasetManifest,
    *,
    dataset_name_prefix: str,
    cache_root: str | Path,
) -> dict[str, Any]:
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import register_coco_instances

    cache_root = Path(cache_root).expanduser().resolve()
    dataset_root = manifest.dataset_root.resolve()

    def register(split: DetectionSplit | None, suffix: str) -> str | None:
        if split is None:
            return None
        annotation_file = _export_split_to_coco(split, manifest, cache_root)
        dataset_name = f"{dataset_name_prefix}_{suffix}"
        if dataset_name not in DatasetCatalog.list():
            register_coco_instances(dataset_name, {}, str(annotation_file), str(split.image_dir.resolve()))
        return dataset_name

    train_name = register(manifest.train, "train")
    val_name = register(manifest.val, "val") or train_name
    test_name = register(manifest.test, "test")

    return {
        "train_name": train_name,
        "val_name": val_name,
        "test_name": test_name,
        "num_classes": manifest.nc,
    }


def build_stable_dino_overrides(
    manifest: DetectionDatasetManifest,
    *,
    dataset_name_prefix: str,
    cache_root: str | Path,
    augmentation_profile: str,
    image_size: int,
    batch_size: int,
    workers: int,
    device: str,
    output_dir: str,
    init_checkpoint: str | None,
) -> list[str]:
    dataset_info = prepare_stable_dino_dataset(
        manifest,
        dataset_name_prefix=dataset_name_prefix,
        cache_root=cache_root,
    )
    aug = build_stable_dino_augmentation(augmentation_profile, image_size)

    def _string_override(key: str, value: str) -> str:
        return f"{key}={json.dumps(str(value))}"

    overrides = [
        _string_override("dataloader.train.dataset.names", str(dataset_info["train_name"])),
        _string_override("dataloader.test.dataset.names", str(dataset_info["val_name"])),
        _string_override("dataloader.evaluator.dataset_name", str(dataset_info["val_name"])),
        f"dataloader.train.total_batch_size={int(batch_size)}",
        f"dataloader.train.num_workers={int(workers)}",
        _string_override("train.output_dir", str(output_dir)),
        _string_override("train.device", str(device)),
        _string_override("model.device", str(device)),
        f"model.num_classes={int(dataset_info['num_classes'])}",
        f"dataloader.train.mapper.augmentation.image_size={int(aug['image_size'])}",
        f"dataloader.train.mapper.augmentation.min_scale={float(aug['min_scale'])}",
        f"dataloader.train.mapper.augmentation.max_scale={float(aug['max_scale'])}",
        _string_override("dataloader.train.mapper.augmentation.random_flip", str(aug["random_flip"])),
    ]
    if init_checkpoint:
        overrides.append(_string_override("train.init_checkpoint", str(init_checkpoint)))
    return overrides
