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


def _convert_yolo_label_line(
    parts: list[str],
    width: int,
    height: int,
    names: list[str],
    annotation_id: int,
    image_id: int,
) -> dict[str, Any] | None:
    class_index = int(parts[0])
    if class_index < 0 or class_index >= len(names):
        raise ValueError(f"Class index {class_index} is out of range for {len(names)} classes")

    x_center = float(parts[1]) * width
    y_center = float(parts[2]) * height
    raw_width = float(parts[3]) * width
    raw_height = float(parts[4]) * height
    x1 = max(0.0, x_center - (raw_width / 2.0))
    y1 = max(0.0, y_center - (raw_height / 2.0))
    x2 = min(float(width), x_center + (raw_width / 2.0))
    y2 = min(float(height), y_center + (raw_height / 2.0))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width <= 0.0 or box_height <= 0.0:
        return None

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_index + 1,
        "bbox": [x1, y1, box_width, box_height],
        "area": box_width * box_height,
        "iscrowd": 0,
    }


def _load_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _split_cache_file(cache_root: Path, split_name: str) -> Path:
    return cache_root / f"{split_name}_coco.json"


def _split_cache_meta_file(cache_root: Path, split_name: str) -> Path:
    return cache_root / f"{split_name}_coco.meta.json"


def _path_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def _label_path_for_image(split: DetectionSplit, image_path: Path) -> Path | None:
    for image_dir, label_dir in zip(split.image_dirs, split.label_dirs):
        try:
            rel_path = image_path.relative_to(image_dir)
        except ValueError:
            continue
        if label_dir is None:
            return None
        return (label_dir / rel_path).with_suffix(".txt")
    raise ValueError(f"Image path is outside split image dirs: {image_path}")


def _split_fingerprint(split: DetectionSplit, image_paths: list[Path]) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    for image_path in image_paths:
        label_path = _label_path_for_image(split, image_path)
        if label_path is not None and label_path.exists():
            labels.append(_path_fingerprint(label_path))
    return {
        "image_dirs": [str(path.resolve()) for path in split.image_dirs],
        "label_dirs": [str(path.resolve()) if path is not None else None for path in split.label_dirs],
        "images": [_path_fingerprint(path) for path in image_paths],
        "labels": labels,
    }


def _load_cache_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _export_split_to_coco(split: DetectionSplit, manifest: DetectionDatasetManifest, cache_root: Path) -> Path:
    if split.annotation_file is not None:
        return split.annotation_file
    if not any(label_dir is not None for label_dir in split.label_dirs):
        raise ValueError(f"Split '{split.name}' does not define labels or COCO annotations")

    cache_root.mkdir(parents=True, exist_ok=True)
    output_path = _split_cache_file(cache_root, split.name)
    meta_path = _split_cache_meta_file(cache_root, split.name)
    image_paths = iter_split_images(split)
    fingerprint = _split_fingerprint(split, image_paths)
    if output_path.exists() and _load_cache_meta(meta_path) == fingerprint:
        return output_path

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        width, height = _load_image_size(image_path)
        images.append(
            {
                "id": image_id,
                "file_name": str(image_path.resolve()),
                "width": width,
                "height": height,
            }
        )
        label_path = _label_path_for_image(split, image_path)
        if label_path is None or not label_path.exists():
            continue
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid YOLO label line in {label_path}: {raw_line}")
            annotation = _convert_yolo_label_line(parts, width, height, manifest.names, annotation_id, image_id)
            if annotation is None:
                continue
            annotations.append(annotation)
            annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": _build_categories(manifest.names),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    meta_path.write_text(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
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
    pin_memory: bool | None,
    persistent_workers: bool | None,
    prefetch_factor: int | None,
    device: str,
    output_dir: str,
    init_checkpoint: str | None,
    eval_split: str = "val",
) -> list[str]:
    dataset_info = prepare_stable_dino_dataset(
        manifest,
        dataset_name_prefix=dataset_name_prefix,
        cache_root=cache_root,
    )
    aug = build_stable_dino_augmentation(augmentation_profile, image_size)
    eval_key = f"{str(eval_split or 'val').strip().lower()}_name"
    eval_name = dataset_info.get(eval_key) or dataset_info.get("val_name") or dataset_info.get("train_name")

    def _string_override(key: str, value: str) -> str:
        return f"{key}={json.dumps(str(value))}"

    overrides = [
        _string_override("dataloader.train.dataset.names", str(dataset_info["train_name"])),
        _string_override("dataloader.test.dataset.names", str(eval_name)),
        _string_override("dataloader.evaluator.dataset_name", str(eval_name)),
        _string_override("dataloader.evaluator.output_dir", str(output_dir)),
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
    if pin_memory is not None:
        overrides.append(f"dataloader.train.pin_memory={bool(pin_memory)}")
    if persistent_workers is not None and int(workers) > 0:
        overrides.append(f"dataloader.train.persistent_workers={bool(persistent_workers)}")
    if prefetch_factor is not None and int(prefetch_factor) > 0 and int(workers) > 0:
        overrides.append(f"dataloader.train.prefetch_factor={int(prefetch_factor)}")
    if init_checkpoint:
        overrides.append(_string_override("train.init_checkpoint", str(init_checkpoint)))
    return overrides
