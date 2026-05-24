from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_COCO_EXPORT_VERSION = 2


@dataclass(frozen=True)
class StableDinoSplitConfig:
    name: str
    image_dirs: tuple[Path, ...]
    label_dirs: tuple[Path | None, ...]
    annotation_file: Path | None

    @property
    def image_dir(self) -> Path:
        return self.image_dirs[0]


@dataclass(frozen=True)
class StableDinoDatasetConfig:
    yaml_path: Path
    dataset_root: Path
    names: list[str]
    nc: int
    train: StableDinoSplitConfig | None
    val: StableDinoSplitConfig | None
    test: StableDinoSplitConfig | None
    raw: dict[str, Any]


def _normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, dict):
        items = sorted(((int(key), str(value)) for key, value in raw_names.items()), key=lambda item: item[0])
        return [value for _, value in items]
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    raise ValueError("Stable-DINO dataset config must define 'names' as a list or mapping.")


def _resolve_root(yaml_path: Path, data: dict[str, Any]) -> Path:
    root_value = str(data.get("path") or "").strip()
    if not root_value:
        return yaml_path.parent.resolve()
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def _resolve_path(root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _resolve_paths(root: Path, value: Any) -> tuple[Path, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    return tuple(path for item in values if (path := _resolve_path(root, item)) is not None)


def _guess_coco_image_dir(root: Path, split_name: str) -> Path:
    split_dir = root / "images" / split_name
    if split_dir.is_dir():
        return split_dir
    image_dir = root / "images"
    if image_dir.is_dir():
        return image_dir
    return root


def _guess_label_dir(image_dir: Path) -> Path | None:
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    return image_dir.parent / "labels"


def _build_split(name: str, root: Path, data: dict[str, Any]) -> StableDinoSplitConfig | None:
    split_paths = _resolve_paths(root, data.get(name))
    if not split_paths:
        return None
    if len(split_paths) == 1 and split_paths[0].suffix.lower() == ".json":
        annotation_file = split_paths[0]
        image_dir = _guess_coco_image_dir(root, name)
        return StableDinoSplitConfig(name=name, image_dirs=(image_dir,), label_dirs=(None,), annotation_file=annotation_file)

    annotation_files = [path for path in split_paths if path.suffix.lower() == ".json"]
    if annotation_files:
        raise ValueError(f"Split '{name}' cannot mix COCO JSON annotations with image directories")
    label_dirs = tuple(_guess_label_dir(path) for path in split_paths)
    return StableDinoSplitConfig(name=name, image_dirs=split_paths, label_dirs=label_dirs, annotation_file=None)


def load_stable_dino_dataset_config(path: str | Path) -> StableDinoDatasetConfig:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Stable-DINO dataset config not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid Stable-DINO dataset config: {yaml_path}")

    names = _normalize_names(data.get("names"))
    nc = int(data.get("nc") or len(names))
    if nc != len(names):
        raise ValueError(f"Dataset nc={nc} does not match names={len(names)} in {yaml_path}")

    dataset_root = _resolve_root(yaml_path, data)
    train = _build_split("train", dataset_root, data)
    val = _build_split("val", dataset_root, data) or _build_split("valid", dataset_root, data)
    test = _build_split("test", dataset_root, data)
    return StableDinoDatasetConfig(
        yaml_path=yaml_path,
        dataset_root=dataset_root,
        names=names,
        nc=nc,
        train=train,
        val=val,
        test=test,
        raw=data,
    )


def _iter_split_images(split: StableDinoSplitConfig) -> list[Path]:
    if split.annotation_file is not None:
        return []
    images: list[Path] = []
    for image_dir in split.image_dirs:
        if not image_dir.exists():
            raise FileNotFoundError(f"Split image directory not found: {image_dir}")
        images.extend(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
    return sorted(images)


def _build_categories(names: list[str]) -> list[dict[str, Any]]:
    return [{"id": index + 1, "name": name, "supercategory": "damage"} for index, name in enumerate(names)]


def _bbox_polygon(x1: float, y1: float, box_width: float, box_height: float) -> list[list[float]]:
    x2 = x1 + box_width
    y2 = y1 + box_height
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


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
        "segmentation": _bbox_polygon(x1, y1, box_width, box_height),
        "iscrowd": 0,
    }


def _load_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _path_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


def _label_path_for_image(split: StableDinoSplitConfig, image_path: Path) -> Path | None:
    for image_dir, label_dir in zip(split.image_dirs, split.label_dirs):
        try:
            rel_path = image_path.relative_to(image_dir)
        except ValueError:
            continue
        if label_dir is None:
            return None
        return (label_dir / rel_path).with_suffix(".txt")
    raise ValueError(f"Image path is outside split image dirs: {image_path}")


def _split_fingerprint(split: StableDinoSplitConfig, image_paths: list[Path]) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    for image_path in image_paths:
        label_path = _label_path_for_image(split, image_path)
        if label_path is not None and label_path.exists():
            labels.append(_path_fingerprint(label_path))
    return {
        "export_version": _COCO_EXPORT_VERSION,
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


def _export_split_to_coco(split: StableDinoSplitConfig, dataset: StableDinoDatasetConfig, cache_root: Path) -> Path:
    if split.annotation_file is not None:
        return split.annotation_file
    if not any(label_dir is not None for label_dir in split.label_dirs):
        raise ValueError(f"Split '{split.name}' does not define labels or COCO annotations")

    cache_root.mkdir(parents=True, exist_ok=True)
    output_path = cache_root / f"{split.name}_coco.json"
    meta_path = cache_root / f"{split.name}_coco.meta.json"
    image_paths = _iter_split_images(split)
    fingerprint = _split_fingerprint(split, image_paths)
    if output_path.exists() and _load_cache_meta(meta_path) == fingerprint:
        return output_path

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, start=1):
        width, height = _load_image_size(image_path)
        images.append({"id": image_id, "file_name": str(image_path.resolve()), "width": width, "height": height})
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
            annotation = _convert_yolo_label_line(parts, width, height, dataset.names, annotation_id, image_id)
            if annotation is None:
                continue
            annotations.append(annotation)
            annotation_id += 1

    payload = {"images": images, "annotations": annotations, "categories": _build_categories(dataset.names)}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    meta_path.write_text(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return output_path


def _normalize_profile(profile: str) -> str:
    value = str(profile or "balanced").strip().lower()
    if value in {"default", "medium"}:
        return "balanced"
    if value == "strong":
        return "aggressive"
    return value


def _build_augmentation(profile: str, image_size: int) -> dict[str, object]:
    normalized = _normalize_profile(profile)
    if normalized == "light":
        return {"image_size": int(image_size), "min_scale": 0.8, "max_scale": 1.25, "random_flip": "horizontal"}
    if normalized == "aggressive":
        return {"image_size": int(image_size), "min_scale": 0.1, "max_scale": 2.0, "random_flip": "horizontal"}
    return {"image_size": int(image_size), "min_scale": 0.3, "max_scale": 1.7, "random_flip": "horizontal"}


def prepare_stable_dino_dataset(
    dataset: StableDinoDatasetConfig,
    *,
    dataset_name_prefix: str,
    cache_root: str | Path,
) -> dict[str, Any]:
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import register_coco_instances

    cache_root = Path(cache_root).expanduser().resolve()

    def register(split: StableDinoSplitConfig | None, suffix: str) -> str | None:
        if split is None:
            return None
        annotation_file = _export_split_to_coco(split, dataset, cache_root)
        dataset_name = f"{dataset_name_prefix}_{suffix}"
        if dataset_name not in DatasetCatalog.list():
            register_coco_instances(dataset_name, {}, str(annotation_file), str(split.image_dir.resolve()))
        return dataset_name

    train_name = register(dataset.train, "train")
    val_name = register(dataset.val, "val") or train_name
    test_name = register(dataset.test, "test")
    return {"train_name": train_name, "val_name": val_name, "test_name": test_name, "num_classes": dataset.nc}


def build_stable_dino_overrides(
    dataset: StableDinoDatasetConfig,
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
    eval_image_size: int | None = None,
    eval_max_size: int | None = None,
) -> list[str]:
    dataset_info = prepare_stable_dino_dataset(
        dataset,
        dataset_name_prefix=dataset_name_prefix,
        cache_root=cache_root,
    )
    aug = _build_augmentation(augmentation_profile, image_size)
    eval_key = f"{str(eval_split or 'val').strip().lower()}_name"
    eval_name = dataset_info.get(eval_key) or dataset_info.get("val_name") or dataset_info.get("train_name")

    eval_short = int(eval_image_size) if eval_image_size and int(eval_image_size) > 0 else int(image_size)
    eval_long = int(eval_max_size) if eval_max_size and int(eval_max_size) > 0 else eval_short * 2
    if eval_long < eval_short:
        eval_long = eval_short

    def string_override(key: str, value: str) -> str:
        return f"{key}={json.dumps(str(value))}"

    overrides = [
        string_override("dataloader.train.dataset.names", str(dataset_info["train_name"])),
        string_override("dataloader.test.dataset.names", str(eval_name)),
        string_override("dataloader.evaluator.dataset_name", str(eval_name)),
        string_override("dataloader.evaluator.output_dir", str(output_dir)),
        f"dataloader.train.total_batch_size={int(batch_size)}",
        f"dataloader.train.num_workers={int(workers)}",
        f"dataloader.test.num_workers={int(workers)}",
        string_override("train.output_dir", str(output_dir)),
        string_override("train.device", str(device)),
        string_override("model.device", str(device)),
        f"model.num_classes={int(dataset_info['num_classes'])}",
        f"dataloader.train.mapper.augmentation.image_size={int(aug['image_size'])}",
        f"dataloader.train.mapper.augmentation.min_scale={float(aug['min_scale'])}",
        f"dataloader.train.mapper.augmentation.max_scale={float(aug['max_scale'])}",
        string_override("dataloader.train.mapper.augmentation.random_flip", str(aug["random_flip"])),
        f"dataloader.test.mapper.augmentation.0.short_edge_length={eval_short}",
        f"dataloader.test.mapper.augmentation.0.max_size={eval_long}",
    ]
    if pin_memory is not None:
        overrides.append(f"dataloader.train.pin_memory={bool(pin_memory)}")
    if persistent_workers is not None and int(workers) > 0:
        overrides.append(f"dataloader.train.persistent_workers={bool(persistent_workers)}")
    if prefetch_factor is not None and int(prefetch_factor) > 0 and int(workers) > 0:
        overrides.append(f"dataloader.train.prefetch_factor={int(prefetch_factor)}")
    if init_checkpoint:
        overrides.append(string_override("train.init_checkpoint", str(init_checkpoint)))
    return overrides
