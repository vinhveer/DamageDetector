from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DetectionSplit:
    name: str
    image_dirs: tuple[Path, ...]
    label_dirs: tuple[Path | None, ...]
    annotation_file: Path | None

    @property
    def image_dir(self) -> Path:
        return self.image_dirs[0]

    @property
    def label_dir(self) -> Path | None:
        return self.label_dirs[0] if self.label_dirs else None


@dataclass(frozen=True)
class DetectionDatasetManifest:
    yaml_path: Path
    dataset_root: Path
    names: list[str]
    nc: int
    train: DetectionSplit | None
    val: DetectionSplit | None
    test: DetectionSplit | None
    raw: dict[str, Any]

    def get_split(self, split_name: str) -> DetectionSplit | None:
        normalized = str(split_name or "").strip().lower()
        if normalized in {"val", "valid", "validation"}:
            return self.val
        if normalized == "test":
            return self.test
        return self.train


def _normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, dict):
        items = sorted(((int(key), str(value)) for key, value in raw_names.items()), key=lambda item: item[0])
        return [value for _, value in items]
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    raise ValueError("Dataset manifest must define 'names' as a list or mapping.")


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
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    return tuple(path for item in values if (path := _resolve_path(root, item)) is not None)


def _guess_label_dir(image_dir: Path) -> Path | None:
    if not image_dir.exists():
        return None
    if image_dir.name == "images":
        candidate = image_dir.parent / "labels"
        return candidate if candidate.exists() else candidate
    candidate = image_dir.parent / "labels"
    return candidate if candidate.exists() else candidate


def _build_split(name: str, root: Path, data: dict[str, Any]) -> DetectionSplit | None:
    split_paths = _resolve_paths(root, data.get(name))
    if not split_paths:
        return None
    if len(split_paths) == 1 and split_paths[0].suffix.lower() == ".json":
        annotation_file = split_paths[0]
        return DetectionSplit(name=name, image_dirs=(root,), label_dirs=(None,), annotation_file=annotation_file)

    annotation_files = [path for path in split_paths if path.suffix.lower() == ".json"]
    if annotation_files:
        raise ValueError(f"Split '{name}' cannot mix COCO JSON annotations with image directories")
    label_dirs = tuple(_guess_label_dir(path) for path in split_paths)
    return DetectionSplit(name=name, image_dirs=tuple(split_paths), label_dirs=label_dirs, annotation_file=None)


def load_detection_dataset(path: str | Path) -> DetectionDatasetManifest:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset manifest: {yaml_path}")

    names = _normalize_names(data.get("names"))
    nc = int(data.get("nc") or len(names))
    if nc != len(names):
        raise ValueError(f"Dataset nc={nc} does not match names={len(names)} in {yaml_path}")

    dataset_root = _resolve_root(yaml_path, data)
    train = _build_split("train", dataset_root, data)
    val = _build_split("val", dataset_root, data) or _build_split("valid", dataset_root, data)
    test = _build_split("test", dataset_root, data)
    return DetectionDatasetManifest(
        yaml_path=yaml_path,
        dataset_root=dataset_root,
        names=names,
        nc=nc,
        train=train,
        val=val,
        test=test,
        raw=data,
    )


def iter_split_images(split: DetectionSplit) -> list[Path]:
    if split.annotation_file is not None:
        return []
    images: list[Path] = []
    for image_dir in split.image_dirs:
        if not image_dir.exists():
            raise FileNotFoundError(f"Split image directory not found: {image_dir}")
        images.extend(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
    return sorted(images)
