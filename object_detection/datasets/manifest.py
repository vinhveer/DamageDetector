from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DetectionSplit:
    name: str
    image_dir: Path
    label_dir: Path | None
    annotation_file: Path | None


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
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _guess_label_dir(image_dir: Path) -> Path | None:
    if not image_dir.exists():
        return None
    if image_dir.name == "images":
        candidate = image_dir.parent / "labels"
        return candidate if candidate.exists() else candidate
    candidate = image_dir.parent / "labels"
    return candidate if candidate.exists() else candidate


def _build_split(name: str, root: Path, data: dict[str, Any]) -> DetectionSplit | None:
    split_path = _resolve_path(root, data.get(name))
    if split_path is None:
        return None
    annotation_file = None
    label_dir = None
    if split_path.suffix.lower() == ".json":
        annotation_file = split_path
    else:
        label_dir = _guess_label_dir(split_path)
    return DetectionSplit(name=name, image_dir=split_path, label_dir=label_dir, annotation_file=annotation_file)


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
    if not split.image_dir.exists():
        raise FileNotFoundError(f"Split image directory not found: {split.image_dir}")
    return sorted(path for path in split.image_dir.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
