from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import Category, CocoSplit, SemiDataset


DEFAULT_SPLITS = ("train", "val")


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"COCO file must contain a JSON object: {path}")
    return payload


def write_json(path: str | Path, payload: Any, *, pretty: bool = True) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None, separators=None if pretty else (",", ":"))
    out.write_text(text, encoding="utf-8")
    return out


def discover_coco_dataset(root: str | Path, *, splits: tuple[str, ...] = DEFAULT_SPLITS) -> SemiDataset:
    dataset_root = Path(root).expanduser().resolve()
    ann_dir = dataset_root / "annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"COCO annotations directory not found: {ann_dir}")

    split_items: list[CocoSplit] = []
    categories: tuple[Category, ...] | None = None
    for split in splits:
        ann_path = ann_dir / f"instances_{split}.json"
        if not ann_path.is_file():
            continue
        payload = load_json(ann_path)
        if categories is None:
            categories = tuple(
                Category(id=int(item["id"]), name=str(item["name"]))
                for item in sorted(payload.get("categories", []), key=lambda row: int(row.get("id", 0)))
            )
        image_dir = dataset_root / "images" / split
        if not image_dir.is_dir():
            image_dir = dataset_root / "images"
        split_items.append(CocoSplit(name=split, annotation_path=ann_path, image_dir=image_dir))

    if not split_items:
        raise FileNotFoundError(f"No instances_<split>.json files found in {ann_dir}")
    if not categories:
        raise ValueError(f"No COCO categories found in {split_items[0].annotation_path}")
    return SemiDataset(root=dataset_root, categories=categories, splits=tuple(split_items))


def resolve_image_path(split: CocoSplit, file_name: str) -> Path:
    raw = Path(str(file_name))
    if raw.is_absolute():
        return raw
    candidates = [split.image_dir / raw, split.image_dir.parent / raw, split.image_dir.parent / split.name / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def coco_image_maps(payload: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images = {int(item["id"]): item for item in payload.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    for ann in payload.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        if image_id in anns_by_image:
            anns_by_image[image_id].append(ann)
    return images, anns_by_image


def list_split_image_paths(root: str | Path, split_name: str = "val") -> list[Path]:
    dataset = discover_coco_dataset(root, splits=(split_name,))
    split = dataset.splits[0]
    payload = load_json(split.annotation_path)
    images, _ = coco_image_maps(payload)
    return [resolve_image_path(split, str(image.get("file_name", ""))) for _, image in sorted(images.items())]


def list_split_image_paths_by_class(root: str | Path, split_name: str = "val") -> dict[str, list[Path]]:
    """Return {class_name: [images with at least one annotation for that class]}."""
    dataset = discover_coco_dataset(root, splits=(split_name,))
    split = dataset.splits[0]
    payload = load_json(split.annotation_path)
    images, anns_by_image = coco_image_maps(payload)
    categories = {category.id: category.name for category in dataset.categories}
    result: dict[str, set[Path]] = {}
    for image_id, anns in anns_by_image.items():
        image = images.get(image_id, {})
        image_path = resolve_image_path(split, str(image.get("file_name", "")))
        for ann in anns:
            label = categories.get(int(ann.get("category_id", -1)))
            if label:
                result.setdefault(label, set()).add(image_path)
    return {label: sorted(paths) for label, paths in sorted(result.items())}
