from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .coco import discover_coco_dataset


def write_stable_dino_dataset_yaml(*, coco_root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    dataset = discover_coco_dataset(coco_root)
    target = Path(output_path).expanduser().resolve() if output_path else dataset.root / "stable_dino_dataset.yaml"
    split_map = {split.name: split.annotation_path for split in dataset.splits}
    payload: dict[str, Any] = {
        "path": str(dataset.root),
        "nc": len(dataset.categories),
        "names": [category.name for category in dataset.categories],
    }
    if "train" in split_map:
        payload["train"] = str(split_map["train"])
    if "val" in split_map:
        payload["val"] = str(split_map["val"])
    elif "valid" in split_map:
        payload["val"] = str(split_map["valid"])
    if "test" in split_map:
        payload["test"] = str(split_map["test"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"dataset_yaml": str(target), "names": payload["names"], "splits": sorted(split_map)}


def ensure_stable_dino_dataset_yaml(coco_root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    dataset = discover_coco_dataset(coco_root)
    target = Path(output_path).expanduser().resolve() if output_path else dataset.root / "stable_dino_dataset.yaml"
    if target.is_file():
        return {"dataset_yaml": str(target), "reused": True}
    payload = write_stable_dino_dataset_yaml(coco_root=dataset.root, output_path=target)
    payload["reused"] = False
    return payload
