from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .coco import coco_image_maps, discover_coco_dataset, load_json, resolve_image_path
from .crops import clamp_xywh_to_xyxy, save_crop


def build_prototype_crop_bank(
    *,
    coco_root: str | Path,
    output_dir: str | Path,
    splits: tuple[str, ...] = ("train",),
    samples_per_class: int = 32,
    min_crop_size: int = 32,
    seed: int = 7,
) -> dict[str, Any]:
    from PIL import Image

    dataset = discover_coco_dataset(coco_root, splits=splits)
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    category_names = {category.id: category.name for category in dataset.categories}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split in dataset.splits:
        payload = load_json(split.annotation_path)
        images, anns_by_image = coco_image_maps(payload)
        for image_id, image in images.items():
            image_path = resolve_image_path(split, str(image.get("file_name", "")))
            width = int(image.get("width", 0) or 0)
            height = int(image.get("height", 0) or 0)
            for ann in anns_by_image.get(image_id, []):
                label = category_names.get(int(ann.get("category_id", -1)))
                if not label:
                    continue
                box = clamp_xywh_to_xyxy(list(ann.get("bbox", [])), width, height)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                if min(x2 - x1, y2 - y1) < int(min_crop_size):
                    continue
                candidates[label].append({"image_path": str(image_path), "box_xyxy": [x1, y1, x2, y2], "split": split.name})

    rng = random.Random(int(seed))
    manifest: dict[str, Any] = {"coco_root": str(Path(coco_root).expanduser().resolve()), "output_dir": str(out_root), "classes": {}}
    for label, rows in sorted(candidates.items()):
        rng.shuffle(rows)
        label_dir = out_root / label.replace(" ", "_")
        label_dir.mkdir(parents=True, exist_ok=True)
        saved_items = []
        for idx, row in enumerate(rows[: max(1, int(samples_per_class))], start=1):
            image_path = Path(row["image_path"])
            with Image.open(image_path) as image:
                crop_path = label_dir / f"{idx:04d}_{image_path.stem}.png"
                save_crop(image.convert("RGB"), tuple(row["box_xyxy"]), crop_path)
            saved_items.append({"path": str(crop_path), **row})
        manifest["classes"][label] = {"count": len(saved_items), "dir": str(label_dir), "items": saved_items}

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"prototype_dir": str(out_root), "manifest": str(manifest_path), "classes": manifest["classes"]}
