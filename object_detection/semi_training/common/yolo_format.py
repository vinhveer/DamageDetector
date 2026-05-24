from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from .coco import coco_image_maps, discover_coco_dataset, load_json, resolve_image_path


def _safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    raise ValueError("link_mode must be one of: symlink, copy")


def _yolo_label_line(ann: dict[str, Any], width: float, height: float, category_to_index: dict[int, int]) -> str | None:
    category_id = int(ann.get("category_id", -1))
    if category_id not in category_to_index:
        return None
    x, y, w, h = (float(v) for v in ann.get("bbox", [0, 0, 0, 0]))
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None
    x1 = max(0.0, min(width, x))
    y1 = max(0.0, min(height, y))
    x2 = max(0.0, min(width, x + w))
    y2 = max(0.0, min(height, y + h))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return None
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return f"{category_to_index[category_id]} {xc / width:.8f} {yc / height:.8f} {bw / width:.8f} {bh / height:.8f}"


def export_coco_root_to_yolo(
    *,
    coco_root: str | Path,
    output_dir: str | Path,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    dataset = discover_coco_dataset(coco_root)
    out_root = Path(output_dir).expanduser().resolve()
    category_to_index = {category.id: idx for idx, category in enumerate(dataset.categories)}
    names = [category.name for category in dataset.categories]
    split_counts: dict[str, dict[str, int]] = {}

    for split in dataset.splits:
        payload = load_json(split.annotation_path)
        images, anns_by_image = coco_image_maps(payload)
        image_out_dir = out_root / split.name / "images"
        label_out_dir = out_root / split.name / "labels"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
        image_count = 0
        ann_count = 0
        for image_id, image in sorted(images.items()):
            src = resolve_image_path(split, str(image.get("file_name", "")))
            dst = image_out_dir / src.name
            _safe_link_or_copy(src, dst, link_mode)
            label_lines = []
            width = float(image.get("width", 0) or 0)
            height = float(image.get("height", 0) or 0)
            for ann in anns_by_image.get(image_id, []):
                line = _yolo_label_line(ann, width, height, category_to_index)
                if line is not None:
                    label_lines.append(line)
            (label_out_dir / f"{src.stem}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            image_count += 1
            ann_count += len(label_lines)
        split_counts[split.name] = {"images": image_count, "annotations": ann_count}

    data_yaml = {
        "path": str(out_root),
        "train": "train/images",
        "val": "val/images" if (out_root / "val" / "images").exists() else "train/images",
        "nc": len(names),
        "names": names,
    }
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"yolo_dir": str(out_root), "data_yaml": str(yaml_path), "names": names, "splits": split_counts}


def ensure_yolo_dataset(coco_root: str | Path, output_dir: str | Path | None = None, *, link_mode: str = "symlink") -> dict[str, Any]:
    root = Path(coco_root).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve() if output_dir else root / "yolo"
    yaml_path = target / "data.yaml"
    if yaml_path.is_file():
        return {"yolo_dir": str(target), "data_yaml": str(yaml_path), "reused": True}
    payload = export_coco_root_to_yolo(coco_root=root, output_dir=target, link_mode=link_mode)
    payload["reused"] = False
    return payload
