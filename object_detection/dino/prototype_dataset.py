from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    label: str
    box_xyxy: tuple[int, int, int, int]


def _canonical_label(text: str) -> str:
    return " ".join(str(text or "").replace("_", " ").replace("-", " ").strip().lower().split())


def _sanitize_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    ix1 = max(0, min(width, int(round(x1))))
    iy1 = max(0, min(height, int(round(y1))))
    ix2 = max(0, min(width, int(round(x2))))
    iy2 = max(0, min(height, int(round(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _read_data_yaml(dataset_dir: str) -> dict[str, Any]:
    root = Path(dataset_dir).expanduser().resolve()
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Cannot find data.yaml in {root}")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YOLO data.yaml in {data_yaml}")
    return data


def _resolve_split_dirs(dataset_dir: str, split: str) -> tuple[Path, Path]:
    root = Path(dataset_dir).expanduser().resolve()
    return root / split / "images", root / split / "labels"


def _load_yolo_boxes(label_path: Path, *, names: Sequence[str], width: int, height: int, pad_ratio: float) -> list[YoloBox]:
    items: list[YoloBox] = []
    raw = label_path.read_text(encoding="utf-8").splitlines()
    for row in raw:
        parts = row.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(names):
            continue
        xc = float(parts[1]) * width
        yc = float(parts[2]) * height
        bw = float(parts[3]) * width
        bh = float(parts[4]) * height
        pad_x = bw * max(0.0, float(pad_ratio))
        pad_y = bh * max(0.0, float(pad_ratio))
        box = _sanitize_box(
            xc - (bw / 2.0) - pad_x,
            yc - (bh / 2.0) - pad_y,
            xc + (bw / 2.0) + pad_x,
            yc + (bh / 2.0) + pad_y,
            width=width,
            height=height,
        )
        if box is None:
            continue
        items.append(YoloBox(class_id=class_id, label=str(names[class_id]), box_xyxy=box))
    return items


def _iter_dataset_samples(dataset_dir: str, splits: Sequence[str]) -> Iterable[tuple[str, Path, Path]]:
    for split in splits:
        image_dir, label_dir = _resolve_split_dirs(dataset_dir, split)
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir(), key=lambda path: path.name.lower()):
            if image_path.suffix.lower() not in _IMAGE_EXTENSIONS or not image_path.is_file():
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            yield split, image_path, label_path


def _box_overlap_ratio(crop_box: tuple[int, int, int, int], gt_box: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = crop_box
    bx1, by1, bx2, by2 = gt_box
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    crop_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    return inter_area / crop_area


def _pick_background_box(
    *,
    width: int,
    height: int,
    positive_boxes: Sequence[tuple[int, int, int, int]],
    typical_sizes: Sequence[tuple[int, int]],
    min_size: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    if width < min_size or height < min_size:
        return None
    sizes = list(typical_sizes) or [(min(width, 256), min(height, 256))]
    for _ in range(80):
        candidate_w, candidate_h = rng.choice(sizes)
        candidate_w = max(min_size, min(width, int(candidate_w)))
        candidate_h = max(min_size, min(height, int(candidate_h)))
        if candidate_w >= width:
            x1 = 0
        else:
            x1 = rng.randint(0, width - candidate_w)
        if candidate_h >= height:
            y1 = 0
        else:
            y1 = rng.randint(0, height - candidate_h)
        crop_box = (x1, y1, x1 + candidate_w, y1 + candidate_h)
        if all(_box_overlap_ratio(crop_box, gt_box) <= 0.02 for gt_box in positive_boxes):
            return crop_box
    step = max(8, min_size // 2)
    for candidate_w, candidate_h in sizes + [(min_size, min_size)]:
        candidate_w = max(min_size, min(width, int(candidate_w)))
        candidate_h = max(min_size, min(height, int(candidate_h)))
        max_x = max(0, width - candidate_w)
        max_y = max(0, height - candidate_h)
        for y1 in range(0, max_y + 1, step):
            for x1 in range(0, max_x + 1, step):
                crop_box = (x1, y1, x1 + candidate_w, y1 + candidate_h)
                if all(_box_overlap_ratio(crop_box, gt_box) <= 0.02 for gt_box in positive_boxes):
                    return crop_box
    return None


def build_prototypes_from_yolo_dataset(
    *,
    dataset_dir: str,
    output_dir: str,
    splits: Sequence[str] = ("train", "valid"),
    samples_per_class: int = 24,
    background_samples: int = 24,
    pad_ratio: float = 0.08,
    min_crop_size: int = 64,
    seed: int = 7,
) -> dict[str, Any]:
    from PIL import Image

    data = _read_data_yaml(dataset_dir)
    raw_names = data.get("names") or []
    if isinstance(raw_names, dict):
        ordered = [raw_names[index] for index in sorted(raw_names)]
    else:
        ordered = list(raw_names)
    names = [str(item) for item in ordered]
    if not names:
        raise ValueError("No class names found in data.yaml")

    root = Path(dataset_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))

    candidates_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in names}
    typical_sizes: list[tuple[int, int]] = []
    background_sources: list[dict[str, Any]] = []

    for split, image_path, label_path in _iter_dataset_samples(str(root), splits):
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            boxes = _load_yolo_boxes(label_path, names=names, width=width, height=height, pad_ratio=pad_ratio)
        if not boxes:
            continue
        positive_boxes = [item.box_xyxy for item in boxes]
        background_sources.append(
            {
                "split": split,
                "image_path": str(image_path),
                "boxes": positive_boxes,
                "width": width,
                "height": height,
            }
        )
        for box in boxes:
            x1, y1, x2, y2 = box.box_xyxy
            typical_sizes.append((x2 - x1, y2 - y1))
            candidates_by_label[box.label].append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label": box.label,
                    "class_id": box.class_id,
                    "box_xyxy": [x1, y1, x2, y2],
                }
            )

    manifest: dict[str, Any] = {
        "dataset_dir": str(root),
        "output_dir": str(out_root),
        "splits": list(splits),
        "classes": {},
        "background": {"count": 0},
    }

    for label in names:
        label_dir = out_root / _canonical_label(label).replace(" ", "_")
        label_dir.mkdir(parents=True, exist_ok=True)
        chosen = list(candidates_by_label.get(label) or [])
        rng.shuffle(chosen)
        saved = 0
        records: list[dict[str, Any]] = []
        for entry in chosen:
            if saved >= max(1, int(samples_per_class)):
                break
            x1, y1, x2, y2 = [int(v) for v in entry["box_xyxy"]]
            with Image.open(entry["image_path"]) as image:
                crop = image.convert("RGB").crop((x1, y1, x2, y2))
            if crop.size[0] < min_crop_size or crop.size[1] < min_crop_size:
                continue
            filename = f"{saved + 1:03d}_{Path(entry['image_path']).stem}.png"
            crop_path = label_dir / filename
            crop.save(crop_path)
            saved += 1
            records.append(
                {
                    "path": str(crop_path),
                    "source_image": str(entry["image_path"]),
                    "box_xyxy": [x1, y1, x2, y2],
                    "split": entry["split"],
                }
            )
        manifest["classes"][label] = {
            "count": saved,
            "dir": str(label_dir),
            "items": records,
        }

    if background_samples > 0:
        bg_dir = out_root / "background"
        bg_dir.mkdir(parents=True, exist_ok=True)
        shuffled_sources = list(background_sources)
        rng.shuffle(shuffled_sources)
        saved = 0
        records: list[dict[str, Any]] = []
        source_index = 0
        while saved < int(background_samples) and shuffled_sources:
            source = shuffled_sources[source_index % len(shuffled_sources)]
            source_index += 1
            crop_box = _pick_background_box(
                width=int(source["width"]),
                height=int(source["height"]),
                positive_boxes=list(source["boxes"]),
                typical_sizes=typical_sizes,
                min_size=int(min_crop_size),
                rng=rng,
            )
            if crop_box is None:
                if source_index >= len(shuffled_sources) * 4:
                    break
                continue
            x1, y1, x2, y2 = crop_box
            with Image.open(source["image_path"]) as image:
                crop = image.convert("RGB").crop((x1, y1, x2, y2))
            if crop.size[0] < min_crop_size or crop.size[1] < min_crop_size:
                continue
            filename = f"{saved + 1:03d}_{Path(source['image_path']).stem}.png"
            crop_path = bg_dir / filename
            crop.save(crop_path)
            saved += 1
            records.append(
                {
                    "path": str(crop_path),
                    "source_image": str(source["image_path"]),
                    "box_xyxy": [x1, y1, x2, y2],
                    "split": source["split"],
                }
            )
        manifest["background"] = {
            "count": saved,
            "dir": str(bg_dir),
            "items": records,
        }

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
