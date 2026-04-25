from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageSummary:
    split: str
    image_path: str
    mask_path: str
    width: int
    height: int
    component_count: int
    box_count: int


def _iter_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _resolve_mask_for_image(mask_dir: Path, image_path: Path) -> Path | None:
    stem = image_path.stem
    preferred = mask_dir / f"{stem}.png"
    if preferred.exists():
        return preferred
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: expensive but robust
    matches = list(mask_dir.rglob(f"{stem}.*"))
    for match in matches:
        if match.is_file() and match.suffix.lower() in IMAGE_EXTS:
            return match
    return None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: Literal["symlink", "hardlink", "copy"]) -> None:
    _ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return

    if mode == "copy":
        import shutil

        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # Likely cross-device; fall back to copy.
            import shutil

            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError:
            # Fall back to copy (Windows / permissions / SIP / etc.).
            import shutil

            shutil.copy2(src, dst)
            return

    raise ValueError(f"Unknown mode: {mode}")


def _tight_bbox_from_mask(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def _clip_bbox(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(int(x0), int(width)))
    y0 = max(0, min(int(y0), int(height)))
    x1 = max(0, min(int(x1), int(width)))
    y1 = max(0, min(int(y1), int(height)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _bbox_to_yolo(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
    class_index: int,
) -> str:
    bw = max(0.0, float(x1 - x0))
    bh = max(0.0, float(y1 - y0))
    cx = float(x0) + (bw / 2.0)
    cy = float(y0) + (bh / 2.0)
    # Normalize.
    nx = cx / float(width)
    ny = cy / float(height)
    nw = bw / float(width)
    nh = bh / float(height)
    return f"{int(class_index)} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"


def _split_component_into_boxes(
    component_mask: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    pad: int,
    min_box_side: int,
    min_box_area: int,
    max_box_len_px: int,
    overlap: float,
    max_aspect_ratio: float,
    max_area_ratio: float,
) -> list[tuple[int, int, int, int]]:
    x, y, w, h = bbox
    x0, y0, x1, y1 = _clip_bbox(x - pad, y - pad, x + w + pad, y + h + pad, image_width, image_height)
    w0 = x1 - x0
    h0 = y1 - y0
    if w0 <= 0 or h0 <= 0:
        return []

    img_area = float(image_width * image_height)
    area_ratio = float(w0 * h0) / img_area if img_area > 0 else 0.0
    aspect = float(max(w0 / max(1, h0), h0 / max(1, w0)))
    too_long = int(max(w0, h0)) > int(max_box_len_px)
    too_aspect = aspect > float(max_aspect_ratio)
    too_large = area_ratio > float(max_area_ratio)

    # If acceptable, return the padded bbox.
    if not (too_long or too_aspect or too_large):
        if w0 >= int(min_box_side) and h0 >= int(min_box_side) and (w0 * h0) >= int(min_box_area):
            return [(x0, y0, x1, y1)]
        return []

    # Otherwise, split into tiles and tighten bbox per tile using the component pixels.
    max_box_len_px = max(8, int(max_box_len_px))
    overlap = float(overlap)
    overlap = 0.0 if overlap < 0 else 0.85 if overlap > 0.85 else overlap
    stride = max(1, int(round(max_box_len_px * (1.0 - overlap))))

    def iter_windows() -> list[tuple[int, int, int, int]]:
        windows: list[tuple[int, int, int, int]] = []
        # If component is big in both dimensions, do a 2D grid.
        split_both = too_large and (w0 > max_box_len_px) and (h0 > max_box_len_px)
        if split_both:
            xs = list(range(x0, max(x0 + 1, x1), stride))
            ys = list(range(y0, max(y0 + 1, y1), stride))
            for wx0 in xs:
                wx1 = min(x1, wx0 + max_box_len_px)
                if wx1 - wx0 < 2:
                    continue
                for wy0 in ys:
                    wy1 = min(y1, wy0 + max_box_len_px)
                    if wy1 - wy0 < 2:
                        continue
                    windows.append((wx0, wy0, wx1, wy1))
            return windows

        # Split only along the long dimension.
        if w0 >= h0:
            xs = list(range(x0, max(x0 + 1, x1), stride))
            for wx0 in xs:
                wx1 = min(x1, wx0 + max_box_len_px)
                if wx1 - wx0 < 2:
                    continue
                windows.append((wx0, y0, wx1, y1))
            return windows
        ys = list(range(y0, max(y0 + 1, y1), stride))
        for wy0 in ys:
            wy1 = min(y1, wy0 + max_box_len_px)
            if wy1 - wy0 < 2:
                continue
            windows.append((x0, wy0, x1, wy1))
        return windows

    boxes: list[tuple[int, int, int, int]] = []
    for wx0, wy0, wx1, wy1 in iter_windows():
        crop = component_mask[wy0:wy1, wx0:wx1]
        tight = _tight_bbox_from_mask(crop)
        if tight is None:
            continue
        tx0, ty0, tx1, ty1 = tight
        gx0, gy0, gx1, gy1 = _clip_bbox(wx0 + tx0 - pad, wy0 + ty0 - pad, wx0 + tx1 + pad, wy0 + ty1 + pad, image_width, image_height)
        bw = gx1 - gx0
        bh = gy1 - gy0
        if bw < int(min_box_side) or bh < int(min_box_side) or (bw * bh) < int(min_box_area):
            continue
        boxes.append((gx0, gy0, gx1, gy1))

    # De-duplicate near-identical boxes produced by overlaps.
    uniq: list[tuple[int, int, int, int]] = []
    seen = set()
    for b in boxes:
        key = (int(b[0] // 2), int(b[1] // 2), int(b[2] // 2), int(b[3] // 2))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(b)
    return uniq


def _mask_to_boxes(
    mask: np.ndarray,
    *,
    threshold: int,
    min_component_area: int,
    image_width: int,
    image_height: int,
    pad: int,
    min_box_side: int,
    min_box_area: int,
    max_box_len_ratio: float,
    max_aspect_ratio: float,
    max_area_ratio: float,
    tile_overlap: float,
) -> tuple[list[tuple[int, int, int, int]], int]:
    binary = (np.asarray(mask, dtype=np.uint8) > int(threshold)).astype(np.uint8)
    if int(binary.sum()) == 0:
        return [], 0

    # Remove tiny specks to avoid many micro-boxes.
    if int(min_component_area) > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        kept_components = 0
        for idx in range(1, int(num_labels)):
            if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_component_area):
                keep[labels == idx] = 1
                kept_components += 1
        binary = keep
    else:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        kept_components = int(num_labels) - 1

    if int(binary.sum()) == 0:
        return [], 0

    # Recompute CC on cleaned mask for bbox stats.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    max_box_len_px = int(round(float(max(image_width, image_height)) * float(max_box_len_ratio)))
    max_box_len_px = max(16, max_box_len_px)

    boxes: list[tuple[int, int, int, int]] = []
    for idx in range(1, int(num_labels)):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        component_mask = (labels == idx).astype(np.uint8)
        boxes.extend(
            _split_component_into_boxes(
                component_mask,
                bbox=(x, y, w, h),
                image_width=int(image_width),
                image_height=int(image_height),
                pad=int(pad),
                min_box_side=int(min_box_side),
                min_box_area=int(min_box_area),
                max_box_len_px=int(max_box_len_px),
                overlap=float(tile_overlap),
                max_aspect_ratio=float(max_aspect_ratio),
                max_area_ratio=float(max_area_ratio),
            )
        )
    return boxes, kept_components


def _write_manifest_yaml(yaml_path: Path, *, dataset_dir: Path, class_name: str, include_test: bool) -> None:
    _ensure_parent(yaml_path)
    # If the manifest lives inside the dataset root, YOLO expects `path: .`.
    # Otherwise keep a relative path (or fall back to absolute).
    try:
        if yaml_path.parent.resolve() == dataset_dir.resolve():
            rel_root = "."
        else:
            rel_root = os.path.relpath(str(dataset_dir.resolve()), str(yaml_path.parent.resolve()))
    except Exception:
        rel_root = str(dataset_dir)
    lines = [
        f"# Auto-generated from Crack500 segmentation masks.",
        f"path: {rel_root}",
        "train: train/images",
        "val: val/images",
    ]
    if include_test:
        lines.append("test: test/images")
    lines.extend(
        [
            "",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    yaml_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Crack500 segmentation masks to YOLO object detection labels (for YOLO + StableDINO)."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("BestDatasets/crack500"),
        help="Source Crack500 dataset root containing train/val/test with images/ and masks/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("BestDatasets/crack500_det"),
        help="Output detection dataset directory.",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("BestDatasets/crack500_det/crack500_det.yaml"),
        help="Output dataset manifest YAML path.",
    )
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--class-name", default="crack")
    parser.add_argument("--class-index", type=int, default=0)
    parser.add_argument("--image-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--min-component-area", type=int, default=20)
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument("--min-box-side", type=int, default=2)
    parser.add_argument("--min-box-area", type=int, default=16)
    parser.add_argument(
        "--max-box-len-ratio",
        type=float,
        default=0.35,
        help="Split boxes whose long side exceeds this ratio of max(image_w, image_h).",
    )
    parser.add_argument("--max-aspect-ratio", type=float, default=12.0)
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.15,
        help="Split boxes whose area exceeds this fraction of image area.",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.15)
    parser.add_argument(
        "--write-empty-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write empty .txt files for images without boxes.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N images per split (0=all).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    src_root = Path(args.src_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    yaml_path = Path(args.yaml).expanduser().resolve()

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    summaries: list[ImageSummary] = []
    split_stats: dict[str, dict[str, int]] = {}

    include_test = False

    for split in list(args.splits):
        image_dir = src_root / split / "images"
        mask_dir = src_root / split / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        if split == "test":
            include_test = True

        images = _iter_images(image_dir)
        if int(args.limit) > 0:
            images = images[: int(args.limit)]
        if not images:
            continue

        for image_path in images:
            mask_path = _resolve_mask_for_image(mask_dir, image_path)
            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for {image_path} under {mask_dir}")

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            height, width = int(image.shape[0]), int(image.shape[1])

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not read mask: {mask_path}")
            if int(mask.shape[0]) != int(height) or int(mask.shape[1]) != int(width):
                mask = cv2.resize(mask, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)

            boxes, component_count = _mask_to_boxes(
                mask,
                threshold=int(args.threshold),
                min_component_area=int(args.min_component_area),
                image_width=int(width),
                image_height=int(height),
                pad=int(args.pad),
                min_box_side=int(args.min_box_side),
                min_box_area=int(args.min_box_area),
                max_box_len_ratio=float(args.max_box_len_ratio),
                max_aspect_ratio=float(args.max_aspect_ratio),
                max_area_ratio=float(args.max_area_ratio),
                tile_overlap=float(args.tile_overlap),
            )

            rel = image_path.relative_to(image_dir)
            out_image_path = out_dir / split / "images" / rel
            out_label_path = (out_dir / split / "labels" / rel).with_suffix(".txt")

            if not args.dry_run:
                _link_or_copy(image_path, out_image_path, str(args.image_mode))
                _ensure_parent(out_label_path)
                if boxes or bool(args.write_empty_labels):
                    lines = [
                        _bbox_to_yolo(
                            x0, y0, x1, y1, int(width), int(height), int(args.class_index)
                        )
                        for (x0, y0, x1, y1) in boxes
                    ]
                    out_label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            summaries.append(
                ImageSummary(
                    split=str(split),
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    width=int(width),
                    height=int(height),
                    component_count=int(component_count),
                    box_count=int(len(boxes)),
                )
            )
            stats = split_stats.setdefault(
                str(split),
                {"images": 0, "images_with_boxes": 0, "components": 0, "boxes": 0},
            )
            stats["images"] += 1
            stats["components"] += int(component_count)
            stats["boxes"] += int(len(boxes))
            if boxes:
                stats["images_with_boxes"] += 1

    if not summaries:
        raise RuntimeError(f"No images processed under {src_root} (splits={args.splits})")

    if not args.dry_run:
        _write_manifest_yaml(
            yaml_path,
            dataset_dir=out_dir,
            class_name=str(args.class_name),
            include_test=bool(include_test),
        )
        report = {
            "src_root": str(src_root),
            "out_dir": str(out_dir),
            "yaml": str(yaml_path),
            "splits": list(args.splits),
            "class_name": str(args.class_name),
            "class_index": int(args.class_index),
            "image_mode": str(args.image_mode),
            "threshold": int(args.threshold),
            "min_component_area": int(args.min_component_area),
            "pad": int(args.pad),
            "min_box_side": int(args.min_box_side),
            "min_box_area": int(args.min_box_area),
            "max_box_len_ratio": float(args.max_box_len_ratio),
            "max_aspect_ratio": float(args.max_aspect_ratio),
            "max_area_ratio": float(args.max_area_ratio),
            "tile_overlap": float(args.tile_overlap),
            "write_empty_labels": bool(args.write_empty_labels),
            "limit": int(args.limit),
            "dry_run": bool(args.dry_run),
            "stats": split_stats,
            "images": [asdict(row) for row in summaries],
        }
        report_path = out_dir / "conversion_report.json"
        _ensure_parent(report_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(yaml_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
