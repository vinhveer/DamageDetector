from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MaskChange:
    split: str
    mask_path: str
    positive_before: int
    positive_after: int
    changed_pixels: int


def _iter_mask_paths(dataset_root: Path, splits: list[str]) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for split in splits:
        mask_dir = dataset_root / split / "masks"
        if not mask_dir.is_dir():
            continue
        for path in sorted(mask_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append((split, path))
    return rows


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if int(min_area) <= 1:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for idx in range(1, int(num_labels)):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_area):
            cleaned[labels == idx] = 1
    return cleaned


def _filter_external_contours(mask: np.ndarray, min_area: int) -> np.ndarray:
    if int(min_area) <= 0:
        return mask
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(binary)
    for contour in contours:
        if float(cv2.contourArea(contour)) >= float(min_area):
            cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)
    return (cleaned > 0).astype(np.uint8)


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return binary
    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()
    while True:
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = eroded
        if int(cv2.countNonZero(work)) == 0:
            break
    return (skeleton > 0).astype(np.uint8)


def _clean_mask(
    mask: np.ndarray,
    *,
    mode: str,
    threshold: int,
    blur_sigma: float,
    median_ksize: int,
    erode_kernel: int,
    open_kernel: int,
    close_kernel: int,
    contour_min_area: int,
    min_area: int,
    rebuild_kernel: int,
    rebuild_iterations: int,
) -> np.ndarray:
    binary = (np.asarray(mask, dtype=np.uint8) > int(threshold)).astype(np.uint8)

    normalized_mode = str(mode or "standard").strip().lower()
    if normalized_mode == "skeleton_rebuild":
        binary = _skeletonize_mask(binary)
        if int(rebuild_kernel) >= 2 and int(rebuild_iterations) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(rebuild_kernel), int(rebuild_kernel)))
            binary = cv2.dilate(binary, kernel, iterations=max(1, int(rebuild_iterations)))
        binary = _remove_small_components(binary, min_area=max(1, int(min_area)))
        return (binary > 0).astype(np.uint8) * 255

    if float(blur_sigma) > 0.0:
        blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=float(blur_sigma))
        binary = (blurred >= 0.5).astype(np.uint8)

    if int(median_ksize) >= 3 and int(median_ksize) % 2 == 1:
        binary = (cv2.medianBlur(binary * 255, int(median_ksize)) > 127).astype(np.uint8)

    if int(erode_kernel) >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(erode_kernel), int(erode_kernel)))
        binary = cv2.erode(binary, kernel, iterations=1)

    if int(open_kernel) >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_kernel), int(open_kernel)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    if int(close_kernel) >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_kernel), int(close_kernel)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    binary = _filter_external_contours(binary, min_area=int(contour_min_area))
    binary = _remove_small_components(binary, min_area=int(min_area))
    return (binary > 0).astype(np.uint8) * 255


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean binary masks for a split dataset in place.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Dataset root containing train/val/test")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--mode", default="standard", choices=["standard", "skeleton_rebuild"])
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--blur-sigma", type=float, default=0.8)
    parser.add_argument("--median-ksize", type=int, default=3)
    parser.add_argument("--erode-kernel", type=int, default=0, help="Use 0 to disable edge shaving erosion")
    parser.add_argument("--open-kernel", type=int, default=0, help="Use 0 to disable opening")
    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument("--contour-min-area", type=int, default=0, help="Keep only external contours with area >= this value")
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--rebuild-kernel", type=int, default=3, help="Dilate kernel for skeleton_rebuild mode")
    parser.add_argument("--rebuild-iterations", type=int, default=1, help="Dilate iterations for skeleton_rebuild mode")
    parser.add_argument("--manifest-name", default="mask_cleanup_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    mask_rows = _iter_mask_paths(dataset_root, list(args.splits))
    if not mask_rows:
        raise FileNotFoundError(f"No mask files found under {dataset_root}")

    changes: list[MaskChange] = []
    split_summary: dict[str, dict[str, int]] = {}

    for split, mask_path in mask_rows:
        image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")
        cleaned = _clean_mask(
            image,
            mode=str(args.mode),
            threshold=int(args.threshold),
            blur_sigma=float(args.blur_sigma),
            median_ksize=int(args.median_ksize),
            erode_kernel=int(args.erode_kernel),
            open_kernel=int(args.open_kernel),
            close_kernel=int(args.close_kernel),
            contour_min_area=int(args.contour_min_area),
            min_area=int(args.min_area),
            rebuild_kernel=int(args.rebuild_kernel),
            rebuild_iterations=int(args.rebuild_iterations),
        )
        before = (image > int(args.threshold)).astype(np.uint8)
        after = (cleaned > 127).astype(np.uint8)
        changed_pixels = int(np.count_nonzero(before != after))
        if changed_pixels > 0 and not args.dry_run:
            ok = cv2.imwrite(str(mask_path), cleaned)
            if not ok:
                raise RuntimeError(f"Could not write cleaned mask: {mask_path}")

        changes.append(
            MaskChange(
                split=split,
                mask_path=str(mask_path),
                positive_before=int(before.sum()),
                positive_after=int(after.sum()),
                changed_pixels=changed_pixels,
            )
        )
        stats = split_summary.setdefault(split, {"file_count": 0, "changed_files": 0, "changed_pixels": 0})
        stats["file_count"] += 1
        stats["changed_pixels"] += changed_pixels
        if changed_pixels > 0:
            stats["changed_files"] += 1

    manifest = {
        "dataset_root": str(dataset_root),
        "splits": list(args.splits),
        "mode": str(args.mode),
        "threshold": int(args.threshold),
        "blur_sigma": float(args.blur_sigma),
        "median_ksize": int(args.median_ksize),
        "erode_kernel": int(args.erode_kernel),
        "open_kernel": int(args.open_kernel),
        "close_kernel": int(args.close_kernel),
        "contour_min_area": int(args.contour_min_area),
        "min_area": int(args.min_area),
        "rebuild_kernel": int(args.rebuild_kernel),
        "rebuild_iterations": int(args.rebuild_iterations),
        "dry_run": bool(args.dry_run),
        "summary": split_summary,
        "changes": [asdict(row) for row in changes],
    }
    manifest_path = dataset_root / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
