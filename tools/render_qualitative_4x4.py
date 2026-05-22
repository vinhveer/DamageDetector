"""Render qualitative 4×4 grid: Input | GT | U-Net v1 (QW) | SAM B1.

Selects 4 cases from CRACK500 test:
  Row A: SAM B1 wins v1 most (argmax IoU_B1 - IoU_v1)
  Row B: v1 wins B1 most (argmax IoU_v1 - IoU_B1)
  Row C: Both fail topology (argmin skel_dice_v1 + skel_dice_b1)
  Row D: Near tie (argmin |IoU_v1 - IoU_B1|)

Output: qualitative_4x4.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--dataset-root", required=True, help="BestDatasets/crack500/test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    ds_root = Path(args.dataset_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load per-image metrics
    v1_csv = eval_root / "unet_v1" / "crack500_test" / "per_image_metrics.csv"
    b1_csv = eval_root / "sam_b1_lora_only" / "crack500_test" / "per_image_metrics.csv"
    df_v1 = pd.read_csv(v1_csv)
    df_b1 = pd.read_csv(b1_csv)

    merged = df_v1.merge(df_b1, on="image_id", suffixes=("_v1", "_b1"))
    merged["delta_iou"] = merged["iou_b1"] - merged["iou_v1"]
    merged["sum_skel"] = merged["skeleton_dice_v1"] + merged["skeleton_dice_b1"]

    # Select 4 cases (filter for meaningful comparisons: both IoU > 0.1)
    meaningful = merged[(merged["iou_v1"] > 0.1) | (merged["iou_b1"] > 0.1)]
    cases = {}
    # Row A: SAM B1 wins most
    cases["A"] = meaningful.loc[meaningful["delta_iou"].idxmax(), "image_id"]
    # Row B: v1 wins most
    cases["B"] = meaningful.loc[meaningful["delta_iou"].idxmin(), "image_id"]
    # Row C: Both fail topology (lowest sum skeleton dice among images with some IoU)
    both_active = merged[(merged["iou_v1"] > 0.05) & (merged["iou_b1"] > 0.05)]
    remaining = both_active[~both_active["image_id"].isin(list(cases.values()))]
    if len(remaining) > 0:
        cases["C"] = remaining.loc[remaining["sum_skel"].idxmin(), "image_id"]
    else:
        remaining = merged[~merged["image_id"].isin(list(cases.values()))]
        cases["C"] = remaining.loc[remaining["sum_skel"].idxmin(), "image_id"]
    # Row D: Near tie (both have decent IoU)
    decent = merged[(merged["iou_v1"] > 0.3) & (merged["iou_b1"] > 0.3)]
    remaining = decent[~decent["image_id"].isin(list(cases.values()))]
    if len(remaining) > 0:
        cases["D"] = remaining.loc[remaining["delta_iou"].abs().idxmin(), "image_id"]
    else:
        remaining = merged[~merged["image_id"].isin(list(cases.values()))]
        cases["D"] = remaining.loc[remaining["delta_iou"].abs().idxmin(), "image_id"]

    # Load images and masks
    v1_mask_dir = eval_root / "unet_v1" / "crack500_test" / "masks"
    b1_mask_dir = eval_root / "sam_b1_lora_only" / "crack500_test" / "masks"

    rows_imgs = []
    for label, img_id in cases.items():
        # Find input image
        img_path = _find_image(ds_root / "images", img_id)
        gt_path = _find_image(ds_root / "masks", img_id)
        v1_mask_path = v1_mask_dir / f"{img_id}__pred.png"
        b1_mask_path = b1_mask_dir / f"{img_id}__pred.png"

        if not all(p.exists() for p in [img_path, gt_path, v1_mask_path, b1_mask_path]):
            print(f"WARN: Missing files for {img_id}, skipping")
            continue

        img = cv2.imread(str(img_path))
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        v1_mask = cv2.imread(str(v1_mask_path), cv2.IMREAD_GRAYSCALE)
        b1_mask = cv2.imread(str(b1_mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize all to same size for grid
        h_target = 256
        w_target = 256
        img = cv2.resize(img, (w_target, h_target))
        gt = cv2.resize(gt, (w_target, h_target))
        v1_mask = cv2.resize(v1_mask, (w_target, h_target))
        b1_mask = cv2.resize(b1_mask, (w_target, h_target))

        # Convert masks to colored overlays
        gt_vis = _mask_to_color(gt, (0, 255, 0))  # green
        v1_vis = _mask_to_color(v1_mask, (255, 100, 0))  # orange
        b1_vis = _mask_to_color(b1_mask, (0, 100, 255))  # blue

        row = np.hstack([img, gt_vis, v1_vis, b1_vis])
        rows_imgs.append(row)

        # Get metrics for annotation
        row_data = merged[merged["image_id"] == img_id].iloc[0]
        print(f"  {label}: {img_id} | v1 IoU={row_data['iou_v1']:.3f} | B1 IoU={row_data['iou_b1']:.3f}")

    if not rows_imgs:
        print("ERROR: No valid rows")
        return

    # Stack rows with separator
    sep = np.ones((4, rows_imgs[0].shape[1], 3), dtype=np.uint8) * 200
    grid_parts = []
    for i, row in enumerate(rows_imgs):
        if i > 0:
            grid_parts.append(sep)
        grid_parts.append(row)
    grid = np.vstack(grid_parts)

    cv2.imwrite(str(out_path), grid)
    print(f"Saved: {out_path}")


def _find_image(directory: Path, image_id: str) -> Path:
    """Find image file matching image_id."""
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
        p = directory / f"{image_id}{ext}"
        if p.exists():
            return p
    # Try with subdirectories
    for p in directory.rglob(f"{image_id}.*"):
        return p
    return directory / f"{image_id}.png"


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    """Resize image to target height, maintaining aspect ratio."""
    if len(img.shape) == 2:
        oh, ow = img.shape
    else:
        oh, ow = img.shape[:2]
    w = int(ow * h / oh)
    interp = cv2.INTER_AREA if h < oh else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def _mask_to_color(mask: np.ndarray, color: tuple) -> np.ndarray:
    """Convert binary mask to colored visualization on dark background."""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask > 127] = color
    vis[mask <= 127] = (30, 30, 30)
    return vis


if __name__ == "__main__":
    main()
