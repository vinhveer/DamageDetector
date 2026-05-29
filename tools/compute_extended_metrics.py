"""Compute extended metrics: BF@2px, HD95, Betti error, CL-F@2px, F2-score.

Reads predicted masks from $EVAL_ROOT/<model>/<dataset>/masks/ and GT from data/datasets/.
Outputs image-level CSV and dataset summary CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import skeletonize

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DATASET_MAP = {
    "crack500": "crack500/test",
    "volker": "crack_segmentation_dataset_volker/test",
    "deepcrack": "deepcrack/test",
}
DATASET_LABEL_MAP = {
    "crack500": "crack500_test",
    "volker": "crack_segmentation_dataset_volker_test",
    "deepcrack": "deepcrack_test",
}


# ─── Metric implementations ──────────────────────────────────────────────────

def boundary_f_score(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    """BF-score: F1 of boundary pixels within tolerance."""
    pred_boundary = _extract_boundary(pred)
    gt_boundary = _extract_boundary(gt)
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0
    gt_dist = distance_transform_edt(~gt_boundary.astype(bool))
    pred_dist = distance_transform_edt(~pred_boundary.astype(bool))
    precision = (gt_dist[pred_boundary > 0] <= tol).mean()
    recall = (pred_dist[gt_boundary > 0] <= tol).mean()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(mask, kernel)
    return (mask - eroded).astype(np.uint8)


def hausdorff_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile Hausdorff distance."""
    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")
    # Compute distances using distance transform for efficiency
    gt_dist = distance_transform_edt(gt == 0)
    pred_dist = distance_transform_edt(pred == 0)
    d_pred_to_gt = gt_dist[pred > 0]
    d_gt_to_pred = pred_dist[gt > 0]
    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def betti_error(pred: np.ndarray, gt: np.ndarray) -> int:
    """Absolute difference in Betti-0 (number of connected components)."""
    n_pred, *_ = cv2.connectedComponents(pred, connectivity=8)
    n_gt, *_ = cv2.connectedComponents(gt, connectivity=8)
    return abs((n_pred - 1) - (n_gt - 1))  # subtract background


def centerline_f_score(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    """F1 of centerline (skeleton) pixels within tolerance."""
    pred_skel = skeletonize(pred > 0).astype(np.uint8)
    gt_skel = skeletonize(gt > 0).astype(np.uint8)
    if pred_skel.sum() == 0 and gt_skel.sum() == 0:
        return 1.0
    if pred_skel.sum() == 0 or gt_skel.sum() == 0:
        return 0.0
    gt_dist = distance_transform_edt(gt_skel == 0)
    pred_dist = distance_transform_edt(pred_skel == 0)
    precision = (gt_dist[pred_skel > 0] <= tol).mean()
    recall = (pred_dist[gt_skel > 0] <= tol).mean()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def f_beta_score(pred: np.ndarray, gt: np.ndarray, beta: float = 2.0) -> float:
    """F-beta score (pixel-level)."""
    tp = float(np.logical_and(pred > 0, gt > 0).sum())
    fp = float(np.logical_and(pred > 0, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt > 0).sum())
    if tp == 0:
        return 1.0 if (fp == 0 and fn == 0) else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    b2 = beta * beta
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


# ─── Dataset iteration ────────────────────────────────────────────────────────

def _iter_gt_masks(dataset_root: Path) -> list[tuple[Path, str]]:
    mask_dir = dataset_root / "masks"
    pairs = []
    for p in sorted(mask_dir.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(mask_dir)
            pairs.append((p, str(rel.with_suffix(""))))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", required=True, help="EVAL_ROOT")
    parser.add_argument("--gt-root", required=True, help="data/datasets/")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", default=["bf", "hd95", "betti", "cl_f", "f2"])
    parser.add_argument("--tolerance-px", type=int, default=2)
    parser.add_argument("--output-image-csv", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    tol = args.tolerance_px
    rows = []

    for model in args.models:
        for dataset in args.datasets:
            print(f"[{model}] [{dataset}]", flush=True)
            ds_label = DATASET_LABEL_MAP[dataset]
            mask_dir = pred_root / model / ds_label / "masks"
            if not mask_dir.exists():
                print(f"  SKIP: {mask_dir} not found")
                continue

            gt_pairs = _iter_gt_masks(gt_root / DATASET_MAP[dataset])
            if args.limit:
                gt_pairs = gt_pairs[:args.limit]

            for gt_path, image_id in gt_pairs:
                # Find corresponding pred mask
                pred_name = f"{image_id.replace('/', '__')}__pred.png"
                pred_path = mask_dir / pred_name
                if not pred_path.exists():
                    continue

                gt = (cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
                pred = (cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

                row = {"model": model, "dataset": dataset, "image_id": image_id}
                if "bf" in args.metrics:
                    row["bf_2px"] = boundary_f_score(pred, gt, tol)
                if "hd95" in args.metrics:
                    row["hd95"] = hausdorff_95(pred, gt)
                if "betti" in args.metrics:
                    row["betti_err"] = betti_error(pred, gt)
                if "cl_f" in args.metrics:
                    row["cl_f_2px"] = centerline_f_score(pred, gt, tol)
                if "f2" in args.metrics:
                    row["f2"] = f_beta_score(pred, gt, beta=2.0)
                rows.append(row)

            print(f"  Computed {len([r for r in rows if r['model']==model and r['dataset']==dataset])} images")

    # Save image-level
    df = pd.DataFrame(rows)
    Path(args.output_image_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_image_csv, index=False)
    print(f"Image-level CSV: {args.output_image_csv} ({len(df)} rows)")

    # Summary
    metric_cols = [c for c in df.columns if c not in ("model", "dataset", "image_id")]
    summary_rows = []
    for (model, dataset), grp in df.groupby(["model", "dataset"]):
        row = {"model": model, "dataset": dataset, "n_images": len(grp)}
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else float("nan")
            row[f"{col}_std"] = vals.std() if len(vals) > 0 else float("nan")
            row[f"{col}_median"] = vals.median() if len(vals) > 0 else float("nan")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    Path(args.output_summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_summary_csv, index=False)
    print(f"Summary CSV: {args.output_summary_csv} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
