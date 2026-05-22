"""SAM B0 zero-shot eval: GT boxes → SAM predictor → threshold sweep → SQLite.

Schema matches run_sam_eval_to_sqlite.py exactly so downstream tools work unchanged.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from device_utils import select_device_str, select_torch_device
from segmentation.sam.finetune.tiled_inference import (
    binary_mask_from_score_map,
    continuity_metrics,
    metric_per_case,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ─── SQLite schema (same as run_sam_eval_to_sqlite.py) ───────────────────────

def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, created_at_utc TEXT NOT NULL,
            output_root TEXT NOT NULL, model_dir TEXT NOT NULL,
            delta_ckpt TEXT NOT NULL, config_path TEXT,
            inference_config_path TEXT, sam_ckpt TEXT NOT NULL,
            device TEXT NOT NULL, eval_mode TEXT NOT NULL,
            vit_name TEXT NOT NULL, delta_type TEXT NOT NULL,
            rank INTEGER NOT NULL, decoder_type TEXT NOT NULL,
            centerline_head INTEGER NOT NULL, img_size INTEGER NOT NULL,
            tile_size INTEGER NOT NULL, tile_overlap INTEGER NOT NULL,
            tile_batch_size INTEGER NOT NULL, refine_batch_size INTEGER NOT NULL,
            thresholds_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS datasets (
            run_id TEXT NOT NULL, dataset_label TEXT NOT NULL,
            dataset_path TEXT NOT NULL, image_count INTEGER NOT NULL,
            best_threshold REAL NOT NULL, save_threshold REAL NOT NULL,
            best_precision REAL NOT NULL, best_recall REAL NOT NULL,
            best_dice REAL NOT NULL, best_iou REAL NOT NULL,
            best_skeleton_dice REAL NOT NULL, best_centerline_precision REAL NOT NULL,
            best_centerline_recall REAL NOT NULL, best_component_fragmentation REAL NOT NULL,
            PRIMARY KEY (run_id, dataset_label)
        );
        CREATE TABLE IF NOT EXISTS dataset_threshold_metrics (
            run_id TEXT NOT NULL, dataset_label TEXT NOT NULL,
            threshold REAL NOT NULL,
            mean_precision REAL NOT NULL, mean_recall REAL NOT NULL,
            mean_dice REAL NOT NULL, mean_iou REAL NOT NULL,
            mean_skeleton_dice REAL NOT NULL, mean_centerline_precision REAL NOT NULL,
            mean_centerline_recall REAL NOT NULL, mean_component_fragmentation REAL NOT NULL,
            PRIMARY KEY (run_id, dataset_label, threshold)
        );
        CREATE TABLE IF NOT EXISTS image_threshold_metrics (
            run_id TEXT NOT NULL, dataset_label TEXT NOT NULL,
            image_rel_path TEXT NOT NULL, image_path TEXT NOT NULL,
            image_name TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
            gt_positive_px INTEGER NOT NULL, threshold REAL NOT NULL,
            pred_positive_px INTEGER NOT NULL,
            precision REAL NOT NULL, recall REAL NOT NULL,
            dice REAL NOT NULL, iou REAL NOT NULL,
            skeleton_dice REAL NOT NULL, centerline_precision REAL NOT NULL,
            centerline_recall REAL NOT NULL, component_fragmentation REAL NOT NULL,
            PRIMARY KEY (run_id, dataset_label, image_rel_path, threshold)
        );
        CREATE INDEX IF NOT EXISTS idx_image_threshold_metrics_lookup
        ON image_threshold_metrics (run_id, dataset_label, threshold);
    """)
    conn.commit()


# ─── GT box extraction ────────────────────────────────────────────────────────

def extract_gt_boxes(gt_mask: np.ndarray, min_area: int = 16, padding: int = 0) -> list[dict]:
    binary = (gt_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    H, W = binary.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)
        boxes.append({"box": [float(x1), float(y1), float(x2), float(y2)],
                      "label": "crack", "score": 1.0})
    return boxes


# ─── Dataset utilities ────────────────────────────────────────────────────────

def _iter_images(dataset_root: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, mask_path) pairs, handling nested dirs like DeepCrack."""
    img_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    pairs = []
    for img_path in sorted(img_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img_path.relative_to(img_dir)
        # Try same extension first, then .png
        mask_path = mask_dir / rel
        if not mask_path.exists():
            mask_path = mask_dir / rel.with_suffix(".png")
        if not mask_path.exists():
            mask_path = mask_dir / rel.with_suffix(".jpg")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def _dataset_label(dataset_path: Path) -> str:
    return f"{dataset_path.parent.name}_{dataset_path.name}"


def _resolve_dataset_root(p: Path) -> Path:
    p = p.expanduser().resolve()
    if (p / "images").is_dir() and (p / "masks").is_dir():
        return p
    test = p / "test"
    if (test / "images").is_dir() and (test / "masks").is_dir():
        return test
    raise FileNotFoundError(f"No images/masks found: {p}")


# ─── SAM predictor loading ────────────────────────────────────────────────────

def _load_sam_predictor(sam_ckpt: str, model_type: str, device: str):
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


# ─── Per-image scoring ────────────────────────────────────────────────────────

def _predict_score_map(predictor, image_rgb: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Run SAM predictor on each box, merge logits → probability map."""
    H, W = image_rgb.shape[:2]
    if not boxes:
        return np.zeros((H, W), dtype=np.float32)

    predictor.set_image(image_rgb)
    merged_logits = np.full((H, W), -10.0, dtype=np.float32)  # very negative = background

    for entry in boxes:
        x1, y1, x2, y2 = entry["box"]
        input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, scores, logits = predictor.predict(
            box=input_box, multimask_output=True
        )
        # Pick best mask by score
        best_idx = int(np.argmax(scores))
        # logits shape: (n_masks, H_low, W_low) — use mask directly
        # Actually masks is (n_masks, H, W) binary; logits is low-res
        # We use the binary mask approach: convert chosen mask to prob 0/1
        # But for threshold sweep we need continuous values.
        # Use logits[best_idx] resized to full resolution
        low_res_logit = logits[best_idx]  # (256, 256) typically
        # Resize to full image
        logit_full = cv2.resize(
            low_res_logit, (W, H), interpolation=cv2.INTER_LINEAR
        )
        merged_logits = np.maximum(merged_logits, logit_full)

    # Convert logits to probability via sigmoid
    prob_map = 1.0 / (1.0 + np.exp(-merged_logits.clip(-20, 20)))
    return prob_map.astype(np.float32)


# ─── Main eval loop ──────────────────────────────────────────────────────────

def evaluate_dataset(
    conn: sqlite3.Connection,
    run_id: str,
    dataset_path: Path,
    predictor,
    thresholds: list[float],
    min_area: int,
    padding: int,
    limit: int | None,
) -> dict:
    dataset_label = _dataset_label(dataset_path)
    pairs = _iter_images(dataset_path)
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError(f"No image pairs in {dataset_path}")

    image_count = len(pairs)
    dataset_sums = {t: np.zeros(8, dtype=np.float64) for t in thresholds}
    image_rows = []

    for idx, (img_path, mask_path) in enumerate(pairs):
        image_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            gt_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        H, W = image_rgb.shape[:2]
        gt_positive_px = int(gt_binary.sum())

        boxes = extract_gt_boxes(gt_binary, min_area=min_area, padding=padding)
        score_map = _predict_score_map(predictor, image_rgb, boxes)

        rel_path = str(img_path.relative_to(dataset_path / "images"))

        for thr in thresholds:
            pred = binary_mask_from_score_map(score_map, float(thr))
            precision, recall, dice, iou = metric_per_case(pred, gt_binary)
            cont = continuity_metrics(pred, gt_binary)
            pred_px = int(pred.sum())
            image_rows.append((
                run_id, dataset_label, rel_path, str(img_path),
                img_path.name, W, H, gt_positive_px, float(thr), pred_px,
                float(precision), float(recall), float(dice), float(iou),
                float(cont["skeleton_dice"]), float(cont["centerline_precision"]),
                float(cont["centerline_recall"]), float(cont["component_fragmentation"]),
            ))
            dataset_sums[float(thr)] += np.array([
                precision, recall, dice, iou,
                cont["skeleton_dice"], cont["centerline_precision"],
                cont["centerline_recall"], cont["component_fragmentation"],
            ])

        print(f"[{dataset_label}] {idx+1}/{image_count} {img_path.name}", flush=True)

    # Write image rows
    conn.executemany("""
        INSERT OR REPLACE INTO image_threshold_metrics (
            run_id, dataset_label, image_rel_path, image_path, image_name,
            width, height, gt_positive_px, threshold, pred_positive_px,
            precision, recall, dice, iou, skeleton_dice,
            centerline_precision, centerline_recall, component_fragmentation
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, image_rows)

    # Dataset threshold metrics + find best
    best_dice, best_thr = -1.0, thresholds[0]
    ds_rows = []
    for thr in thresholds:
        means = dataset_sums[float(thr)] / max(1, image_count)
        ds_rows.append((run_id, dataset_label, float(thr),
                        *[float(m) for m in means]))
        if float(means[2]) > best_dice:
            best_dice = float(means[2])
            best_thr = float(thr)

    conn.executemany("""
        INSERT OR REPLACE INTO dataset_threshold_metrics (
            run_id, dataset_label, threshold,
            mean_precision, mean_recall, mean_dice, mean_iou,
            mean_skeleton_dice, mean_centerline_precision,
            mean_centerline_recall, mean_component_fragmentation
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, ds_rows)

    # Best row for datasets table
    best_means = dataset_sums[best_thr] / max(1, image_count)
    conn.execute("""
        INSERT OR REPLACE INTO datasets (
            run_id, dataset_label, dataset_path, image_count,
            best_threshold, save_threshold,
            best_precision, best_recall, best_dice, best_iou,
            best_skeleton_dice, best_centerline_precision,
            best_centerline_recall, best_component_fragmentation
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (run_id, dataset_label, str(dataset_path), image_count,
          best_thr, best_thr,
          *[float(m) for m in best_means]))

    conn.commit()

    # Write summary JSON
    summary = {
        "dataset_label": dataset_label, "image_count": image_count,
        "best_threshold": best_thr, "best_dice": float(best_means[2]),
        "best_iou": float(best_means[3]),
    }
    ds_dir = Path(conn.execute("SELECT output_root FROM runs WHERE run_id=?",
                               (run_id,)).fetchone()[0]) / dataset_label.split("_")[-1]
    # Use simpler naming
    out_dir = Path(conn.execute("SELECT output_root FROM runs WHERE run_id=?",
                                (run_id,)).fetchone()[0])
    summary_path = out_dir / f"{dataset_label}_metrics_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    # Per-image CSV
    csv_path = out_dir / dataset_label / "per_image_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("image_id,threshold,dice,iou,precision,recall,skeleton_dice,"
                "centerline_precision,centerline_recall,component_fragmentation\n")
        for row in image_rows:
            if abs(row[8] - best_thr) < 1e-6:
                stem = Path(row[4]).stem
                f.write(f"{stem},{row[8]:.4f},{row[12]:.6f},{row[13]:.6f},"
                        f"{row[10]:.6f},{row[11]:.6f},{row[14]:.6f},"
                        f"{row[15]:.6f},{row[16]:.6f},{row[17]:.6f}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="SAM B0 zero-shot eval with GT boxes")
    parser.add_argument("datasets", nargs="+", help="Dataset paths")
    parser.add_argument("--sam-ckpt", required=True)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sqlite-name", default="sam_b0_eval.sqlite3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt-mode", default="gt_box")
    parser.add_argument("--min-component-area", type=int, default=16)
    parser.add_argument("--box-padding", type=int, default=0)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device_str = select_device_str(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load predictor once
    print(f"Loading SAM {args.sam_model_type} on {device_str}...", flush=True)
    predictor = _load_sam_predictor(args.sam_ckpt, args.sam_model_type, device_str)

    # SQLite
    db_path = output_root / args.sqlite_name
    conn = sqlite3.connect(str(db_path))
    _ensure_schema(conn)

    run_id = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO runs (run_id, created_at_utc, output_root, model_dir,
            delta_ckpt, config_path, inference_config_path, sam_ckpt,
            device, eval_mode, vit_name, delta_type, rank, decoder_type,
            centerline_head, img_size, tile_size, tile_overlap,
            tile_batch_size, refine_batch_size, thresholds_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (run_id, datetime.now(timezone.utc).isoformat(), str(output_root),
          "zero_shot", "none", None, None, args.sam_ckpt,
          device_str, "zero_shot_gt_box", args.sam_model_type,
          "none", 0, "default", 0, 1024, 1024, 0, 1, 1,
          json.dumps(args.thresholds)))
    conn.commit()

    # Manifest
    manifest = {
        "run_id": run_id, "sam_ckpt": args.sam_ckpt,
        "model_type": args.sam_model_type, "prompt_mode": args.prompt_mode,
        "min_component_area": args.min_component_area,
        "box_padding": args.box_padding, "device": device_str,
        "thresholds": args.thresholds,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Eval each dataset
    for ds_path_str in args.datasets:
        ds_root = _resolve_dataset_root(Path(ds_path_str))
        summary = evaluate_dataset(
            conn=conn, run_id=run_id, dataset_path=ds_root,
            predictor=predictor, thresholds=args.thresholds,
            min_area=args.min_component_area, padding=args.box_padding,
            limit=args.limit,
        )
        print(f"  → {summary['dataset_label']}: Dice={summary['best_dice']:.4f} "
              f"@ thr={summary['best_threshold']:.2f}", flush=True)

    conn.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
