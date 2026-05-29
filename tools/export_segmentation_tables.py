"""S6: Trích 3 bảng CSV từ 6 SQLite eval cho R1 backfill.

Outputs (vào $TABLES_ROOT):
  - main_segmentation_results.csv   — long-form: (model, dataset) × metric cốt lõi
  - sam_ablation_results.csv        — 4 SAM (B0/B1/B2/B3) × 3 dataset
  - unet_v1_vs_v2_results.csv       — pivot v1 vs v2 với Δ
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

# Mapping model_tag → human label (dùng trong báo cáo)
MODELS = [
    ("unet_v1",            "U-Net v1",            "unet_v1/unet_v1_eval.sqlite3"),
    ("unet_v2_cldice_ema", "U-Net v2 (clDice+EMA)", "unet_v2_cldice_ema/unet_v2_eval.sqlite3"),
    ("sam_b0_zeroshot",    "SAM B0 (zero-shot, GT box)", "sam_b0_zeroshot/sam_b0_eval.sqlite3"),
    ("sam_b1_lora_only",   "SAM B1 (LoRA r4)",    "sam_b1_lora_only/sam_b1_eval.sqlite3"),
    ("sam_b2_lora_hq",     "SAM B2 (LoRA+HQ r4)", "sam_b2_lora_hq/sam_b2_eval.sqlite3"),
    ("sam_b3_full",        "SAM B3 (LoRA r16+HQ)", "sam_b3_full/sam_b3_eval.sqlite3"),
]

# Normalize dataset_label → short label cho bảng
DATASET_MAP = {
    "crack500_test":                          "CRACK500",
    "crack_segmentation_dataset_volker_test": "Volker",
    "deepcrack_test":                         "DeepCrack",
}

METRIC_COLS = [
    "best_threshold", "best_dice", "best_iou", "best_precision", "best_recall",
    "best_skeleton_dice", "best_centerline_precision", "best_centerline_recall",
    "best_component_fragmentation",
]


def fetch_all_rows(eval_root: Path) -> list[dict]:
    rows = []
    for tag, label, rel_db in MODELS:
        db = eval_root / rel_db
        if not db.exists():
            print(f"  [WARN] missing: {db}")
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cols = "image_count, " + ", ".join(METRIC_COLS)
        cur = conn.execute(f"SELECT dataset_label, {cols} FROM datasets ORDER BY dataset_label")
        for r in cur.fetchall():
            ds_label = r[0]
            short_ds = DATASET_MAP.get(ds_label, ds_label)
            rows.append({
                "model_tag":   tag,
                "model":       label,
                "dataset":     short_ds,
                "n_images":    r[1],
                "threshold":   r[2],
                "dice":        r[3],
                "iou":         r[4],
                "precision":   r[5],
                "recall":      r[6],
                "skel_dice":   r[7],
                "cl_precision": r[8],
                "cl_recall":   r[9],
                "frag":        r[10],
            })
        conn.close()
    return rows


def write_main(rows: list[dict], path: Path) -> None:
    cols = ["model", "dataset", "n_images", "threshold",
            "dice", "iou", "precision", "recall",
            "skel_dice", "cl_precision", "cl_recall", "frag"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        # Dataset order: CRACK500 → Volker → DeepCrack
        # Model order: theo MODELS
        order = {DATASET_MAP[k]: i for i, k in enumerate(DATASET_MAP)}
        rows_sorted = sorted(rows, key=lambda r: (order.get(r["dataset"], 99),
                                                  [m[0] for m in MODELS].index(r["model_tag"])))
        for r in rows_sorted:
            w.writerow([r["model"], r["dataset"], r["n_images"],
                        f"{r['threshold']:.2f}", f"{r['dice']:.4f}",
                        f"{r['iou']:.4f}", f"{r['precision']:.4f}",
                        f"{r['recall']:.4f}", f"{r['skel_dice']:.4f}",
                        f"{r['cl_precision']:.4f}", f"{r['cl_recall']:.4f}",
                        f"{r['frag']:.4f}"])
    print(f"  wrote {path.name} — {len(rows_sorted)} rows")


def write_sam_ablation(rows: list[dict], path: Path) -> None:
    """Chỉ 4 SAM model × 3 dataset, wide-form thân thiện hơn cho LaTeX."""
    sam_tags = {"sam_b0_zeroshot", "sam_b1_lora_only", "sam_b2_lora_hq", "sam_b3_full"}
    sam_rows = [r for r in rows if r["model_tag"] in sam_tags]
    # Build wide: index by (model, dataset)
    by_key = {(r["model_tag"], r["dataset"]): r for r in sam_rows}
    cols = ["model", "decoder", "lora_rank", "centerline_aux",
            "ds", "thr", "dice", "iou", "precision", "recall", "skel_dice"]
    # Ablation metadata per model
    META = {
        "sam_b0_zeroshot":  ("baseline", "—", "no (frozen)"),
        "sam_b1_lora_only": ("baseline", "4", "no"),
        "sam_b2_lora_hq":   ("hq",       "4", "yes"),
        "sam_b3_full":      ("hq",       "16", "yes"),
    }
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for tag, label, _ in MODELS:
            if tag not in sam_tags:
                continue
            decoder, rank, cl_aux = META[tag]
            for ds in ("CRACK500", "Volker", "DeepCrack"):
                r = by_key.get((tag, ds))
                if r is None:
                    continue
                w.writerow([label, decoder, rank, cl_aux, ds,
                            f"{r['threshold']:.2f}", f"{r['dice']:.4f}",
                            f"{r['iou']:.4f}", f"{r['precision']:.4f}",
                            f"{r['recall']:.4f}", f"{r['skel_dice']:.4f}"])
    print(f"  wrote {path.name}")


def write_v1_vs_v2(rows: list[dict], path: Path) -> None:
    """Pivot v1 vs v2 với delta — bảng cho mục R1.3."""
    by_key = {(r["model_tag"], r["dataset"]): r for r in rows
              if r["model_tag"] in {"unet_v1", "unet_v2_cldice_ema"}}
    cols = ["dataset", "metric", "v1", "v2", "delta_abs", "delta_pct"]
    metrics = [("dice", "Dice"), ("iou", "IoU"),
               ("precision", "Precision"), ("recall", "Recall"),
               ("skel_dice", "Skel. Dice"),
               ("cl_precision", "CL Precision"), ("cl_recall", "CL Recall"),
               ("frag", "Fragmentation")]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ds in ("CRACK500", "Volker", "DeepCrack"):
            v1 = by_key.get(("unet_v1", ds))
            v2 = by_key.get(("unet_v2_cldice_ema", ds))
            if not v1 or not v2:
                continue
            for key, name in metrics:
                delta = v2[key] - v1[key]
                pct = delta / v1[key] * 100 if v1[key] else 0.0
                w.writerow([ds, name, f"{v1[key]:.4f}", f"{v2[key]:.4f}",
                            f"{delta:+.4f}", f"{pct:+.2f}%"])
    print(f"  wrote {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root",
                    default="/Users/nguyenquangvinh/Desktop/Lab/training_runs/v1/segmentation_post_kaggle/eval")
    ap.add_argument("--tables-root",
                    default="/Users/nguyenquangvinh/Desktop/Lab/training_runs/v1/segmentation_post_kaggle/tables")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    tables_root = Path(args.tables_root)
    tables_root.mkdir(parents=True, exist_ok=True)

    rows = fetch_all_rows(eval_root)
    print(f"Loaded {len(rows)} rows from 6 SQLite (expected 18 = 6 model × 3 dataset)")

    write_main(rows, tables_root / "main_segmentation_results.csv")
    write_sam_ablation(rows, tables_root / "sam_ablation_results.csv")
    write_v1_vs_v2(rows, tables_root / "unet_v1_vs_v2_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
