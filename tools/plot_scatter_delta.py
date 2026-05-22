"""Generate scatter delta plots: IoU per image, model A vs model B.

Reads per_image_metrics.csv from eval root, produces 1 PDF per dataset.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_LABEL_MAP = {
    "crack500": "crack500_test",
    "volker": "crack_segmentation_dataset_volker_test",
    "deepcrack87": "deepcrack_test",
}
# Compare SAM B1 (best in-domain) vs U-Net v1 (best OOD)
MODEL_A = "sam_b1_lora_only"
MODEL_B = "unet_v1"
LABEL_A = "SAM B1"
LABEL_B = "U-Net v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-a", default=MODEL_A)
    parser.add_argument("--model-b", default=MODEL_B)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_short, ds_label in DATASET_LABEL_MAP.items():
        csv_a = eval_root / args.model_a / ds_label / "per_image_metrics.csv"
        csv_b = eval_root / args.model_b / ds_label / "per_image_metrics.csv"
        if not csv_a.exists() or not csv_b.exists():
            print(f"SKIP {ds_short}: missing CSV")
            continue

        df_a = pd.read_csv(csv_a)
        df_b = pd.read_csv(csv_b)
        merged = df_a.merge(df_b, on="image_id", suffixes=("_a", "_b"))

        x = merged["iou_b"].to_numpy()
        y = merged["iou_a"].to_numpy()
        delta = y - x

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)

        # Scatter
        ax1.scatter(x, y, s=12, alpha=0.6, c="#475569", edgecolors="none")
        ax1.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
        ax1.set_xlabel(f"{LABEL_B} IoU", fontsize=10)
        ax1.set_ylabel(f"{LABEL_A} IoU", fontsize=10)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect("equal")
        n_above = (delta > 0).sum()
        n_below = (delta < 0).sum()
        ax1.set_title(f"{ds_short}: {LABEL_A} vs {LABEL_B}\n"
                      f"({LABEL_A} wins {n_above}, {LABEL_B} wins {n_below})", fontsize=10)
        ax1.grid(alpha=0.2)

        # Delta histogram
        ax2.hist(delta, bins=40, color="#7c3aed", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax2.axvline(0, color="#64748b", ls="--", lw=1)
        ax2.axvline(np.median(delta), color="#dc2626", ls="-", lw=1.5,
                    label=f"median Δ={np.median(delta):.3f}")
        ax2.set_xlabel(f"ΔIoU ({LABEL_A} − {LABEL_B})", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title(f"Distribution of per-image ΔIoU", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.2)

        plt.tight_layout()
        out_path = out_dir / f"per_image_{ds_short}_iou_scatter_delta.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    main()
