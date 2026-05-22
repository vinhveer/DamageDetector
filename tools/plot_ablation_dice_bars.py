"""Bar chart: Dice comparison across 6 models × 3 datasets (S8.3).

Reads main_segmentation_results.csv → grouped bar chart PDF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ["unet_v1", "unet_v2_cldice_ema", "sam_b0_zeroshot",
               "sam_b1_lora_only", "sam_b2_lora_hq", "sam_b3_full"]
MODEL_LABELS = ["U-Net v1", "U-Net v2", "SAM B0\n(zero-shot)", "SAM B1\n(LoRA)", "SAM B2\n(HQ)", "SAM B3\n(full)"]
DATASET_ORDER = ["crack500", "volker", "deepcrack"]
DATASET_LABELS = ["CRACK500", "Volker", "DeepCrack"]
COLORS = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1", "#76b7b2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="main_segmentation_results.csv")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument("--metric", default="best_dice", help="Column to plot")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Normalize dataset names
    ds_map = {"crack500_test": "crack500", "crack_segmentation_dataset_volker_test": "volker", "deepcrack_test": "deepcrack"}
    if "dataset_label" in df.columns:
        df["dataset"] = df["dataset_label"].map(ds_map).fillna(df.get("dataset", ""))
    if "model" not in df.columns and "model_tag" in df.columns:
        df["model"] = df["model_tag"]

    fig, ax = plt.subplots(figsize=(10, 5))
    n_datasets = len(DATASET_ORDER)
    n_models = len(MODEL_ORDER)
    bar_width = 0.12
    x = np.arange(n_datasets)

    for i, (model, label, color) in enumerate(zip(MODEL_ORDER, MODEL_LABELS, COLORS)):
        vals = []
        for ds in DATASET_ORDER:
            row = df[(df["model"] == model) & (df["dataset"] == ds)]
            vals.append(float(row[args.metric].iloc[0]) if len(row) > 0 else 0.0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=label, color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, fontsize=11)
    ax.set_ylabel("Dice Score", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title("Segmentation Dice — 6 Models × 3 Datasets", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
