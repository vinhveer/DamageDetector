"""Generate radar (CRACK500) and bar (Volker) charts for extended metrics.

Output:
  extended_metrics_radar_crack500.pdf
  extended_metrics_volker_bars.pdf
"""
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

EVAL_ROOT = Path("/Users/nguyenquangvinh/Desktop/Lab/training_runs/v1/segmentation_post_kaggle/eval")
THESIS_FIG = Path("/Users/nguyenquangvinh/Desktop/Lab/DoAnTotNghiep_NguyenQuangVinh_64132989/figures")

MODELS = {
    "U-Net v1": ("unet_v1", "unet_v1_eval.sqlite3"),
    "U-Net v2": ("unet_v2_cldice_ema", "unet_v2_eval.sqlite3"),
    "SAM B0": ("sam_b0_zeroshot", "sam_b0_eval.sqlite3"),
    "SAM B1": ("sam_b1_lora_only", "sam_b1_eval.sqlite3"),
    "SAM B2": ("sam_b2_lora_hq", "sam_b2_eval.sqlite3"),
    "SAM B3": ("sam_b3_full", "sam_b3_eval.sqlite3"),
}

METRICS = ["best_dice", "best_iou", "best_skeleton_dice", "best_centerline_precision", "best_component_fragmentation"]
METRIC_LABELS = ["Dice", "IoU", "Skel. Dice", "CL-Precision", "CF (inv)"]

def load_metrics(dataset_label):
    """Load best metrics for all models on a dataset."""
    data = {}
    for name, (folder, db_file) in MODELS.items():
        db = EVAL_ROOT / folder / db_file
        if not db.exists():
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cur = conn.execute(
            "SELECT best_dice, best_iou, best_skeleton_dice, best_centerline_precision, best_component_fragmentation FROM datasets WHERE dataset_label = ?",
            (dataset_label,))
        row = cur.fetchone()
        conn.close()
        if row:
            data[name] = list(row)
    return data

def plot_radar(data, title, out_path):
    """Radar chart: normalize each metric to [0,1], invert CF."""
    models = list(data.keys())
    values = np.array([data[m] for m in models])
    # Invert CF (lower is better, 1.0 is ideal)
    values[:, 4] = 1.0 / np.maximum(values[:, 4], 0.01)
    # Normalize each metric to [0, max]
    maxv = values.max(axis=0)
    maxv[maxv == 0] = 1
    norm = values / maxv

    angles = np.linspace(0, 2 * np.pi, len(METRIC_LABELS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, m in enumerate(models):
        vals = norm[i].tolist() + [norm[i][0]]
        ax.plot(angles, vals, 'o-', linewidth=1.5, label=m, markersize=4)
        ax.fill(angles, vals, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, size=9)
    ax.set_title(title, size=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")

def plot_bars(data, title, out_path):
    """Grouped bar chart for Volker metrics."""
    models = list(data.keys())
    values = np.array([data[m] for m in models])
    n_metrics = len(METRIC_LABELS)
    n_models = len(models)
    x = np.arange(n_metrics)
    width = 0.12

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values[i], width, label=m, edgecolor='black', linewidth=0.3)
        # Highlight U-Net v2 bars
        if m == "U-Net v2":
            for b in bars:
                b.set_edgecolor('red')
                b.set_linewidth(1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")

if __name__ == "__main__":
    THESIS_FIG.mkdir(parents=True, exist_ok=True)

    # CRACK500 radar
    crack_data = load_metrics("crack500_test")
    if crack_data:
        plot_radar(crack_data, "CRACK500: Extended Metrics (normalized)", THESIS_FIG / "extended_metrics_radar_crack500.pdf")

    # Volker bars
    volker_data = load_metrics("crack_segmentation_dataset_volker_test")
    if volker_data:
        plot_bars(volker_data, "Volker (OOD): Extended Metrics", THESIS_FIG / "extended_metrics_volker_bars.pdf")
