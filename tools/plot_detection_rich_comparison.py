"""Create rich comparison plots for YOLO vs StableDINO detection results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_science_style, save_report_figure

apply_science_style()


MODEL_LABELS = {
    "yolo_crack_500": "YOLO26x",
    "stable_dino_crack_500": "StableDINO",
}

COLORS = {
    "YOLO26x": "#2f6fbb",
    "StableDINO": "#d1495b",
}


def label_models(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model"] = df["detector"].map(MODEL_LABELS).fillna(df["detector"])
    return df


def savefig(path: Path) -> None:
    plt.tight_layout()
    save_report_figure(plt.gcf(), path)


def plot_confidence_curves(sweep: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True)
    flat_axes = axes.ravel()
    for ax, metric in zip(flat_axes, metrics):
        for model, group in sweep.groupby("model"):
            ax.plot(
                group["conf"],
                group[metric],
                marker="o",
                linewidth=2,
                label=model,
                color=COLORS.get(model),
            )
        ax.set_title(metric.upper() if metric == "f1" else metric.title())
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel(metric.title())
        ax.grid(True, alpha=0.25)
    flat_axes[0].legend(frameon=False)
    flat_axes[-1].axis("off")
    fig.suptitle("Precision, Recall và F1 theo ngưỡng confidence", y=1.03)
    savefig(out_dir / "confidence_precision_recall_f1.svg")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for ax, metric in zip(axes, ["precision", "recall"]):
        for model, group in sweep.groupby("model"):
            ax.plot(
                group["conf"],
                group[metric],
                marker="o",
                linewidth=2,
                label=model,
                color=COLORS.get(model),
            )
        ax.set_title(metric.title())
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel(metric.title())
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Precision và Recall theo ngưỡng confidence", y=1.03)
    savefig(out_dir / "confidence_precision_recall.svg")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model, group in sweep.groupby("model"):
        ax.plot(
            group["conf"],
            group["f1"],
            marker="o",
            linewidth=2,
            label=model,
            color=COLORS.get(model),
        )
    ax.set_title("F1 theo ngưỡng confidence")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    savefig(out_dir / "confidence_f1.svg")


def plot_map_curves(sweep: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["map_50", "map_50_95"],
        ["mAP@0.5", "mAP@0.5:0.95"],
    ):
        for model, group in sweep.groupby("model"):
            ax.plot(
                group["conf"],
                group[metric],
                marker="o",
                linewidth=2,
                label=model,
                color=COLORS.get(model),
            )
        ax.set_title(title)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("mAP theo ngưỡng confidence", y=1.03)
    savefig(out_dir / "confidence_map_curves.svg")


def plot_error_counts(metrics: pd.DataFrame, out_dir: Path) -> None:
    cols = ["tp", "fp", "fn"]
    x = np.arange(len(metrics))
    width = 0.23
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, col in enumerate(cols):
        values = metrics[col].to_numpy()
        bars = ax.bar(x + (i - 1) * width, values, width, label=col.upper())
        ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["model"])
    ax.set_ylabel("Số lượng box")
    ax.set_title("TP, FP và FN tại confidence tốt nhất theo F1")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    savefig(out_dir / "detection_tp_fp_fn_counts.svg")


def plot_speed_quality(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for _, row in metrics.iterrows():
        model = row["model"]
        ax.scatter(
            row["avg_time_ms"],
            row["f1"],
            s=420,
            color=COLORS.get(model),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.annotate(
            f"{model}\n{row['fps']:.2f} FPS",
            (row["avg_time_ms"], row["f1"]),
            xytext=(-46, -30) if model == "StableDINO" else (8, 8),
            textcoords="offset points",
            fontsize=10,
            ha="right" if model == "StableDINO" else "left",
            va="top" if model == "StableDINO" else "bottom",
        )
    ax.set_xlabel("Thời gian suy luận trung bình (ms/ảnh)")
    ax.set_ylabel("F1-score")
    ax.set_title("Trade-off chất lượng phát hiện và tốc độ")
    ax.grid(True, alpha=0.25)
    savefig(out_dir / "speed_quality_tradeoff.svg")


def plot_per_image_boxplots(image_metrics: pd.DataFrame, out_dir: Path) -> None:
    models = list(image_metrics["model"].drop_duplicates())
    data_f1 = [image_metrics.loc[image_metrics["model"] == m, "f1"] for m in models]
    data_iou = [image_metrics.loc[image_metrics["model"] == m, "avg_best_iou"] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, data, title, ylabel in zip(
        axes,
        [data_f1, data_iou],
        ["F1 theo ảnh", "IoU tốt nhất trung bình theo ảnh"],
        ["F1", "Avg best IoU"],
    ):
        bp = ax.boxplot(data, tick_labels=models, patch_artist=True, showfliers=False)
        for patch, model in zip(bp["boxes"], models):
            patch.set_facecolor(COLORS.get(model, "#999999"))
            patch.set_alpha(0.65)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Phân bố chất lượng phát hiện theo từng ảnh", y=1.03)
    savefig(out_dir / "per_image_f1_iou_boxplots.svg")


def plot_per_image_scatter(image_metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    wide = image_metrics.pivot(index="image_id", columns="model", values="f1").reset_index()
    if {"YOLO26x", "StableDINO"}.issubset(wide.columns):
        wide["delta_stabledino_minus_yolo"] = wide["StableDINO"] - wide["YOLO26x"]
    else:
        wide["delta_stabledino_minus_yolo"] = np.nan
    wide.to_csv(out_dir / "per_image_f1_delta.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(wide["YOLO26x"], wide["StableDINO"], s=22, alpha=0.55, color="#5f6c7b")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("YOLO26x F1 theo ảnh")
    ax.set_ylabel("StableDINO F1 theo ảnh")
    ax.set_title("So sánh F1 theo ảnh")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    delta = wide["delta_stabledino_minus_yolo"].dropna()
    ax.hist(delta, bins=21, color="#7a9e9f", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("StableDINO F1 - YOLO26x F1")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố độ chênh F1")
    ax.grid(True, axis="y", alpha=0.25)
    savefig(out_dir / "per_image_f1_scatter_delta.svg")
    return wide


def plot_prediction_distributions(predictions: pd.DataFrame, out_dir: Path) -> None:
    pred = predictions[predictions["source"] == "prediction"].copy()
    pred = pred[pred["status"].isin(["TP", "FP"])]
    models = list(pred["model"].drop_duplicates())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for model, group in pred.groupby("model"):
        ax.hist(
            group["confidence"].dropna(),
            bins=18,
            alpha=0.55,
            label=model,
            color=COLORS.get(model),
            density=True,
        )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Mật độ")
    ax.set_title("Phân bố confidence của prediction")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    data = [pred.loc[(pred["model"] == m) & (pred["status"] == "TP"), "best_iou"] for m in models]
    bp = ax.boxplot(data, tick_labels=models, patch_artist=True, showfliers=False)
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(COLORS.get(model, "#999999"))
        patch.set_alpha(0.65)
    ax.set_ylabel("Best IoU của TP")
    ax.set_title("Chất lượng định vị của prediction đúng")
    ax.grid(True, axis="y", alpha=0.25)
    savefig(out_dir / "prediction_confidence_iou_distributions.svg")


def plot_gt_area_recall(predictions: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    gt = predictions[predictions["source"] == "gt"].copy()
    gt["detected"] = gt["status"].eq("matched_gt")
    quantiles = gt["area_px"].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy().copy()
    quantiles[0] -= 1e-6
    labels = ["Nhỏ", "Vừa", "Lớn"]
    gt["area_bucket"] = pd.cut(gt["area_px"], bins=quantiles, labels=labels, include_lowest=True)
    bucket = (
        gt.groupby(["model", "area_bucket"], observed=True)
        .agg(num_gt=("detected", "size"), recall=("detected", "mean"), mean_area_px=("area_px", "mean"))
        .reset_index()
    )
    bucket.to_csv(out_dir / "gt_area_bucket_recall.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(labels))
    width = 0.34
    for i, model in enumerate(["YOLO26x", "StableDINO"]):
        group = bucket[bucket["model"] == model].set_index("area_bucket").reindex(labels)
        bars = ax.bar(
            x + (i - 0.5) * width,
            group["recall"],
            width,
            label=model,
            color=COLORS.get(model),
            alpha=0.8,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nhóm diện tích box ground-truth")
    ax.set_ylabel("Recall")
    ax.set_title("Recall theo kích thước box vết nứt")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    savefig(out_dir / "gt_area_bucket_recall.svg")
    return bucket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--yolo-dir", type=Path, required=True)
    parser.add_argument("--dino-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = label_models(pd.read_csv(args.comparison_dir / "metrics_summary.csv"))
    sweep = label_models(pd.read_csv(args.comparison_dir / "confidence_sweep_metrics.csv"))
    image_metrics = label_models(
        pd.concat(
            [
                pd.read_csv(args.yolo_dir / "image_metrics.csv"),
                pd.read_csv(args.dino_dir / "image_metrics.csv"),
            ],
            ignore_index=True,
        )
    )
    predictions = label_models(
        pd.concat(
            [
                pd.read_csv(args.yolo_dir / "predictions.csv"),
                pd.read_csv(args.dino_dir / "predictions.csv"),
            ],
            ignore_index=True,
        )
    )

    metrics.to_csv(args.output_dir / "metrics_summary_labeled.csv", index=False)
    image_metrics.to_csv(args.output_dir / "per_image_metrics_labeled.csv", index=False)

    plot_confidence_curves(sweep, args.output_dir)
    plot_map_curves(sweep, args.output_dir)
    plot_error_counts(metrics, args.output_dir)
    plot_speed_quality(metrics, args.output_dir)
    plot_per_image_boxplots(image_metrics, args.output_dir)
    plot_per_image_scatter(image_metrics, args.output_dir)
    plot_prediction_distributions(predictions, args.output_dir)
    plot_gt_area_recall(predictions, args.output_dir)

    print(f"Wrote detection plots to {args.output_dir}")


if __name__ == "__main__":
    main()
