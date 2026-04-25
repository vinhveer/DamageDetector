from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required to generate comparison plots.") from exc


MODEL_COLORS = {
    "SAM": "#d97706",
    "UNet": "#0f766e",
}


@dataclass(frozen=True)
class RunRecord:
    model_name: str
    dataset_name: str
    sqlite_path: Path


@dataclass(frozen=True)
class DatasetMetrics:
    model_name: str
    dataset_name: str
    sqlite_path: str
    image_count: int
    best_threshold: float
    best_iou: float
    best_dice: float
    threshold_rows: list[dict[str, float]]
    image_iou_at_best_threshold: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot comparison charts for segmentation model SQLite evaluation outputs."
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        nargs=3,
        metavar=("MODEL", "DATASET", "SQLITE"),
        required=True,
        help="Comparison input as MODEL DATASET SQLITE. Repeat exactly once per model/dataset pair.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where PNG charts and summary JSON will be written.",
    )
    return parser.parse_args()


def _load_metrics(record: RunRecord) -> DatasetMetrics:
    conn = sqlite3.connect(str(record.sqlite_path))
    try:
        dataset_row = conn.execute(
            """
            SELECT dataset_label, image_count, best_threshold, best_iou, best_dice
            FROM datasets
            """
        ).fetchone()
        if dataset_row is None:
            raise ValueError(f"Missing datasets row in {record.sqlite_path}")

        threshold_rows = conn.execute(
            """
            SELECT threshold, mean_iou, mean_dice
            FROM dataset_threshold_metrics
            ORDER BY threshold ASC
            """
        ).fetchall()
        image_rows = conn.execute(
            """
            SELECT iou
            FROM image_threshold_metrics
            WHERE threshold = ?
            ORDER BY iou ASC
            """,
            (float(dataset_row[2]),),
        ).fetchall()
    finally:
        conn.close()

    return DatasetMetrics(
        model_name=record.model_name,
        dataset_name=record.dataset_name,
        sqlite_path=str(record.sqlite_path),
        image_count=int(dataset_row[1]),
        best_threshold=float(dataset_row[2]),
        best_iou=float(dataset_row[3]),
        best_dice=float(dataset_row[4]),
        threshold_rows=[
            {"threshold": float(row[0]), "mean_iou": float(row[1]), "mean_dice": float(row[2])}
            for row in threshold_rows
        ],
        image_iou_at_best_threshold=[float(row[0]) for row in image_rows],
    )


def _color_for_model(model_name: str) -> str:
    return MODEL_COLORS.get(model_name, "#334155")


def _plot_best_metrics(metrics_by_dataset: dict[str, list[DatasetMetrics]], output_path: Path) -> None:
    dataset_names = list(metrics_by_dataset.keys())
    metric_names = [("best_iou", "Best IoU"), ("best_dice", "Best Dice")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    bar_width = 0.35

    for ax, (field_name, title) in zip(axes, metric_names):
        x = np.arange(len(dataset_names))
        for offset_idx, model_name in enumerate(["SAM", "UNet"]):
            values = []
            for dataset_name in dataset_names:
                value = next((getattr(item, field_name) for item in metrics_by_dataset[dataset_name] if item.model_name == model_name), np.nan)
                values.append(value)
            positions = x + (offset_idx - 0.5) * bar_width
            ax.bar(
                positions,
                values,
                width=bar_width,
                color=_color_for_model(model_name),
                label=model_name,
            )
            for pos, value in zip(positions, values):
                if not np.isnan(value):
                    ax.text(pos, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper right")
    fig.suptitle("Best Metrics by Dataset")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_curves(metrics_by_dataset: dict[str, list[DatasetMetrics]], output_path: Path) -> None:
    dataset_names = list(metrics_by_dataset.keys())
    fig, axes = plt.subplots(len(dataset_names), 2, figsize=(12, 4.5 * len(dataset_names)), dpi=180, squeeze=False)
    curve_defs = [("mean_iou", "IoU"), ("mean_dice", "Dice")]

    for row_idx, dataset_name in enumerate(dataset_names):
        for col_idx, (field_name, metric_label) in enumerate(curve_defs):
            ax = axes[row_idx][col_idx]
            for item in metrics_by_dataset[dataset_name]:
                thresholds = [row["threshold"] for row in item.threshold_rows]
                values = [row[field_name] for row in item.threshold_rows]
                ax.plot(
                    thresholds,
                    values,
                    marker="o",
                    linewidth=2,
                    color=_color_for_model(item.model_name),
                    label=f"{item.model_name} (best={getattr(item, 'best_' + metric_label.lower()):.3f})",
                )
                ax.axvline(item.best_threshold, linestyle="--", color=_color_for_model(item.model_name), alpha=0.35)
            ax.set_title(f"{dataset_name} - {metric_label} vs Threshold")
            ax.set_xlabel("Threshold")
            ax.set_ylabel(metric_label)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25)
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iou_boxplots(metrics_by_dataset: dict[str, list[DatasetMetrics]], output_path: Path) -> None:
    dataset_names = list(metrics_by_dataset.keys())
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(6 * len(dataset_names), 5), dpi=180, squeeze=False)

    for idx, dataset_name in enumerate(dataset_names):
        ax = axes[0][idx]
        items = metrics_by_dataset[dataset_name]
        box_data = [item.image_iou_at_best_threshold for item in items]
        labels = [f"{item.model_name}\nthr={item.best_threshold:.2f}" for item in items]
        box = ax.boxplot(box_data, patch_artist=True, tick_labels=labels, widths=0.5, showfliers=False)
        for patch, item in zip(box["boxes"], items):
            patch.set_facecolor(_color_for_model(item.model_name))
            patch.set_alpha(0.75)
        ax.set_title(f"{dataset_name} - Per-image IoU")
        ax.set_ylabel("IoU at best threshold")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)

        for pos, item in enumerate(items, start=1):
            median = float(np.median(item.image_iou_at_best_threshold)) if item.image_iou_at_best_threshold else 0.0
            ax.text(pos, 0.03, f"median={median:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_summary(metrics_by_dataset: dict[str, list[DatasetMetrics]], output_path: Path) -> None:
    serializable = {
        dataset_name: [
            {
                "model_name": item.model_name,
                "sqlite_path": item.sqlite_path,
                "image_count": item.image_count,
                "best_threshold": item.best_threshold,
                "best_iou": item.best_iou,
                "best_dice": item.best_dice,
                "median_iou_at_best_threshold": float(np.median(item.image_iou_at_best_threshold))
                if item.image_iou_at_best_threshold
                else 0.0,
            }
            for item in items
        ]
        for dataset_name, items in metrics_by_dataset.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        RunRecord(model_name=model_name, dataset_name=dataset_name, sqlite_path=Path(sqlite_path).expanduser().resolve())
        for model_name, dataset_name, sqlite_path in args.runs
    ]

    metrics_by_dataset: dict[str, list[DatasetMetrics]] = {}
    for record in records:
        item = _load_metrics(record)
        metrics_by_dataset.setdefault(record.dataset_name, []).append(item)

    for dataset_name, items in metrics_by_dataset.items():
        metrics_by_dataset[dataset_name] = sorted(items, key=lambda item: item.model_name)

    _plot_best_metrics(metrics_by_dataset, output_dir / "best_metrics_comparison.png")
    _plot_threshold_curves(metrics_by_dataset, output_dir / "threshold_curves_comparison.png")
    _plot_iou_boxplots(metrics_by_dataset, output_dir / "per_image_iou_boxplots.png")
    _write_summary(metrics_by_dataset, output_dir / "comparison_summary.json")

    print(str(output_dir))


if __name__ == "__main__":
    main()
