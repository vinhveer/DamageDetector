from __future__ import annotations

import argparse
import csv
import json
import math
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
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("matplotlib and numpy are required to generate comparison plots.") from exc

from plot_style import apply_science_style, save_report_figure

apply_science_style()


MODEL_COLORS = {
    "SAM": "#d97706",
    "UNet": "#0f766e",
}
METRIC_FIELDS = ["iou", "dice", "precision", "recall", "skeleton_dice", "component_fragmentation"]


@dataclass(frozen=True)
class RunInput:
    model: str
    dataset: str
    sqlite_path: Path


@dataclass
class RunMetrics:
    model: str
    dataset: str
    sqlite_path: str
    image_count: int
    best_threshold: float
    best: dict[str, float]
    thresholds: list[dict[str, float]]
    images_at_best: list[dict[str, float | str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rich plots for U-Net vs SAM segmentation evaluation.")
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        nargs=3,
        metavar=("MODEL", "DATASET", "SQLITE"),
        required=True,
        help="Input as MODEL DATASET SQLITE. Repeat for each model/dataset pair.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for figures and CSV/JSON tables.")
    return parser.parse_args()


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column for row in rows)


def _fetch_one_dict(conn: sqlite3.Connection, query: str, params: tuple = ()) -> dict[str, float | str | int]:
    cursor = conn.execute(query, params)
    row = cursor.fetchone()
    if row is None:
        return {}
    names = [item[0] for item in cursor.description]
    return dict(zip(names, row))


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_run(record: RunInput) -> RunMetrics:
    conn = sqlite3.connect(str(record.sqlite_path))
    try:
        dataset_row = _fetch_one_dict(conn, "SELECT * FROM datasets LIMIT 1")
        if not dataset_row:
            raise ValueError(f"Missing datasets table row in {record.sqlite_path}")
        best_threshold = _safe_float(dataset_row.get("best_threshold"))
        threshold_rows = conn.execute("SELECT * FROM dataset_threshold_metrics ORDER BY threshold ASC").fetchall()
        threshold_names = [item[0] for item in conn.execute("SELECT * FROM dataset_threshold_metrics LIMIT 1").description]
        thresholds = [dict(zip(threshold_names, row)) for row in threshold_rows]

        has_specificity = _has_column(conn, "image_threshold_metrics", "specificity")
        has_accuracy = _has_column(conn, "image_threshold_metrics", "accuracy")
        select_parts = [
            "image_rel_path",
            "image_name",
            "width",
            "height",
            "gt_positive_px",
            "pred_positive_px",
            "precision",
            "recall",
            "dice",
            "iou",
            "skeleton_dice",
            "centerline_precision",
            "centerline_recall",
            "component_fragmentation",
        ]
        if has_specificity:
            select_parts.append("specificity")
        if has_accuracy:
            select_parts.append("accuracy")
        image_rows = conn.execute(
            f"SELECT {', '.join(select_parts)} FROM image_threshold_metrics WHERE threshold = ? ORDER BY image_rel_path ASC",
            (best_threshold,),
        ).fetchall()
    finally:
        conn.close()

    images = []
    for row in image_rows:
        item = dict(zip(select_parts, row))
        width = _safe_float(item.get("width"), 0.0)
        height = _safe_float(item.get("height"), 0.0)
        total_px = max(width * height, 1.0)
        gt_px = _safe_float(item.get("gt_positive_px"), 0.0)
        pred_px = _safe_float(item.get("pred_positive_px"), 0.0)
        dice = _safe_float(item.get("dice"), 0.0)
        tp = max(0.0, min(gt_px, pred_px, 0.5 * dice * (gt_px + pred_px)))
        fp = max(0.0, pred_px - tp)
        fn = max(0.0, gt_px - tp)
        tn = max(0.0, total_px - tp - fp - fn)
        if math.isnan(_safe_float(item.get("specificity"))):
            item["specificity"] = tn / max(tn + fp, 1.0)
        if math.isnan(_safe_float(item.get("accuracy"))):
            item["accuracy"] = (tp + tn) / max(total_px, 1.0)
        item["gt_positive_ratio"] = gt_px / total_px
        item["pred_positive_ratio"] = pred_px / total_px
        item["area_ratio_error"] = pred_px / max(gt_px, 1.0)
        images.append(item)

    best = {
        "threshold": best_threshold,
        "precision": _safe_float(dataset_row.get("best_precision")),
        "recall": _safe_float(dataset_row.get("best_recall")),
        "dice": _safe_float(dataset_row.get("best_dice")),
        "iou": _safe_float(dataset_row.get("best_iou")),
        "specificity": _safe_float(dataset_row.get("best_specificity")),
        "accuracy": _safe_float(dataset_row.get("best_accuracy")),
        "skeleton_dice": _safe_float(dataset_row.get("best_skeleton_dice")),
        "centerline_precision": _safe_float(dataset_row.get("best_centerline_precision")),
        "centerline_recall": _safe_float(dataset_row.get("best_centerline_recall")),
        "component_fragmentation": _safe_float(dataset_row.get("best_component_fragmentation")),
    }
    for derived_field in ("specificity", "accuracy"):
        if math.isnan(best[derived_field]):
            values = [_safe_float(row.get(derived_field)) for row in images]
            finite_values = [value for value in values if not math.isnan(value)]
            best[derived_field] = float(np.mean(finite_values)) if finite_values else math.nan
    return RunMetrics(
        model=record.model,
        dataset=record.dataset,
        sqlite_path=str(record.sqlite_path),
        image_count=int(dataset_row.get("image_count", len(images))),
        best_threshold=best_threshold,
        best=best,
        thresholds=[_normalize_threshold_row(row) for row in thresholds],
        images_at_best=images,
    )


def _normalize_threshold_row(row: dict[str, object]) -> dict[str, float]:
    return {
        "threshold": _safe_float(row.get("threshold")),
        "precision": _safe_float(row.get("mean_precision")),
        "recall": _safe_float(row.get("mean_recall")),
        "dice": _safe_float(row.get("mean_dice")),
        "iou": _safe_float(row.get("mean_iou")),
        "specificity": _safe_float(row.get("mean_specificity")),
        "accuracy": _safe_float(row.get("mean_accuracy")),
        "skeleton_dice": _safe_float(row.get("mean_skeleton_dice")),
        "centerline_precision": _safe_float(row.get("mean_centerline_precision")),
        "centerline_recall": _safe_float(row.get("mean_centerline_recall")),
        "component_fragmentation": _safe_float(row.get("mean_component_fragmentation")),
    }


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, "#334155")


def write_tables(metrics: list[RunMetrics], output_dir: Path) -> None:
    summary_rows = []
    threshold_rows = []
    image_rows = []
    for item in metrics:
        row = {
            "dataset": item.dataset,
            "model": item.model,
            "sqlite_path": item.sqlite_path,
            "image_count": item.image_count,
            "best_threshold": item.best_threshold,
            **{f"best_{key}": value for key, value in item.best.items() if key != "threshold"},
        }
        summary_rows.append(row)
        for threshold_row in item.thresholds:
            threshold_rows.append({"dataset": item.dataset, "model": item.model, **threshold_row})
        for image_row in item.images_at_best:
            image_rows.append({"dataset": item.dataset, "model": item.model, "threshold": item.best_threshold, **image_row})

    _write_csv(output_dir / "metrics_summary.csv", summary_rows)
    _write_csv(output_dir / "threshold_sweep.csv", threshold_rows)
    _write_csv(output_dir / "per_image_best_threshold_metrics.csv", image_rows)

    payload = _json_safe({
        "runs": [
            {
                "dataset": item.dataset,
                "model": item.model,
                "sqlite_path": item.sqlite_path,
                "image_count": item.image_count,
                "best_threshold": item.best_threshold,
                "best": item.best,
                "image_iou_median": _nanmedian([_safe_float(row.get("iou")) for row in item.images_at_best]),
                "image_dice_median": _nanmedian([_safe_float(row.get("dice")) for row in item.images_at_best]),
            }
            for item in metrics
        ]
    })
    (output_dir / "comparison_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _nanmedian(values: list[float]) -> float:
    arr = np.array([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return math.nan
    return float(np.median(arr))


def _json_safe(value):
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def group_by_dataset(metrics: list[RunMetrics]) -> dict[str, list[RunMetrics]]:
    grouped: dict[str, list[RunMetrics]] = {}
    for item in metrics:
        grouped.setdefault(item.dataset, []).append(item)
    for dataset in grouped:
        grouped[dataset] = sorted(grouped[dataset], key=lambda item: item.model)
    return grouped


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def plot_best_metric_grid(grouped: dict[str, list[RunMetrics]], output_path: Path) -> None:
    datasets = list(grouped)
    metrics = [("iou", "IoU"), ("dice", "Dice"), ("precision", "Precision"), ("recall", "Recall")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=180)
    axes = axes.ravel()
    width = 0.36
    for ax, (field, title) in zip(axes, metrics):
        x = np.arange(len(datasets))
        for idx, model in enumerate(["SAM", "UNet"]):
            values = []
            thresholds = []
            for dataset in datasets:
                item = next((run for run in grouped[dataset] if run.model == model), None)
                values.append(item.best[field] if item else np.nan)
                thresholds.append(item.best_threshold if item else np.nan)
            pos = x + (idx - 0.5) * width
            ax.bar(pos, values, width=width, color=_color(model), label=model)
            for p, value, threshold in zip(pos, values, thresholds):
                if not np.isnan(value):
                    ax.text(p, value + 0.012, f"{value:.3f}\nthr={threshold:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower right")
    fig.suptitle("Best Segmentation Metrics by Dataset")
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_best_metric_pair(grouped: dict[str, list[RunMetrics]], fields: list[tuple[str, str]], output_path: Path) -> None:
    datasets = list(grouped)
    fig, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 4.4), dpi=180, squeeze=False)
    width = 0.36
    for ax, (field, title) in zip(axes.ravel(), fields):
        x = np.arange(len(datasets))
        for idx, model in enumerate(["SAM", "UNet"]):
            values = []
            thresholds = []
            for dataset in datasets:
                item = next((run for run in grouped[dataset] if run.model == model), None)
                values.append(item.best[field] if item else np.nan)
                thresholds.append(item.best_threshold if item else np.nan)
            pos = x + (idx - 0.5) * width
            ax.bar(pos, values, width=width, color=_color(model), label=model)
            for p, value, threshold in zip(pos, values, thresholds):
                if not np.isnan(value):
                    ax.text(p, value + 0.012, f"{value:.3f}\nthr={threshold:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].legend(loc="lower right")
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_threshold_curves(grouped: dict[str, list[RunMetrics]], output_path: Path) -> None:
    datasets = list(grouped)
    fields = [("iou", "IoU"), ("dice", "Dice"), ("precision", "Precision"), ("recall", "Recall")]
    rows_per_dataset = 2
    cols = 2
    fig, axes = plt.subplots(len(datasets) * rows_per_dataset, cols, figsize=(10.5, 12.0), dpi=180, squeeze=False)
    for row_idx, dataset in enumerate(datasets):
        for field_idx, (field, title) in enumerate(fields):
            ax = axes[row_idx * rows_per_dataset + field_idx // cols][field_idx % cols]
            for item in grouped[dataset]:
                xs = [row["threshold"] for row in item.thresholds]
                ys = [row[field] for row in item.thresholds]
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=3.5, color=_color(item.model), label=item.model)
                ax.axvline(item.best_threshold, color=_color(item.model), linestyle="--", alpha=0.35)
            ax.set_title(f"{dataset} - {title}")
            ax.set_xlabel("Threshold")
            ax.set_ylabel(title)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if row_idx == 0 and field_idx == len(fields) - 1:
                ax.legend(loc="lower left")
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_threshold_curve_pair(grouped: dict[str, list[RunMetrics]], dataset: str, fields: list[tuple[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 4.2), dpi=180, squeeze=False)
    for ax, (field, title) in zip(axes.ravel(), fields):
        for item in grouped[dataset]:
            xs = [row["threshold"] for row in item.thresholds]
            ys = [row[field] for row in item.thresholds]
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=3.5, color=_color(item.model), label=item.model)
            ax.axvline(item.best_threshold, color=_color(item.model), linestyle="--", alpha=0.35)
        ax.set_title(f"{dataset} - {title}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(loc="lower left")
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_iou_dice_boxplots(grouped: dict[str, list[RunMetrics]], output_path: Path) -> None:
    datasets = list(grouped)
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10.5, 9.0), dpi=180, squeeze=False)
    for row_idx, dataset in enumerate(datasets):
        for col_idx, field in enumerate(["iou", "dice"]):
            ax = axes[row_idx][col_idx]
            items = grouped[dataset]
            data = [[_safe_float(row.get(field)) for row in item.images_at_best] for item in items]
            labels = [f"{item.model}\nthr={item.best_threshold:.2f}" for item in items]
            box = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)
            for patch, item in zip(box["boxes"], items):
                patch.set_facecolor(_color(item.model))
                patch.set_alpha(0.72)
            ax.set_title(f"{dataset} - per-image {field.upper()}")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
            for pos, values in enumerate(data, start=1):
                ax.text(pos, 0.03, f"med={_nanmedian(values):.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_iou_dice_boxplot_dataset(grouped: dict[str, list[RunMetrics]], dataset: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=180, squeeze=False)
    for ax, field in zip(axes.ravel(), ["iou", "dice"]):
        items = grouped[dataset]
        data = [[_safe_float(row.get(field)) for row in item.images_at_best] for item in items]
        labels = [f"{item.model}\nthr={item.best_threshold:.2f}" for item in items]
        box = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)
        for patch, item in zip(box["boxes"], items):
            patch.set_facecolor(_color(item.model))
            patch.set_alpha(0.72)
        ax.set_title(f"{dataset} - per-image {field.upper()}")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        for pos, values in enumerate(data, start=1):
            ax.text(pos, 0.03, f"med={_nanmedian(values):.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_pairwise_delta(grouped: dict[str, list[RunMetrics]], output_path: Path) -> None:
    datasets = [name for name, items in grouped.items() if {item.model for item in items} >= {"SAM", "UNet"}]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4.2 * len(datasets)), dpi=180, squeeze=False)
    for row_idx, dataset in enumerate(datasets):
        sam = next(item for item in grouped[dataset] if item.model == "SAM")
        unet = next(item for item in grouped[dataset] if item.model == "UNet")
        sam_map = {str(row["image_rel_path"]): row for row in sam.images_at_best}
        unet_map = {str(row["image_rel_path"]): row for row in unet.images_at_best}
        keys = sorted(set(sam_map) & set(unet_map))
        x_unet = np.array([_safe_float(unet_map[key].get("iou")) for key in keys], dtype=np.float64)
        y_sam = np.array([_safe_float(sam_map[key].get("iou")) for key in keys], dtype=np.float64)
        delta = y_sam - x_unet
        gt_ratio = np.array([_safe_float(unet_map[key].get("gt_positive_ratio")) for key in keys], dtype=np.float64)

        ax = axes[row_idx][0]
        ax.scatter(x_unet, y_sam, s=16, alpha=0.65, color="#475569")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("U-Net IoU")
        ax.set_ylabel("SAM IoU")
        ax.set_title(f"{dataset}: per-image IoU scatter")
        ax.grid(alpha=0.25)

        ax = axes[row_idx][1]
        order = np.argsort(gt_ratio)
        ax.scatter(np.arange(len(delta)), delta[order], s=16, alpha=0.7, color="#7c3aed")
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
        ax.set_xlabel("Images sorted by GT crack ratio")
        ax.set_ylabel("SAM IoU - U-Net IoU")
        ax.set_title(f"{dataset}: model advantage by image")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_pairwise_delta_dataset(grouped: dict[str, list[RunMetrics]], dataset: str, output_path: Path) -> None:
    sam = next(item for item in grouped[dataset] if item.model == "SAM")
    unet = next(item for item in grouped[dataset] if item.model == "UNet")
    sam_map = {str(row["image_rel_path"]): row for row in sam.images_at_best}
    unet_map = {str(row["image_rel_path"]): row for row in unet.images_at_best}
    keys = sorted(set(sam_map) & set(unet_map))
    x_unet = np.array([_safe_float(unet_map[key].get("iou")) for key in keys], dtype=np.float64)
    y_sam = np.array([_safe_float(sam_map[key].get("iou")) for key in keys], dtype=np.float64)
    delta = y_sam - x_unet
    gt_ratio = np.array([_safe_float(unet_map[key].get("gt_positive_ratio")) for key in keys], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=180, squeeze=False)
    ax = axes.ravel()[0]
    ax.scatter(x_unet, y_sam, s=16, alpha=0.65, color="#475569")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("U-Net IoU")
    ax.set_ylabel("SAM IoU")
    ax.set_title(f"{dataset}: per-image IoU scatter")
    ax.grid(alpha=0.25)

    ax = axes.ravel()[1]
    order = np.argsort(gt_ratio)
    ax.scatter(np.arange(len(delta)), delta[order], s=16, alpha=0.7, color="#7c3aed")
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xlabel("Images sorted by GT crack ratio")
    ax.set_ylabel("SAM IoU - U-Net IoU")
    ax.set_title(f"{dataset}: model advantage by image")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_area_bucket_performance(grouped: dict[str, list[RunMetrics]], output_path: Path) -> None:
    datasets = list(grouped)
    labels = ["small", "medium", "large"]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 4.0 * len(datasets)), dpi=180, squeeze=False)
    for row_idx, dataset in enumerate(datasets):
        for col_idx, field in enumerate(["iou", "recall"]):
            ax = axes[row_idx][col_idx]
            x = np.arange(len(labels))
            width = 0.36
            for idx, item in enumerate(grouped[dataset]):
                ratios = np.array([_safe_float(row.get("gt_positive_ratio"), 0.0) for row in item.images_at_best], dtype=np.float64)
                values = np.array([_safe_float(row.get(field)) for row in item.images_at_best], dtype=np.float64)
                if ratios.size == 0:
                    bucket_values = [np.nan, np.nan, np.nan]
                else:
                    q1, q2 = np.quantile(ratios, [1 / 3, 2 / 3])
                    masks = [ratios <= q1, (ratios > q1) & (ratios <= q2), ratios > q2]
                    bucket_values = [float(np.nanmean(values[mask])) if np.any(mask) else np.nan for mask in masks]
                pos = x + (idx - 0.5) * width
                ax.bar(pos, bucket_values, width=width, color=_color(item.model), label=item.model)
                for p, value in zip(pos, bucket_values):
                    if not np.isnan(value):
                        ax.text(p, value + 0.01, f"{value:.3f}", ha="center", fontsize=7)
            ax.set_title(f"{dataset} - {field.upper()} by crack area bucket")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="lower right")
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_area_bucket_dataset(grouped: dict[str, list[RunMetrics]], dataset: str, output_path: Path) -> None:
    labels = ["small", "medium", "large"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=180, squeeze=False)
    for ax, field in zip(axes.ravel(), ["iou", "recall"]):
        x = np.arange(len(labels))
        width = 0.36
        for idx, item in enumerate(grouped[dataset]):
            ratios = np.array([_safe_float(row.get("gt_positive_ratio"), 0.0) for row in item.images_at_best], dtype=np.float64)
            values = np.array([_safe_float(row.get(field)) for row in item.images_at_best], dtype=np.float64)
            if ratios.size == 0:
                bucket_values = [np.nan, np.nan, np.nan]
            else:
                q1, q2 = np.quantile(ratios, [1 / 3, 2 / 3])
                masks = [ratios <= q1, (ratios > q1) & (ratios <= q2), ratios > q2]
                bucket_values = [float(np.nanmean(values[mask])) if np.any(mask) else np.nan for mask in masks]
            pos = x + (idx - 0.5) * width
            ax.bar(pos, bucket_values, width=width, color=_color(item.model), label=item.model)
            for p, value in zip(pos, bucket_values):
                if not np.isnan(value):
                    ax.text(p, value + 0.01, f"{value:.3f}", ha="center", fontsize=7)
        ax.set_title(f"{dataset} - {field.upper()} by crack area bucket")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].legend(loc="lower right")
    fig.tight_layout()
    save_report_figure(fig, output_path)


def plot_heatmap(grouped: dict[str, list[RunMetrics]], output_path: Path) -> None:
    rows = []
    labels = []
    for dataset, items in grouped.items():
        for item in items:
            rows.append([item.best.get(field, math.nan) for field in ["iou", "dice", "precision", "recall", "skeleton_dice"]])
            labels.append(f"{dataset}\n{item.model}")
    data = np.array(rows, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.8, max(4.5, 0.58 * len(labels))), dpi=180)
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    columns = ["IoU", "Dice", "Precision", "Recall", "Skel.Dice"]
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text = "-" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if not np.isnan(value) and value < 0.65 else "black", fontsize=8)
    ax.set_title("Best-threshold Metric Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_report_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [RunInput(model, dataset, Path(sqlite_path).expanduser().resolve()) for model, dataset, sqlite_path in args.runs]
    metrics = [load_run(record) for record in records]
    grouped = group_by_dataset(metrics)

    write_tables(metrics, output_dir)
    plot_best_metric_grid(grouped, output_dir / "best_metrics_grid.svg")
    plot_best_metric_pair(grouped, [("iou", "IoU"), ("dice", "Dice")], output_dir / "best_metrics_iou_dice.svg")
    plot_best_metric_pair(grouped, [("precision", "Precision"), ("recall", "Recall")], output_dir / "best_metrics_precision_recall.svg")
    plot_threshold_curves(grouped, output_dir / "threshold_curves_all_metrics.svg")
    for dataset in grouped:
        slug = _slug(dataset)
        plot_threshold_curve_pair(grouped, dataset, [("iou", "IoU"), ("dice", "Dice")], output_dir / f"threshold_curves_{slug}_iou_dice.svg")
        plot_threshold_curve_pair(grouped, dataset, [("precision", "Precision"), ("recall", "Recall")], output_dir / f"threshold_curves_{slug}_precision_recall.svg")
    plot_iou_dice_boxplots(grouped, output_dir / "per_image_iou_dice_boxplots.svg")
    for dataset in grouped:
        plot_iou_dice_boxplot_dataset(grouped, dataset, output_dir / f"per_image_{_slug(dataset)}_iou_dice_boxplots.svg")
    plot_pairwise_delta(grouped, output_dir / "per_image_iou_scatter_delta.svg")
    for dataset, items in grouped.items():
        if {item.model for item in items} >= {"SAM", "UNet"}:
            plot_pairwise_delta_dataset(grouped, dataset, output_dir / f"per_image_{_slug(dataset)}_iou_scatter_delta.svg")
    plot_area_bucket_performance(grouped, output_dir / "area_bucket_iou_recall.svg")
    for dataset in grouped:
        plot_area_bucket_dataset(grouped, dataset, output_dir / f"area_bucket_{_slug(dataset)}_iou_recall.svg")
    plot_heatmap(grouped, output_dir / "best_metric_heatmap.svg")
    print(str(output_dir))


if __name__ == "__main__":
    main()
