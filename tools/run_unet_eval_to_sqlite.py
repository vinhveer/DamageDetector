from __future__ import annotations

import argparse
import ast
import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from device_utils import select_device_str, select_torch_device
from segmentation.sam.finetune.tiled_inference import continuity_metrics
from segmentation.unet.model_io import load_model_from_checkpoint, load_training_config_from_path
from segmentation.unet.predict_lib.inference import predict_probabilities
from segmentation.unet.predict_lib.metrics import load_binary_mask, mask_metrics
from segmentation.unet.predict_lib.postprocess import postprocess_binary_mask
from segmentation.unet.predict_lib.preprocess import load_image_rgb


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSummary:
    dataset_label: str
    dataset_path: str
    image_count: int
    best_threshold: float
    best_dice: float
    best_iou: float
    best_precision: float
    best_recall: float
    best_specificity: float
    best_accuracy: float
    best_skeleton_dice: float
    best_centerline_precision: float
    best_centerline_recall: float
    best_component_fragmentation: float


def _parse_thresholds(raw: object) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        values = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = [part.strip() for part in text.split(",") if part.strip()]
        values = parsed if isinstance(parsed, (list, tuple)) else [parsed]
    else:
        values = [raw]

    out: list[float] = []
    seen: set[float] = set()
    for item in values:
        value = float(item)
        if not (0.0 < value < 1.0):
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return sorted(out)


def _resolve_dataset_eval_root(dataset_path: Path) -> Path:
    candidate = dataset_path.expanduser().resolve()
    if (candidate / "images").is_dir() and (candidate / "masks").is_dir():
        return candidate
    test_candidate = candidate / "test"
    if (test_candidate / "images").is_dir() and (test_candidate / "masks").is_dir():
        return test_candidate
    raise FileNotFoundError(
        f"Dataset must contain images/ and masks/ or a test/images + test/masks split root: {candidate}"
    )


def _dataset_label(dataset_path: Path) -> str:
    return f"{dataset_path.parent.name}_{dataset_path.name}"


def _resolve_thresholds(args: argparse.Namespace, model_path: Path) -> list[float]:
    cli_values = [float(v) for v in (args.thresholds or []) if 0.0 < float(v) < 1.0]
    if cli_values:
        return sorted(set(cli_values))

    config = load_training_config_from_path(str(model_path)) or {}
    config_args = config.get("args") or {}
    thresholds = _parse_thresholds(config_args.get("metric_thresholds"))
    metric_threshold = config_args.get("metric_threshold")
    if metric_threshold is not None:
        thresholds = sorted(set(thresholds + [float(metric_threshold)]))
    return thresholds or [0.5]


def _resolve_mode_and_size(args: argparse.Namespace, model_path: Path) -> tuple[str, int, int, int]:
    config = load_training_config_from_path(str(model_path)) or {}
    config_args = config.get("args") or {}
    mode = str(args.mode or "tile").strip().lower()
    input_size = int(args.input_size if args.input_size else config_args.get("input_size", 512))
    overlap = int(args.tile_overlap if args.tile_overlap is not None else max(0, input_size // 2))
    batch_size = int(args.tile_batch_size if args.tile_batch_size else 4)
    return mode, input_size, overlap, batch_size


def _iter_cases(dataset_root: Path) -> list[str]:
    image_dir = dataset_root / "images"
    rows: list[str] = []
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append(path.relative_to(image_dir).as_posix())
    if not rows:
        raise FileNotFoundError(f"No images found under: {image_dir}")
    return rows


def _find_mask_path(masks_dir: Path, relative_image_path: Path) -> Path:
    direct = masks_dir / relative_image_path
    if direct.is_file():
        return direct
    matches = sorted(path for path in (masks_dir / relative_image_path.parent).glob(f"{relative_image_path.stem}.*") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Missing mask for image: {relative_image_path.as_posix()}")
    return matches[0]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            output_root TEXT NOT NULL,
            model_path TEXT NOT NULL,
            train_config_path TEXT,
            device TEXT NOT NULL,
            mode TEXT NOT NULL,
            input_size INTEGER NOT NULL,
            tile_overlap INTEGER NOT NULL,
            tile_batch_size INTEGER NOT NULL,
            apply_postprocessing INTEGER NOT NULL,
            thresholds_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS datasets (
            run_id TEXT NOT NULL,
            dataset_label TEXT NOT NULL,
            dataset_path TEXT NOT NULL,
            image_count INTEGER NOT NULL,
            best_threshold REAL NOT NULL,
            best_dice REAL NOT NULL,
            best_iou REAL NOT NULL,
            best_precision REAL NOT NULL,
            best_recall REAL NOT NULL,
            best_specificity REAL NOT NULL,
            best_accuracy REAL NOT NULL,
            best_skeleton_dice REAL NOT NULL,
            best_centerline_precision REAL NOT NULL,
            best_centerline_recall REAL NOT NULL,
            best_component_fragmentation REAL NOT NULL,
            PRIMARY KEY (run_id, dataset_label)
        );

        CREATE TABLE IF NOT EXISTS dataset_threshold_metrics (
            run_id TEXT NOT NULL,
            dataset_label TEXT NOT NULL,
            threshold REAL NOT NULL,
            mean_dice REAL NOT NULL,
            mean_iou REAL NOT NULL,
            mean_precision REAL NOT NULL,
            mean_recall REAL NOT NULL,
            mean_specificity REAL NOT NULL,
            mean_accuracy REAL NOT NULL,
            mean_skeleton_dice REAL NOT NULL,
            mean_centerline_precision REAL NOT NULL,
            mean_centerline_recall REAL NOT NULL,
            mean_component_fragmentation REAL NOT NULL,
            PRIMARY KEY (run_id, dataset_label, threshold)
        );

        CREATE TABLE IF NOT EXISTS image_threshold_metrics (
            run_id TEXT NOT NULL,
            dataset_label TEXT NOT NULL,
            image_rel_path TEXT NOT NULL,
            image_path TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            image_name TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            gt_positive_px INTEGER NOT NULL,
            threshold REAL NOT NULL,
            pred_positive_px INTEGER NOT NULL,
            dice REAL NOT NULL,
            iou REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            specificity REAL NOT NULL,
            accuracy REAL NOT NULL,
            tp INTEGER NOT NULL,
            fp INTEGER NOT NULL,
            fn INTEGER NOT NULL,
            tn INTEGER NOT NULL,
            skeleton_dice REAL NOT NULL,
            centerline_precision REAL NOT NULL,
            centerline_recall REAL NOT NULL,
            component_fragmentation REAL NOT NULL,
            PRIMARY KEY (run_id, dataset_label, image_rel_path, threshold)
        );
        """
    )
    conn.commit()


def _evaluate_dataset(
    *,
    conn: sqlite3.Connection,
    run_id: str,
    dataset_root: Path,
    thresholds: list[float],
    model,
    device,
    mode: str,
    input_size: int,
    tile_overlap: int,
    tile_batch_size: int,
    apply_postprocessing: bool,
    limit: int | None,
) -> tuple[DatasetSummary, dict[float, dict[str, float]]]:
    dataset_label = _dataset_label(dataset_root)
    case_names = _iter_cases(dataset_root)
    if limit is not None:
        case_names = case_names[: max(0, int(limit))]

    image_rows: list[tuple] = []
    dataset_sums: dict[float, dict[str, float]] = {
        thr: {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "accuracy": 0.0,
            "skeleton_dice": 0.0,
            "centerline_precision": 0.0,
            "centerline_recall": 0.0,
            "component_fragmentation": 0.0,
        }
        for thr in thresholds
    }

    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    for index, case_name in enumerate(case_names, start=1):
        rel_path = Path(case_name)
        image_path = images_dir / rel_path
        mask_path = _find_mask_path(masks_dir, rel_path)
        image = load_image_rgb(str(image_path))
        prob_map = predict_probabilities(
            model,
            image,
            device,
            mode=mode,
            input_size=input_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
        )
        gt = load_binary_mask(str(mask_path), target_size=image.size).astype(np.uint8)
        width, height = image.size
        gt_positive_px = int(gt.sum())

        for thr in thresholds:
            pred = prob_map > float(thr)
            pred = postprocess_binary_mask(pred, apply_postprocessing=apply_postprocessing)
            pred_u8 = pred.astype(np.uint8)
            metrics = mask_metrics(pred_u8, gt)
            continuity = continuity_metrics(pred_u8, gt)
            pred_positive_px = int(pred_u8.sum())
            image_rows.append(
                (
                    run_id,
                    dataset_label,
                    rel_path.as_posix(),
                    str(image_path),
                    str(mask_path),
                    image_path.name,
                    int(width),
                    int(height),
                    gt_positive_px,
                    float(thr),
                    pred_positive_px,
                    float(metrics["dice"]),
                    float(metrics["iou"]),
                    float(metrics["precision"]),
                    float(metrics["recall"]),
                    float(metrics["specificity"]),
                    float(metrics["accuracy"]),
                    int(metrics["tp"]),
                    int(metrics["fp"]),
                    int(metrics["fn"]),
                    int(metrics["tn"]),
                    float(continuity["skeleton_dice"]),
                    float(continuity["centerline_precision"]),
                    float(continuity["centerline_recall"]),
                    float(continuity["component_fragmentation"]),
                )
            )
            for key in dataset_sums[float(thr)]:
                dataset_sums[float(thr)][key] += float(
                    continuity[key] if key in continuity else metrics[key]
                )
        print(f"[{dataset_label}] {index}/{len(case_names)} {case_name}", flush=True)

    conn.executemany(
        """
        INSERT OR REPLACE INTO image_threshold_metrics (
            run_id, dataset_label, image_rel_path, image_path, mask_path, image_name, width, height,
            gt_positive_px, threshold, pred_positive_px, dice, iou, precision, recall, specificity, accuracy,
            tp, fp, fn, tn, skeleton_dice, centerline_precision, centerline_recall, component_fragmentation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        image_rows,
    )

    image_count = len(case_names)
    threshold_summary: dict[float, dict[str, float]] = {}
    threshold_rows: list[tuple] = []
    for thr in thresholds:
        means = {key: float(value) / max(1, image_count) for key, value in dataset_sums[float(thr)].items()}
        threshold_summary[float(thr)] = means
        threshold_rows.append(
            (
                run_id,
                dataset_label,
                float(thr),
                means["dice"],
                means["iou"],
                means["precision"],
                means["recall"],
                means["specificity"],
                means["accuracy"],
                means["skeleton_dice"],
                means["centerline_precision"],
                means["centerline_recall"],
                means["component_fragmentation"],
            )
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO dataset_threshold_metrics (
            run_id, dataset_label, threshold, mean_dice, mean_iou, mean_precision, mean_recall,
            mean_specificity, mean_accuracy, mean_skeleton_dice, mean_centerline_precision,
            mean_centerline_recall, mean_component_fragmentation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        threshold_rows,
    )

    best_threshold = max(threshold_summary.keys(), key=lambda thr: (threshold_summary[thr]["iou"], threshold_summary[thr]["dice"], -abs(thr - 0.5)))
    best = threshold_summary[float(best_threshold)]
    summary = DatasetSummary(
        dataset_label=dataset_label,
        dataset_path=str(dataset_root),
        image_count=image_count,
        best_threshold=float(best_threshold),
        best_dice=float(best["dice"]),
        best_iou=float(best["iou"]),
        best_precision=float(best["precision"]),
        best_recall=float(best["recall"]),
        best_specificity=float(best["specificity"]),
        best_accuracy=float(best["accuracy"]),
        best_skeleton_dice=float(best["skeleton_dice"]),
        best_centerline_precision=float(best["centerline_precision"]),
        best_centerline_recall=float(best["centerline_recall"]),
        best_component_fragmentation=float(best["component_fragmentation"]),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO datasets (
            run_id, dataset_label, dataset_path, image_count, best_threshold, best_dice, best_iou,
            best_precision, best_recall, best_specificity, best_accuracy, best_skeleton_dice,
            best_centerline_precision, best_centerline_recall, best_component_fragmentation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            summary.dataset_label,
            summary.dataset_path,
            summary.image_count,
            summary.best_threshold,
            summary.best_dice,
            summary.best_iou,
            summary.best_precision,
            summary.best_recall,
            summary.best_specificity,
            summary.best_accuracy,
            summary.best_skeleton_dice,
            summary.best_centerline_precision,
            summary.best_centerline_recall,
            summary.best_component_fragmentation,
        ),
    )
    conn.commit()
    return summary, threshold_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UNet evaluation across thresholds and store per-image metrics in SQLite.")
    parser.add_argument("--model", required=True, help="Path to UNet best_model.pth")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sqlite-name", default="unet_eval_metrics.sqlite3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mode", default=None, choices=["tile", "letterbox", "resize"])
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--no-postprocessing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("dataset", help="Dataset root containing images/masks or split root with test/images + test/masks")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    dataset_root = _resolve_dataset_eval_root(Path(args.dataset))
    thresholds = _resolve_thresholds(args, model_path)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_root / args.sqlite_name

    device = select_torch_device(args.device)
    device_name = select_device_str(args.device)
    model, _model_config = load_model_from_checkpoint(str(model_path), device)
    mode, input_size, tile_overlap, tile_batch_size = _resolve_mode_and_size(args, model_path)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    created_at = datetime.now(timezone.utc).isoformat()
    config_path = model_path.parent / "train_config.json"

    conn = sqlite3.connect(str(sqlite_path))
    try:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, created_at_utc, output_root, model_path, train_config_path, device, mode,
                input_size, tile_overlap, tile_batch_size, apply_postprocessing, thresholds_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                str(output_root),
                str(model_path),
                str(config_path) if config_path.is_file() else "",
                device_name,
                mode,
                int(input_size),
                int(tile_overlap),
                int(tile_batch_size),
                0 if args.no_postprocessing else 1,
                json.dumps(thresholds),
            ),
        )
        conn.commit()

        summary, threshold_summary = _evaluate_dataset(
            conn=conn,
            run_id=run_id,
            dataset_root=dataset_root,
            thresholds=thresholds,
            model=model,
            device=device,
            mode=mode,
            input_size=input_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            apply_postprocessing=not args.no_postprocessing,
            limit=args.limit,
        )
    finally:
        conn.close()

    summary_path = output_root / f"{summary.dataset_label}_metrics_summary.json"
    payload = {
        "run_id": run_id,
        "dataset_label": summary.dataset_label,
        "dataset_path": summary.dataset_path,
        "image_count": summary.image_count,
        "best_threshold": summary.best_threshold,
        "best_metric": {
            "dice": summary.best_dice,
            "iou": summary.best_iou,
            "precision": summary.best_precision,
            "recall": summary.best_recall,
            "specificity": summary.best_specificity,
            "accuracy": summary.best_accuracy,
        },
        "continuity": {
            "skeleton_dice": summary.best_skeleton_dice,
            "centerline_precision": summary.best_centerline_precision,
            "centerline_recall": summary.best_centerline_recall,
            "component_fragmentation": summary.best_component_fragmentation,
        },
        "metric_by_thr": {str(float(thr)): values for thr, values in threshold_summary.items()},
        "sqlite_path": str(sqlite_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(str(sqlite_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
