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
from segmentation.datasets import sam_finetune as sam_datasets
from segmentation.sam.finetune.runtime import (
    load_inference_config,
    resolve_predict_mode,
    resolve_predict_threshold,
    resolve_refine_settings,
    resolve_tile_settings,
)
from segmentation.sam.finetune.test import _load_finetuned_sam, config_to_dict
from segmentation.sam.finetune.tiled_inference import (
    best_threshold_result,
    binary_mask_from_score_map,
    coarse_refine_model_score_map,
    continuity_metrics,
    metric_per_case,
    tiled_model_score_map,
)


DEFAULT_SAM_CKPT = "/Users/nguyenquangvinh/Desktop/Lab/results/sam_vit_b_01ec64.pth"


@dataclass(frozen=True)
class DatasetSummary:
    dataset_label: str
    dataset_path: str
    image_count: int
    best_threshold: float
    save_threshold: float
    best_precision: float
    best_recall: float
    best_dice: float
    best_iou: float
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


def _resolve_thresholds(args: argparse.Namespace, model_dir: Path) -> list[float]:
    cli_values = [float(v) for v in (args.thresholds or []) if 0.0 < float(v) < 1.0]
    if cli_values:
        return sorted(set(cli_values))

    config_path = Path(args.config).resolve() if args.config else model_dir / "config.txt"
    if config_path.is_file():
        config_dict = config_to_dict(str(config_path))
        config_values = _parse_thresholds(config_dict.get("val_thresholds"))
        if config_values:
            return config_values

    inference_config = load_inference_config(str(model_dir / "best_model.pth"))
    config_values = _parse_thresholds(inference_config.get("val_thresholds"))
    if config_values:
        return config_values

    best_thr = resolve_predict_threshold(str(model_dir / "best_model.pth"), "auto")
    return [float(best_thr)]


def _model_settings(
    *,
    delta_ckpt: Path,
    sam_ckpt: Path,
    config_path: Path | None,
    device,
    device_name: str,
    override_eval_mode: str | None,
    refine_delta_ckpt: str | None,
) -> dict:
    config_dict = config_to_dict(str(config_path)) if config_path and config_path.is_file() else {}
    inference_config = load_inference_config(str(delta_ckpt))

    vit_name = str(config_dict.get("vit_name", "vit_b"))
    delta_type = str(config_dict.get("delta_type", "lora"))
    middle_dim = int(config_dict.get("middle_dim", 32))
    scaling_factor = float(config_dict.get("scaling_factor", 0.1))
    rank = int(config_dict.get("rank", 4))
    decoder_type = str(inference_config.get("decoder_type", config_dict.get("decoder_type", "auto")))
    centerline_head = bool(
        inference_config.get(
            "centerline_head",
            str(config_dict.get("centerline_head", "False")).strip().lower() in {"1", "true", "yes"},
        )
    )
    img_size = int(inference_config.get("img_size", config_dict.get("img_size", 512)))
    eval_mode = resolve_predict_mode(str(delta_ckpt), override_eval_mode)
    tile_size, tile_overlap = resolve_tile_settings(
        str(delta_ckpt),
        inference_config.get("img_size", img_size),
        inference_config.get("tile_overlap", -1),
    )
    tile_batch_size = int(inference_config.get("tile_batch_size", config_dict.get("tile_batch_size", 1)))
    refine_batch_size = int(inference_config.get("refine_batch_size", config_dict.get("refine_batch_size", tile_batch_size)))

    coarse_model, resolved_img_size, resolved_decoder = _load_finetuned_sam(
        ckpt=str(sam_ckpt),
        vit_name=vit_name,
        img_size=img_size,
        delta_type=delta_type,
        delta_ckpt=str(delta_ckpt),
        middle_dim=middle_dim,
        scaling_factor=scaling_factor,
        rank=rank,
        decoder_type=decoder_type,
        centerline_head=centerline_head,
        device=device,
    )

    refine_model = None
    refine_settings = None
    if eval_mode == "coarse_refine":
        refine_delta = str(refine_delta_ckpt or "").strip()
        if not refine_delta:
            raise ValueError("Resolved eval_mode=coarse_refine but no --refine-delta-ckpt was provided.")
        refine_delta_path = Path(refine_delta).resolve()
        refine_delta_type = str(config_dict.get("refine_delta_type", delta_type) or delta_type)
        refine_rank = int(config_dict.get("refine_rank", rank) or rank)
        refine_decoder_type = str(inference_config.get("refine_decoder_type", config_dict.get("refine_decoder_type", "auto")))
        refine_centerline_head = bool(
            inference_config.get(
                "refine_centerline_head",
                str(config_dict.get("refine_centerline_head", "False")).strip().lower() in {"1", "true", "yes"},
            )
        )
        refine_settings = resolve_refine_settings(str(refine_delta_path))
        refine_model, _refine_img_size, _refine_decoder = _load_finetuned_sam(
            ckpt=str(sam_ckpt),
            vit_name=vit_name,
            img_size=int(refine_settings["refine_tile_size"]),
            delta_type=refine_delta_type,
            delta_ckpt=str(refine_delta_path),
            middle_dim=middle_dim,
            scaling_factor=scaling_factor,
            rank=refine_rank,
            decoder_type=refine_decoder_type,
            centerline_head=refine_centerline_head,
            device=device,
        )

    return {
        "coarse_model": coarse_model,
        "refine_model": refine_model,
        "refine_settings": refine_settings,
        "device_name": device_name,
        "eval_mode": eval_mode,
        "img_size": int(resolved_img_size),
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
        "tile_batch_size": int(tile_batch_size),
        "refine_batch_size": int(refine_batch_size),
        "delta_type": delta_type,
        "rank": int(rank),
        "decoder_type": resolved_decoder,
        "centerline_head": bool(centerline_head),
        "vit_name": vit_name,
    }


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            output_root TEXT NOT NULL,
            model_dir TEXT NOT NULL,
            delta_ckpt TEXT NOT NULL,
            config_path TEXT,
            inference_config_path TEXT,
            sam_ckpt TEXT NOT NULL,
            device TEXT NOT NULL,
            eval_mode TEXT NOT NULL,
            vit_name TEXT NOT NULL,
            delta_type TEXT NOT NULL,
            rank INTEGER NOT NULL,
            decoder_type TEXT NOT NULL,
            centerline_head INTEGER NOT NULL,
            img_size INTEGER NOT NULL,
            tile_size INTEGER NOT NULL,
            tile_overlap INTEGER NOT NULL,
            tile_batch_size INTEGER NOT NULL,
            refine_batch_size INTEGER NOT NULL,
            thresholds_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS datasets (
            run_id TEXT NOT NULL,
            dataset_label TEXT NOT NULL,
            dataset_path TEXT NOT NULL,
            image_count INTEGER NOT NULL,
            best_threshold REAL NOT NULL,
            save_threshold REAL NOT NULL,
            best_precision REAL NOT NULL,
            best_recall REAL NOT NULL,
            best_dice REAL NOT NULL,
            best_iou REAL NOT NULL,
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
            mean_precision REAL NOT NULL,
            mean_recall REAL NOT NULL,
            mean_dice REAL NOT NULL,
            mean_iou REAL NOT NULL,
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
            image_name TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            gt_positive_px INTEGER NOT NULL,
            threshold REAL NOT NULL,
            pred_positive_px INTEGER NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            dice REAL NOT NULL,
            iou REAL NOT NULL,
            skeleton_dice REAL NOT NULL,
            centerline_precision REAL NOT NULL,
            centerline_recall REAL NOT NULL,
            component_fragmentation REAL NOT NULL,
            PRIMARY KEY (run_id, dataset_label, image_rel_path, threshold)
        );

        CREATE INDEX IF NOT EXISTS idx_image_threshold_metrics_lookup
        ON image_threshold_metrics (run_id, dataset_label, threshold);
        """
    )
    conn.commit()


def _dataset_label(dataset_path: Path) -> str:
    return f"{dataset_path.parent.name}_{dataset_path.name}"


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


def _image_rows_for_case(
    *,
    run_id: str,
    dataset_label: str,
    dataset_path: Path,
    case_name: str,
    image_hwc: np.ndarray,
    label: np.ndarray,
    score_map: np.ndarray,
    thresholds: list[float],
) -> tuple[list[tuple], dict[float, np.ndarray], dict[float, dict[str, float]]]:
    image_path = (dataset_path / "images" / case_name).resolve()
    label_binary = (np.asarray(label) > 0).astype(np.uint8)
    gt_positive_px = int(label_binary.sum())
    height, width = [int(v) for v in image_hwc.shape[:2]]
    rows: list[tuple] = []
    sum_metrics = {thr: np.zeros(8, dtype=np.float64) for thr in thresholds}
    continuity_by_thr: dict[float, dict[str, float]] = {}

    for thr in thresholds:
        pred = binary_mask_from_score_map(score_map, float(thr))
        precision, recall, dice, iou = metric_per_case(pred, label_binary)
        continuity = continuity_metrics(pred, label_binary)
        pred_positive_px = int(np.asarray(pred, dtype=np.uint8).sum())
        rows.append(
            (
                run_id,
                dataset_label,
                case_name,
                str(image_path),
                image_path.name,
                width,
                height,
                gt_positive_px,
                float(thr),
                pred_positive_px,
                float(precision),
                float(recall),
                float(dice),
                float(iou),
                float(continuity["skeleton_dice"]),
                float(continuity["centerline_precision"]),
                float(continuity["centerline_recall"]),
                float(continuity["component_fragmentation"]),
            )
        )
        sum_metrics[float(thr)] += np.array(
            [
                precision,
                recall,
                dice,
                iou,
                continuity["skeleton_dice"],
                continuity["centerline_precision"],
                continuity["centerline_recall"],
                continuity["component_fragmentation"],
            ],
            dtype=np.float64,
        )
        continuity_by_thr[float(thr)] = continuity
    return rows, sum_metrics, continuity_by_thr


def _compute_score_map(
    *,
    image_hwc: np.ndarray,
    settings: dict,
    delta_ckpt: Path,
) -> np.ndarray:
    eval_mode = str(settings["eval_mode"]).strip().lower()
    if eval_mode == "coarse_refine":
        refine_settings = settings["refine_settings"]
        if refine_settings is None or settings["refine_model"] is None:
            raise ValueError("coarse_refine mode requires refine model settings.")
        score_map, _coarse_map, _refine_outputs = coarse_refine_model_score_map(
            image_hwc,
            coarse_model=settings["coarse_model"],
            coarse_image_size=int(settings["img_size"]),
            coarse_tile_size=int(settings["tile_size"]),
            coarse_tile_overlap=int(settings["tile_overlap"]),
            refine_model=settings["refine_model"],
            refine_image_size=int(refine_settings["refine_tile_size"]),
            refine_tile_size=int(refine_settings["refine_tile_size"]),
            refine_tile_sizes=refine_settings["refine_tile_sizes"],
            refine_max_rois=int(refine_settings["refine_max_rois"]),
            refine_roi_padding=int(refine_settings["refine_roi_padding"]),
            refine_merge_mode=str(refine_settings["refine_merge_mode"]),
            refine_score_threshold=float(refine_settings["refine_score_threshold"]),
            positive_band_low=float(refine_settings["positive_band_low"]),
            positive_band_high=float(refine_settings["positive_band_high"]),
            threshold=float(resolve_predict_threshold(str(delta_ckpt), "auto")),
            multimask_output=False,
            use_amp=False,
            tile_batch_size=int(settings["tile_batch_size"]),
            refine_batch_size=int(settings["refine_batch_size"]),
        )
        return np.asarray(score_map, dtype=np.float32)

    return tiled_model_score_map(
        image_hwc,
        tile_size=int(settings["tile_size"]),
        tile_overlap=int(settings["tile_overlap"]),
        model=settings["coarse_model"],
        image_size=int(settings["img_size"]),
        multimask_output=False,
        use_amp=False,
        tile_batch_size=int(settings["tile_batch_size"]),
    ).astype(np.float32)


def _evaluate_dataset(
    *,
    conn: sqlite3.Connection,
    run_id: str,
    dataset_path: Path,
    thresholds: list[float],
    settings: dict,
    delta_ckpt: Path,
    limit: int | None,
) -> tuple[DatasetSummary, dict[float, tuple[float, float, float, float]], dict[float, dict[str, float]], str]:
    dataset_label = _dataset_label(dataset_path)
    case_names = sam_datasets.list_image_files(str(dataset_path / "images"))
    if limit is not None:
        case_names = case_names[: max(0, int(limit))]
    if not case_names:
        raise ValueError(f"No images found in dataset: {dataset_path}")

    dataset_sums = {thr: np.zeros(8, dtype=np.float64) for thr in thresholds}
    image_rows: list[tuple] = []

    for index, case_name in enumerate(case_names):
        image_hwc, label = sam_datasets.load_image_mask_arrays(str(dataset_path), case_name)
        score_map = _compute_score_map(image_hwc=image_hwc, settings=settings, delta_ckpt=delta_ckpt)
        case_rows, case_sums, _case_continuity = _image_rows_for_case(
            run_id=run_id,
            dataset_label=dataset_label,
            dataset_path=dataset_path,
            case_name=case_name,
            image_hwc=image_hwc,
            label=label,
            score_map=score_map,
            thresholds=thresholds,
        )
        image_rows.extend(case_rows)
        for thr in thresholds:
            dataset_sums[float(thr)] += case_sums[float(thr)]
        print(
            f"[{dataset_label}] {index + 1}/{len(case_names)} {case_name}",
            flush=True,
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO image_threshold_metrics (
            run_id, dataset_label, image_rel_path, image_path, image_name, width, height,
            gt_positive_px, threshold, pred_positive_px, precision, recall, dice, iou,
            skeleton_dice, centerline_precision, centerline_recall, component_fragmentation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        image_rows,
    )

    dataset_metric_rows: list[tuple] = []
    metric_by_thr: dict[float, tuple[float, float, float, float]] = {}
    continuity_by_thr: dict[float, dict[str, float]] = {}
    image_count = len(case_names)
    for thr in thresholds:
        means = dataset_sums[float(thr)] / float(max(1, image_count))
        metric_by_thr[float(thr)] = (
            float(means[0]),
            float(means[1]),
            float(means[2]),
            float(means[3]),
        )
        continuity_by_thr[float(thr)] = {
            "skeleton_dice": float(means[4]),
            "centerline_precision": float(means[5]),
            "centerline_recall": float(means[6]),
            "component_fragmentation": float(means[7]),
        }
        dataset_metric_rows.append(
            (
                run_id,
                dataset_label,
                float(thr),
                float(means[0]),
                float(means[1]),
                float(means[2]),
                float(means[3]),
                float(means[4]),
                float(means[5]),
                float(means[6]),
                float(means[7]),
            )
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO dataset_threshold_metrics (
            run_id, dataset_label, threshold, mean_precision, mean_recall, mean_dice, mean_iou,
            mean_skeleton_dice, mean_centerline_precision, mean_centerline_recall, mean_component_fragmentation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        dataset_metric_rows,
    )

    best_threshold, best_metric = best_threshold_result(metric_by_thr)
    save_threshold = float(best_threshold)
    best_continuity = continuity_by_thr[float(best_threshold)]
    summary = DatasetSummary(
        dataset_label=dataset_label,
        dataset_path=str(dataset_path.resolve()),
        image_count=image_count,
        best_threshold=float(best_threshold),
        save_threshold=float(save_threshold),
        best_precision=float(best_metric[0]),
        best_recall=float(best_metric[1]),
        best_dice=float(best_metric[2]),
        best_iou=float(best_metric[3]),
        best_skeleton_dice=float(best_continuity["skeleton_dice"]),
        best_centerline_precision=float(best_continuity["centerline_precision"]),
        best_centerline_recall=float(best_continuity["centerline_recall"]),
        best_component_fragmentation=float(best_continuity["component_fragmentation"]),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO datasets (
            run_id, dataset_label, dataset_path, image_count, best_threshold, save_threshold,
            best_precision, best_recall, best_dice, best_iou, best_skeleton_dice,
            best_centerline_precision, best_centerline_recall, best_component_fragmentation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            summary.dataset_label,
            summary.dataset_path,
            summary.image_count,
            summary.best_threshold,
            summary.save_threshold,
            summary.best_precision,
            summary.best_recall,
            summary.best_dice,
            summary.best_iou,
            summary.best_skeleton_dice,
            summary.best_centerline_precision,
            summary.best_centerline_recall,
            summary.best_component_fragmentation,
        ),
    )
    conn.commit()

    summary_path = dataset_label + "_metrics_summary.json"
    return summary, metric_by_thr, continuity_by_thr, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM finetune evaluation across thresholds and store per-image metrics in SQLite.")
    parser.add_argument("--delta-ckpt", required=True, help="Path to best_model.pth")
    parser.add_argument("--config", default=None, help="Optional config.txt path. Defaults to sibling config.txt")
    parser.add_argument("--sam-ckpt", default=DEFAULT_SAM_CKPT)
    parser.add_argument("--output-root", required=True, help="Directory where SQLite and summaries are written")
    parser.add_argument("--sqlite-name", default="sam_eval_metrics.sqlite3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--eval-mode", default=None, choices=["auto", "tile_full_box", "coarse_refine"])
    parser.add_argument("--refine-delta-ckpt", default=None, help="Required only when eval-mode resolves to coarse_refine")
    parser.add_argument("--thresholds", type=float, nargs="*", default=None, help="Override threshold sweep")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit per dataset for smoke tests")
    parser.add_argument("datasets", nargs="+", help="Dataset roots containing images/ and masks/")
    args = parser.parse_args()

    delta_ckpt = Path(args.delta_ckpt).resolve()
    if not delta_ckpt.is_file():
        raise FileNotFoundError(f"Delta checkpoint not found: {delta_ckpt}")
    model_dir = delta_ckpt.parent
    config_path = Path(args.config).resolve() if args.config else model_dir / "config.txt"
    sam_ckpt = Path(args.sam_ckpt).resolve()
    if not sam_ckpt.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}")

    thresholds = _resolve_thresholds(args, model_dir)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_root / args.sqlite_name

    device = select_torch_device(args.device)
    device_name = select_device_str(args.device)
    settings = _model_settings(
        delta_ckpt=delta_ckpt,
        sam_ckpt=sam_ckpt,
        config_path=config_path,
        device=device,
        device_name=device_name,
        override_eval_mode=args.eval_mode,
        refine_delta_ckpt=args.refine_delta_ckpt,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    created_at = datetime.now(timezone.utc).isoformat()
    inference_config_path = model_dir / "inference_config.json"

    conn = sqlite3.connect(str(sqlite_path))
    try:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, created_at_utc, output_root, model_dir, delta_ckpt, config_path, inference_config_path,
                sam_ckpt, device, eval_mode, vit_name, delta_type, rank, decoder_type, centerline_head,
                img_size, tile_size, tile_overlap, tile_batch_size, refine_batch_size, thresholds_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                str(output_root),
                str(model_dir),
                str(delta_ckpt),
                str(config_path) if config_path.is_file() else "",
                str(inference_config_path) if inference_config_path.is_file() else "",
                str(sam_ckpt),
                settings["device_name"],
                settings["eval_mode"],
                settings["vit_name"],
                settings["delta_type"],
                settings["rank"],
                settings["decoder_type"],
                1 if settings["centerline_head"] else 0,
                settings["img_size"],
                settings["tile_size"],
                settings["tile_overlap"],
                settings["tile_batch_size"],
                settings["refine_batch_size"],
                json.dumps(thresholds),
            ),
        )
        conn.commit()

        manifest = {
            "run_id": run_id,
            "created_at_utc": created_at,
            "sqlite_path": str(sqlite_path),
            "output_root": str(output_root),
            "delta_ckpt": str(delta_ckpt),
            "config_path": str(config_path) if config_path.is_file() else "",
            "inference_config_path": str(inference_config_path) if inference_config_path.is_file() else "",
            "sam_ckpt": str(sam_ckpt),
            "device": settings["device_name"],
            "eval_mode": settings["eval_mode"],
            "thresholds": thresholds,
            "datasets": [],
        }

        for dataset in args.datasets:
            dataset_path = _resolve_dataset_eval_root(Path(dataset))
            summary, metric_by_thr, continuity_by_thr, summary_filename = _evaluate_dataset(
                conn=conn,
                run_id=run_id,
                dataset_path=dataset_path,
                thresholds=thresholds,
                settings=settings,
                delta_ckpt=delta_ckpt,
                limit=args.limit,
            )
            payload = {
                "run_id": run_id,
                "dataset_label": summary.dataset_label,
                "dataset_path": summary.dataset_path,
                "image_count": summary.image_count,
                "best_threshold": summary.best_threshold,
                "save_threshold": summary.save_threshold,
                "best_metric": {
                    "precision": summary.best_precision,
                    "recall": summary.best_recall,
                    "dice": summary.best_dice,
                    "iou": summary.best_iou,
                },
                "continuity": {
                    "skeleton_dice": summary.best_skeleton_dice,
                    "centerline_precision": summary.best_centerline_precision,
                    "centerline_recall": summary.best_centerline_recall,
                    "component_fragmentation": summary.best_component_fragmentation,
                },
                "metric_by_thr": {
                    str(float(thr)): {
                        "precision": float(metric_by_thr[float(thr)][0]),
                        "recall": float(metric_by_thr[float(thr)][1]),
                        "dice": float(metric_by_thr[float(thr)][2]),
                        "iou": float(metric_by_thr[float(thr)][3]),
                        "skeleton_dice": float(continuity_by_thr[float(thr)]["skeleton_dice"]),
                        "centerline_precision": float(continuity_by_thr[float(thr)]["centerline_precision"]),
                        "centerline_recall": float(continuity_by_thr[float(thr)]["centerline_recall"]),
                        "component_fragmentation": float(continuity_by_thr[float(thr)]["component_fragmentation"]),
                    }
                    for thr in thresholds
                },
            }
            summary_path = output_root / summary_filename
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            manifest["datasets"].append(
                {
                    "dataset_label": summary.dataset_label,
                    "dataset_path": summary.dataset_path,
                    "summary_path": str(summary_path),
                }
            )

        manifest_path = output_root / "run_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    finally:
        conn.close()

    print(str(sqlite_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
