from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_eval(
    *,
    project_root: Path,
    dataset_path: Path,
    model_dir: Path,
    sam_ckpt: Path,
    output_root: Path,
    device: str,
) -> dict:
    output_dir = output_root / dataset_path.parent.name / dataset_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "segmentation.sam.finetune.test",
        "--volume_path",
        str(dataset_path),
        "--output_dir",
        str(output_dir),
        "--ckpt",
        str(sam_ckpt),
        "--delta_ckpt",
        str(model_dir / "best_model.pth"),
        "--delta_type",
        "lora",
        "--vit_name",
        "vit_b",
        "--rank",
        "16",
        "--decoder_type",
        "hq",
        "--centerline_head",
        "--eval_mode",
        "tile_full_box",
        "--img_size",
        "768",
        "--tile_size",
        "768",
        "--tile_overlap",
        "384",
        "--tile_batch_size",
        "1",
        "--pred_threshold",
        "auto",
        "--val_thresholds",
        "0.45",
        "0.5",
        "0.55",
        "0.6",
        "0.65",
        "0.7",
        "0.75",
        "0.8",
        "0.85",
        "0.9",
        "--device",
        device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    subprocess.run(cmd, cwd=project_root, env=env, check=True)
    summary_path = output_dir / "metrics_summary.json"
    return _read_json(summary_path)


def _summary_row(dataset_label: str, payload: dict) -> dict:
    best = payload.get("best_metric", {})
    continuity = payload.get("continuity", {})
    return {
        "dataset": dataset_label,
        "volume_path": payload.get("volume_path", ""),
        "device": payload.get("device", ""),
        "eval_mode": payload.get("eval_mode", ""),
        "best_threshold": payload.get("best_threshold", ""),
        "save_threshold": payload.get("save_threshold", ""),
        "precision": best.get("precision", ""),
        "recall": best.get("recall", ""),
        "f1": best.get("f1", ""),
        "iou": best.get("iou", ""),
        "skeleton_dice": continuity.get("skeleton_dice", ""),
        "centerline_precision": continuity.get("centerline_precision", ""),
        "centerline_recall": continuity.get("centerline_recall", ""),
        "component_fragmentation": continuity.get("component_fragmentation", ""),
    }


def _threshold_rows(dataset_label: str, payload: dict) -> list[dict]:
    rows: list[dict] = []
    for threshold, metrics in sorted(payload.get("metric_by_thr", {}).items(), key=lambda item: float(item[0])):
        rows.append(
            {
                "dataset": dataset_label,
                "threshold": float(threshold),
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1": metrics.get("f1", ""),
                "iou": metrics.get("iou", ""),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM finetune evaluation on multiple datasets and export Excel metrics.")
    parser.add_argument("--model-dir", required=True, help="Directory containing best_model.pth and inference_config.json")
    parser.add_argument("--sam-ckpt", default="/Users/nguyenquangvinh/Desktop/Lab/results/sam_vit_b_01ec64.pth")
    parser.add_argument("--output-root", default="/Users/nguyenquangvinh/Desktop/Lab/eval_outputs")
    parser.add_argument("--excel-path", default="/Users/nguyenquangvinh/Desktop/Lab/eval_outputs/sam_eval_metrics.xlsx")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("datasets", nargs="+", help="Dataset roots containing images/ and masks/")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir).resolve()
    sam_ckpt = Path(args.sam_ckpt).resolve()
    output_root = Path(args.output_root).resolve()
    excel_path = Path(args.excel_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    threshold_rows: list[dict] = []

    for dataset in args.datasets:
        dataset_path = Path(dataset).resolve()
        payload = _run_eval(
            project_root=project_root,
            dataset_path=dataset_path,
            model_dir=model_dir,
            sam_ckpt=sam_ckpt,
            output_root=output_root,
            device=args.device,
        )
        dataset_label = dataset_path.parent.name
        summary_rows.append(_summary_row(dataset_label, payload))
        threshold_rows.extend(_threshold_rows(dataset_label, payload))

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="summary")
        pd.DataFrame(threshold_rows).to_excel(writer, index=False, sheet_name="by_threshold")

    print(excel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
