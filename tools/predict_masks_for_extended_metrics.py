"""Predict binary masks at best_threshold for extended metrics (S7) and qualitative (S8.2).

Reads *_metrics_summary.json to get best_threshold per (model, dataset),
loads model once, predicts all images, saves binary .png to $EVAL_ROOT/<model>/<dataset>/masks/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from device_utils import select_device_str

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DATASET_MAP = {
    "crack500": "crack500/test",
    "volker": "crack_segmentation_dataset_volker/test",
    "deepcrack": "deepcrack/test",
}

# Maps model tag → how to find its summary JSON dataset_label
DATASET_LABEL_MAP = {
    "crack500": "crack500_test",
    "volker": "crack_segmentation_dataset_volker_test",
    "deepcrack": "deepcrack_test",
}

MODEL_TYPES = {
    "unet_v1": "unet",
    "unet_v2_cldice_ema": "unet",
    "sam_b0_zeroshot": "sam_zeroshot",
    "sam_b1_lora_only": "sam_finetune",
    "sam_b2_lora_hq": "sam_finetune",
    "sam_b3_full": "sam_finetune",
}

# Map model tag → actual directory name under MODEL_ROOT
MODEL_DIR_MAP = {
    "unet_v1": "unet_v1_baseline",
    "unet_v2_cldice_ema": "unet_v2_cldice_ema",
    "sam_b0_zeroshot": "sam_b0_zeroshot",
    "sam_b1_lora_only": "sam_b1_lora_only",
    "sam_b2_lora_hq": "sam_b2_lora_hq",
    "sam_b3_full": "sam_b3_full",
}


def _iter_images(dataset_root: Path) -> list[tuple[Path, str]]:
    """Return (image_path, relative_stem) handling nested dirs."""
    img_dir = dataset_root / "images"
    pairs = []
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(img_dir)
            pairs.append((p, str(rel.with_suffix(""))))
    return pairs


def _get_best_threshold(eval_root: Path, model: str, dataset: str) -> float:
    """Read best_threshold from metrics_summary.json."""
    label = DATASET_LABEL_MAP[dataset]
    summary_path = eval_root / model / f"{label}_metrics_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        return float(data["best_threshold"])
    # Fallback: try to find any summary
    for f in (eval_root / model).glob("*metrics_summary.json"):
        data = json.loads(f.read_text())
        if data.get("dataset_label") == label:
            return float(data["best_threshold"])
    raise FileNotFoundError(f"No summary for {model}/{dataset}")


def predict_unet(model_path: Path, images: list[tuple[Path, str]],
                 threshold: float, output_dir: Path, device: str,
                 tta: bool = True, gaussian_weight: bool = True,
                 multiscale: tuple = (0.75, 1.0, 1.25)):
    """Predict binary masks. Default ON: TTA + Gaussian + Multi-scale (match eval QW)."""
    from segmentation.unet.model_io import load_model_from_checkpoint
    from segmentation.unet.predict_lib.inference import predict_probabilities
    from segmentation.unet.predict_lib.preprocess import load_image_rgb

    model, config = load_model_from_checkpoint(str(model_path), device=device)
    input_size = config.get("args", {}).get("input_size", 512)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path, stem in images:
        img = load_image_rgb(str(img_path))
        prob = predict_probabilities(model, img, device,
                                     input_size=input_size,
                                     tile_overlap=128, tile_batch_size=4,
                                     tta=tta,
                                     multiscale=multiscale,
                                     gaussian_weight=gaussian_weight)
        binary = ((prob >= threshold) * 255).astype(np.uint8)
        out_path = output_dir / f"{stem.replace('/', '__')}__pred.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), binary)
    print(f"    Saved {len(images)} masks to {output_dir}")


def predict_sam_finetune(model_tag: str, eval_root: Path, model_root: Path,
                         sam_ckpt: str, images: list[tuple[Path, str]],
                         threshold: float, output_dir: Path, device: str):
    from segmentation.sam.finetune.test import _load_finetuned_sam, config_to_dict
    from segmentation.sam.finetune.tiled_inference import tiled_model_score_map

    config_path = model_root / model_tag / "refine" / "config.txt"
    delta_ckpt = model_root / model_tag / "refine" / "best_model.pth"
    config_dict = config_to_dict(str(config_path))
    img_size = int(config_dict.get("img_size", 768))

    model, _, _ = _load_finetuned_sam(
        ckpt=sam_ckpt,
        vit_name=str(config_dict.get("vit_name", "vit_b")),
        img_size=img_size,
        delta_type=str(config_dict.get("delta_type", "lora")),
        delta_ckpt=str(delta_ckpt),
        middle_dim=int(config_dict.get("middle_dim", 32)),
        scaling_factor=float(config_dict.get("scaling_factor", 0.1)),
        rank=int(config_dict.get("rank", 4)),
        decoder_type=str(config_dict.get("decoder_type", "baseline")),
        centerline_head=str(config_dict.get("centerline_head", "False")).lower() in ("true", "1"),
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path, stem in images:
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        score_map = tiled_model_score_map(
            img, tile_size=img_size, tile_overlap=img_size // 4,
            model=model, image_size=img_size,
            multimask_output=False, use_amp=False, tile_batch_size=2,
        )
        binary = ((score_map >= threshold) * 255).astype(np.uint8)
        out_path = output_dir / f"{stem.replace('/', '__')}__pred.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), binary)
    print(f"    Saved {len(images)} masks to {output_dir}")


def predict_sam_zeroshot(sam_ckpt: str, images: list[tuple[Path, str]],
                         dataset_root: Path, threshold: float,
                         output_dir: Path, device: str):
    from tools.run_sam_zeroshot_eval_to_sqlite import (
        _load_sam_predictor, _predict_score_map, extract_gt_boxes,
    )
    predictor = _load_sam_predictor(sam_ckpt, "vit_b", device)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = dataset_root / "masks"

    for img_path, stem in images:
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        # Load GT mask for box extraction
        rel = img_path.relative_to(dataset_root / "images")
        gt_path = mask_dir / rel.with_suffix(".png")
        if not gt_path.exists():
            gt_path = mask_dir / rel
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else np.zeros(img.shape[:2], np.uint8)
        boxes = extract_gt_boxes((gt > 0).astype(np.uint8), min_area=16)
        score_map = _predict_score_map(predictor, img, boxes)
        binary = ((score_map >= threshold) * 255).astype(np.uint8)
        out_path = output_dir / f"{stem.replace('/', '__')}__pred.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), binary)
    print(f"    Saved {len(images)} masks to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--dataset-root", required=True, help="data/datasets/ root")
    parser.add_argument("--sam-ckpt", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_TYPES.keys()))
    parser.add_argument("--datasets", nargs="+", default=["crack500", "volker", "deepcrack"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device_str(args.device)
    eval_root = Path(args.eval_root)
    model_root = Path(args.model_root)
    ds_root = Path(args.dataset_root)
    sam_ckpt = args.sam_ckpt or str(ds_root.parent.parent / "training_runs/v2/sam-finetune-lora-hq/sam_vit_b_01ec64.pth")

    for model in args.models:
        mtype = MODEL_TYPES[model]
        for dataset in args.datasets:
            print(f"[{model}] [{dataset}]", flush=True)
            try:
                thr = _get_best_threshold(eval_root, model, dataset)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            ds_path = ds_root / DATASET_MAP[dataset]
            images = _iter_images(ds_path)
            output_dir = eval_root / model / DATASET_LABEL_MAP[dataset] / "masks"

            if mtype == "unet":
                ckpt = model_root / MODEL_DIR_MAP[model] / "best_model.pth"
                predict_unet(ckpt, images, thr, output_dir, device)
            elif mtype == "sam_finetune":
                predict_sam_finetune(model, eval_root, model_root, sam_ckpt,
                                     images, thr, output_dir, device)
            elif mtype == "sam_zeroshot":
                predict_sam_zeroshot(sam_ckpt, images, ds_path, thr, output_dir, device)


if __name__ == "__main__":
    main()
