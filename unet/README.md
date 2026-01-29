# Crack Detection U-Net

## Overview

This project implements crack segmentation using an improved U-Net architecture (with attention mechanisms). It supports:

- Training on datasets like `CRACK500`
- Predicting a single image or a whole folder (`predict.py`)
- Evaluating predicted masks vs CrackForest groundTruth `.mat` annotations and exporting metrics (`compare_result.py`)

## Project Structure

```
unet/unet_model.py   # Improved U-Net model definition
dataset_lib/         # Dataset loading and preprocessing (modular)
train.py             # Training entrypoint
predict.py           # Predict a single image or a folder
predict_lib/         # Prediction utilities (single + folder)
ui/                  # UI implementation (entrypoint at ui.py)
compare_result.py    # Metrics vs CrackForest groundTruth (.mat) + CSV/XLSX export
compare/             # CrackForest dataset + evaluation utilities/outputs
test_pic/            # Test images
results/             # Prediction outputs (default)
output_results/      # Trained model weights + training outputs
CRACK500/            # CRACK500 dataset folder
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Training

Edit `train_config.yaml` and run:

```bash
python train.py --config train_config.yaml
```

Key config fields to edit:

- `preprocess`: `patch` (default), `letterbox`, or `stretch`
- `input_size`, `patches_per_image`, `val_stride`
- `no_augment` (true/false)
- `pos_weight`, `bce_weight`, `dice_weight`
- `metric_threshold`, `metric_thresholds`, `scheduler_metric`

Loss tuning for class imbalance (thin cracks / few positives):

Set `pos_weight`, `bce_weight`, `dice_weight` in `train_config.yaml`.

Safe speed-ups (no data/gradient changes):

- Cache preprocessing for letterbox/stretch (requires `no_augment: true`): set `cache_dir`
- Disable extra plots: set `no_visualize: true` and/or `no_loss_curve: true`

Model checkpoints are saved under `output_results/<timestamp>/` (default: `output_results/YYYYMMDD_HHMMSS/best_model.pth`).

## Prediction

### Predict a Single Image

```bash
python predict.py --image test_pic/your_image.jpg --model output_results/best_model.pth --output results --threshold 0.5
```

Args:

- `--image`: input image path
- `--model`: model weights path (default: `output_results/best_model.pth`)
- `--output`: output folder (default: `results`)
- `--threshold`: binarization threshold (default: `0.5`)
- `--no-postprocessing`: disable post-processing
- `--mode`: `tile` (default), `letterbox`, or `resize`
- `--input-size`: model input size (default: `256`)
- `--tile-overlap`: overlap between tiles (default: `input_size//2`)

Note: If `matplotlib` is not installed, `predict.py` still runs and saves `*_prediction.png` using a PIL-based fallback.

### Predict a Folder

```bash
python predict.py --input-dir test_pic --model output_results/best_model.pth --output results --threshold 0.5
```

Optional:

- `--recursive`: scan subfolders recursively
- `--no-postprocessing`: disable post-processing
- `--mode`: `tile` (default), `letterbox`, or `resize`
- `--input-size`: model input size (default: `256`)
- `--tile-overlap`: overlap between tiles (default: `input_size//2`)

### Output Files

For each input image:

- `{name}_prediction.png`: visualization (original / probability / overlay)
- `{name}_mask.png`: binary mask (white = crack)

## Evaluation (GroundTruth `.mat` or mask images)

Ground truth can be taken from CrackForest `.mat` files (the `Segmentation` field; default crack label = `2`) or from binary mask images (e.g. `*_mask.png`).

### Step 1: Run UNet on CrackForest images

```bash
python predict.py --input-dir compare/CrackForest-dataset/image --model output_results/best_model.pth --output compare/result_unet --threshold 0.5
```

### Step 2: Compute metrics and export Excel/CSV

```bash
# Excel (.xlsx)
python compare_result.py --pred-dir compare/result_unet --gt-dir compare/CrackForest-dataset/groundTruth --out-xlsx compare/compare_metrics.xlsx

# Optional CSV
python compare_result.py --pred-dir compare/result_unet --gt-dir compare/CrackForest-dataset/groundTruth --out-csv compare/compare_metrics.csv
```

Metrics per image: IoU, Dice, Precision, Recall, F1-score (+ TP/FP/FN/TN). The `.xlsx` includes `metrics` and `summary` sheets.

If your ground truth is a folder of mask images (e.g. `compare_test_dataset/masks/*.png`):

```bash
python compare_result.py --pred-dir compare/result_unet --gt-dir compare_test_dataset/masks --gt-suffix _mask.png --out-xlsx compare/compare_metrics.xlsx
```
