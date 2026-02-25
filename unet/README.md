# Crack Detection U-Net

## Overview

This project implements crack segmentation using an improved U-Net architecture (with attention mechanisms). It supports:

- Training on datasets like `CRACK500`
- Predicting a single image or a whole folder (`predict.py`)
- Evaluating predicted masks vs CrackForest groundTruth `.mat` annotations and exporting metrics (`compare_result.py`)

## Project Structure

```
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

Run `train.py` with command-line arguments:

```bash
python train.py --train-images path/to/images --train-masks path/to/masks --val-images path/to/val_images --val-masks path/to/val_masks
```

### Arguments

#### Dataset Paths (Required)
- `--train-images`: Path to training images folder.
- `--train-masks`: Path to training masks folder.
- `--val-images`: Path to validation images folder.
- `--val-masks`: Path to validation masks folder.
- `--mask-prefix`: Prefix for mask files (default: `auto`).

#### Output & Logging
- `--output-dir`: Directory to save results (default: `output_results`).
- `--visualize`: Enable visualization (default: disabled).
- `--loss-curve`: Enable loss curve plotting (default: disabled).
- `--visualize-every`: Visualize every N epochs (0 to disable, default: 0).

#### Preprocessing
- `--preprocess`: Method: `patch`, `letterbox`, `stretch` (default: `patch`).
- `--input-size`: Model input size (default: 256).
- `--patches-per-image`: Number of patches per image (default: 1).
- `--max-patch-tries`: Max tries to find a patch with crack (default: 5).
- `--val-stride`: Stride for validation patching (default: 0 = input_size).

#### Augmentation
- `--no-augment`: Disable augmentation.
- `--aug-prob`: Augmentation probability (default: 0.5).
- `--rotate-limit`: Rotation limit in degrees (default: 10).
- `--brightness-limit`: Brightness limit (default: 0.2).
- `--contrast-limit`: Contrast limit (default: 0.2).

#### Caching
- `--cache-data`: Cache data in memory (default: false).
- `--cache-dir`: Directory to cache preprocessed images.
- `--cache-rebuild`: Rebuild cache.

#### Training Hyperparameters
- `--batch-size`: Batch size (default: 16).
- `--epochs`: Number of epochs (default: 80).
- `--learning-rate`: Learning rate (default: 0.0005).
- `--weight-decay`: Weight decay (default: 0.00001).
- `--seed`: Random seed (default: 42).
- `--early-stop-patience`: Early stopping patience (default: 15).
- `--grad-accum-steps`: Gradient accumulation steps (default: 1).
- `--num-workers`: Number of workers (default: 8).
- `--prefetch-factor`: Prefetch factor (default: 2).
- `--no-persistent-workers`: Disable persistent workers (default: enabled).
- `--pin-memory`: Enable pin memory (default: disabled).

#### Model
- `--encoder-name`: Encoder name (default: `efficientnet-b4`).
- `--encoder-weights`: Encoder weights (default: `imagenet`).

#### Loss Weights
- `--pos-weight`: Positive class weight (default: 5.0).
- `--bce-weight`: BCE loss weight (default: 0.4).
- `--dice-weight`: Dice loss weight (default: 0.6).
- `--focal-weight`: Focal loss weight (default: 0.0).
- `--focal-alpha`: Focal loss alpha (default: 0.25).
- `--focal-gamma`: Focal loss gamma (default: 2.0).

#### Metrics & Scheduler
- `--metric-threshold`: Threshold for metrics (default: 0.5).
- `--metric-thresholds`: Comma-separated thresholds (default: "").
- `--scheduler-metric`: Metric for scheduler (default: `loss`).
- `--scheduler-factor`: Scheduler factor (default: 0.5).
- `--scheduler-patience`: Scheduler patience (default: 10).
- `--scheduler-t0`: Scheduler T0 (default: 10).
- `--scheduler-tmult`: Scheduler T_mult (default: 2).

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

## Example Train Command (Kaggle)
```
!torchrun --standalone --nproc_per_node=2 unet/train.py --train-images "/kaggle/input/datasets/vinhnquntu/crack500croped/crack500_crop/train/images" --train-masks "/kaggle/input/datasets/vinhnquntu/crack500croped/crack500_crop/train/masks" --val-images "/kaggle/input/datasets/vinhnquntu/crack500croped/crack500_crop/val/images" --val-masks "/kaggle/input/datasets/vinhnquntu/crack500croped/crack500_crop/val/masks" --output-dir output_results_p4000 --preprocess letterbox --preprocess-train random_crop --preprocess-val patch --input-size 512 --patches-per-image 4 --encoder-name tu-convnext_tiny --encoder-weights imagenet --batch-size 4 --grad-accum-steps 4 --epochs 100 --seed 42 --aug-prob 0.5 --rotate-limit 10 --brightness-limit 0.2 --contrast-limit 0.2 --pos-weight 2.0 --pos-weight-min 2.0 --pos-weight-max 20.0 --pos-weight-sample 200 --bce-weight 1.0 --dice-weight 0.5 --focal-weight 0.5 --focal-alpha 0.25 --focal-gamma 2.0 --metric-threshold 0.5 --metric-thresholds "0.2,0.3,0.4,0.5,0.6" --scheduler-metric loss --learning-rate 0.0001 --weight-decay 0.01 --scheduler-t0 10 --scheduler-tmult 2 --early-stop-patience 100 --num-workers 2 --pin-memory --prefetch-factor 8
```