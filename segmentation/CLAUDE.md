# CLAUDE.md — segmentation

This file provides guidance to Claude Code when working in `segmentation/`.

## Vai trò của module

Module segmentation chứa **3 model pipeline hoàn toàn độc lập** cho crack segmentation:

| Model | Thư mục | Khi dùng |
|-------|---------|---------|
| **UNet** | `unet/` | Fast inference, fully supervised, cần ground-truth masks để train |
| **SAM zero-shot** | `sam/runtime/` (exposed qua `sam/no_finetune/`) | Không cần train, dùng box prompts từ DINO |
| **SAM-LoRA** | `sam/finetune/` | Kết hợp nền tảng SAM với domain adaptation qua LoRA/Adapter |

Cả 3 đều expose cùng interface qua RPC (`call(method, params, log_fn, stop_checker)`):
- `warmup` — load model vào memory
- `predict` — single image
- `predict_batch` — list of images
- `segment_boxes` — nhận DINO boxes, trả về masks
- `run_rois` — UNet-specific: chạy trên region of interest

---

## UNet

### Kiến trúc
- Framework: `segmentation_models_pytorch` (smp)
- Model: `smp.Unet(encoder_name="efficientnet-b4", in_channels=3, classes=1, decoder_attention_type="scse")`
- Output: sigmoid probability map [0,1], threshold để tạo binary mask

### Training

```bash
python -m segmentation.unet.train \
  --train-images ./data/train/images \
  --train-masks  ./data/train/masks \
  --val-images   ./data/val/images \
  --val-masks    ./data/val/masks \
  --output-dir   ./output \
  --encoder-name efficientnet-b4 \
  --preprocess   patch \        # patch | letterbox | resize
  --input-size   256 \
  --batch-size   16 \
  --epochs       80 \
  --augment-profile balanced \  # light | balanced | aggressive | outdomain
  --bce-weight 0.4 --dice-weight 0.6 \
  --best-model-metric best_iou \
  --early-stop-patience 15
```

**3 preprocessing modes:**
- `patch`: random crop patches (tốt nhất cho training, ảnh nhỏ)
- `letterbox`: resize giữ aspect ratio, pad (tốt cho validation)
- `resize`: simple resize (nhanh nhất)

**Loss function:** `BCE × bce_weight + Dice × dice_weight + Tversky × tversky_weight`

**Checkpoint format:**
```python
{
    "state_dict": OrderedDict(...),
    "model_config": {"arch": "Unet", "encoder_name": "efficientnet-b4", ...},
    "epoch": 42,
    "metrics": {"best_iou": 0.85, "best_dice": 0.89}
}
# Sidecar: training_config.json cùng thư mục với .pth
```

`model_io.py` load checkpoint theo thứ tự: checkpoint["model_config"] → sidecar training_config.json → DEFAULT_MODEL_CONFIG.

### Prediction

```bash
python -m segmentation.unet predict \
  --model ./best_model.pth \
  --image image.jpg \
  --output-dir ./results \
  --threshold 0.5 \
  --mode tile          # tile | letterbox | resize
```

**Tiled prediction flow** (quan trọng nhất):
```
predict_probabilities(image, mode="tile")
  → sliding window: step = tile_size - overlap
  → batch tiles → model(batch) → sigmoid
  → weighted accumulation: pred_sum / count_sum  ← tránh seams
  → return float32 probability map (H, W)
```

### Sử dụng qua inference_api

```python
# params nhận bởi UnetRunner.call()
{
    "image_path": "/path/to/img.jpg",
    "params": {
        "model_path": "/path/model.pth",
        "threshold": 0.5,
        "mode": "tile",
        "tile_size": 512,
        "tile_overlap": 64,
        "output_dir": "/results",
        "task_group": "crack_only"  # hoặc "more_damage"
    }
}
```

---

## SAM Zero-Shot (no_finetune / runtime)

### Cơ chế
- Dùng Meta SAM (ViT-B/L/H) không fine-tune
- **Auto Mask Generation (AMG)**: grid of points → tất cả masks → scoring → select
- **Box prompt mode**: nhận DINO boxes → refine thành mask chính xác hơn

### Custom scoring cho crack

`score_mask_for_crack()` trong `runtime/engine.py` tính score dựa trên:
1. **Elongation**: crack thường dài và hẹp (high elongation → high score)
2. **Thinness**: perimeter² / area (crack có tỷ lệ cao)
3. **Fill ratio**: crack không fill toàn bộ bbox

```bash
python -m segmentation.sam no_finetune predict \
  --checkpoint ./sam_vit_h_4b8939.pth \
  --sam-model-type vit_h \
  --image image.jpg \
  --output-dir ./results_sam
```

### Profiles
- `FAST`: 16 points/side — nhanh, ít memory
- `QUALITY`: 24 points/side
- `ULTRA`: 32 points/side — chậm nhất, tốt nhất

---

## SAM-LoRA Finetune

### Delta modules

SAM gốc được frozen, chỉ thêm **delta layers** vào Image Encoder:
- `sam_lora_image_encoder.py`: Low-rank adaptation (LoRA rank=4 by default)
- `sam_adapter_image_encoder.py`: Bottleneck adapter (middle_dim, scaling_factor)
- `sam_adapter_lora_image_encoder.py`: Cả hai cùng lúc

```python
# runtime.py: apply_delta_to_sam()
apply_delta_to_sam(sam_model, delta_type="lora", delta_checkpoint="delta.pth", rank=4)
# → inject LoRA/Adapter vào ViT blocks của SAM image encoder
```

### Training

```bash
python -m segmentation.sam.finetune.train \
  --root_path  ./train_data \   # thư mục có images/ và masks/ bên trong
  --val_path   ./val_data \
  --output     ./output/training \
  --vit_name   vit_h \
  --ckpt       ./checkpoints/sam_vit_h_4b8939.pth \
  --delta_type lora \
  --rank       4 \
  --base_lr    0.001 \
  --batch_size 12 \
  --max_epochs 150 \
  --augment_profile balanced \
  --bce_weight 1.0 --dice_weight 0.35 --tversky_weight 0.35 \
  --use_amp    # mixed precision
```

**Dataset format:** thư mục có `images/*.jpg` và `masks/*.png` (binary masks, white = crack).

**Prompt policy** (`trainer.py`):
- `hybrid`: kết hợp box prompt + random points (tốt nhất)
- `legacy`: nhiều points hơn, ít box hơn

### Inference modes (tiled_inference.py)

```
"tile_full_box":   tiled DINO → SAM trên từng tile, merge masks
"coarse_refine":   coarse prediction → tìm ROIs → refine với SAM
```

### Threshold file

Sau training, `test.py` sweep threshold và lưu `best_threshold.txt` cùng thư mục với delta checkpoint. `SamFinetuneRunner` đọc file này khi predict nếu không truyền threshold.

---

## Datasets (shared)

### Augmentation profiles (datasets/core/augment.py)

`build_crack_profile_augment(profile_name)` trả về Albumentations pipeline:
- `light`: flips + minor brightness/contrast
- `balanced`: flips + rotate + affine + moderate color jitter  ← default
- `aggressive`: thêm noise, blur, dropout, compression artifacts
- `outdomain`: mô phỏng ảnh từ domain khác (extreme color shifts)

Khi sửa augmentation, **không hardcode** vào dataset class mà sửa profile trong `augment.py`.

### CrackDataset (datasets/unet/crack_dataset.py)

High-performance UNet dataset:
- RAM caching: load tất cả ảnh/masks vào memory lần đầu
- Multi-patching: mỗi ảnh cho ra nhiều patches (giảm disk I/O)
- Crop policy: `smart` (ưu tiên patches có crack), `random`, `fast`

### GenericDataset (datasets/sam_finetune/dataset.py)

SAM finetune dataset:
- Đọc từ `images/` và `masks/` folders
- `prompts.py`: tạo SAM prompt tensors (box_coords, point_coords, point_labels)
- Không cache memory vì ảnh SAM thường lớn hơn

---

## CLI Quick Reference

```bash
# UNet
python -m segmentation.unet.train  [--train-images ...] [--val-images ...] [--output-dir ...]
python -m segmentation.unet predict [--model ...] [--image ...] [--threshold ...]
python -m segmentation.unet predict-batch [--model ...] [--images ...]
python -m segmentation.unet run-rois [--model ...] [--image ...] [--roi-box x1 y1 x2 y2]

# SAM zero-shot
python -m segmentation.sam no_finetune predict [--checkpoint ...] [--image ...]
python -m segmentation.sam no_finetune segment-boxes [--checkpoint ...] [--image ...] [--boxes-json ...]

# SAM finetune training
python -m segmentation.sam.finetune.train [--root_path ...] [--ckpt ...] [--delta_type lora]
python -m segmentation.sam.finetune.test [--dataset_path ...] [--sam_ckpt ...] [--delta_ckpt ...]
python -m segmentation.sam.finetune.pseudo_label [--input_dir ...] [--output_dir ...]

# SAM finetune inference
python -m segmentation.sam finetune predict [--checkpoint ...] [--delta-checkpoint ...]
```

---

## Lưu ý khi sửa code

- **UNet model config** được lưu trong checkpoint — nếu thay đổi kiến trúc, phải bump version hoặc thêm migration logic trong `model_io.py`.
- **SAM-LoRA delta layers** inject vào từng ViT block của image encoder — số lượng blocks phụ thuộc `vit_name` (vit_b: 12 blocks, vit_h: 32 blocks). Rank không thể thay đổi sau khi train.
- **Tiled prediction** của cả UNet lẫn SAM-LoRA dùng **weighted accumulation** (không phải max) để blend tile boundaries — đừng thay bằng simple overwrite.
- `datasets/core/augment.py` là nguồn sự thật cho augmentation — không copy augmentation code vào các class khác.
- Positive class weight trong UNet training được tự động ước tính từ mask stats nếu không truyền `--pos-weight`.
