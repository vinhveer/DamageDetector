<h3>Code is developed from SAMed: https://github.com/hitachinsk/SAMed</h3>


<h3>1. Download pretrained SAM weights (ViT-B, ViT-L, ViT-H) from https://github.com/facebookresearch/segment-anything and put it in the "checkpoints" folder</h3> 

<h3>2. Train:</h3>

Dataset layout is generic-only and must contain:

```text
dataset_root/
  images/
  masks/
```

Training now uses:
- hybrid prompt policy by default
- auto `pos_weight`
- BCE + Dice + Tversky + Focal
- tiled full-image validation as the primary model-selection metric
- optional `HQ + LoRA` mode for sharper boundary/detail reconstruction

```bash
python -m segmentation.sam.finetune.train --root_path /path/to/train --val_path /path/to/val --warmup --AdamW --img_size 512 --n_gpu 1 --batch_size 8 --base_lr 0.0004 --warmup_period 300 --tf32 --use_amp --max_epochs 140 --stop_epoch 100 --vit_name vit_b --ckpt /path/to/sam_vit_b.pth --delta_type lora --rank 4 --patches_per_image 4 --background_crop_prob 0.2 --near_background_crop_prob 0.15 --prompt_policy hybrid_val_balanced --pos_weight auto --bce_weight 1.0 --dice_weight 0.35 --tversky_weight 0.35 --focal_weight 0.25 --focal_alpha 0.25 --focal_gamma 2.0 --val_thresholds 0.35 0.4 0.45 0.5 0.55 0.6 --tile_overlap 256 --save_interval 1 --run_name baseline_a
```

`HQ + LoRA` benchmark run:

```bash
python -m segmentation.sam.finetune.train --root_path /path/to/train --val_path /path/to/val --warmup --AdamW --img_size 512 --n_gpu 1 --batch_size 8 --base_lr 0.0004 --warmup_period 300 --tf32 --use_amp --max_epochs 140 --stop_epoch 100 --vit_name vit_b --ckpt /path/to/sam_vit_b.pth --delta_type lora --rank 4 --decoder_type hq --patches_per_image 4 --background_crop_prob 0.2 --near_background_crop_prob 0.15 --prompt_policy hybrid_val_balanced --pos_weight auto --bce_weight 1.0 --dice_weight 0.35 --tversky_weight 0.35 --focal_weight 0.25 --focal_alpha 0.25 --focal_gamma 2.0 --val_thresholds 0.35 0.4 0.45 0.5 0.55 0.6 --tile_overlap 256 --save_interval 1 --run_name hq_lora_a
```

<h3>3. Test:</h3>

`test.py` now defaults to tiled full-image evaluation/inference-realistic metrics. Legacy crop + GT-box evaluation is still available with `--eval_mode legacy_full_box` or `--legacy_box_eval`.

```bash
python -m segmentation.sam.finetune.test --volume_path /path/to/test --img_size 512 --ckpt /path/to/sam_vit_b.pth --vit_name vit_b --delta_type lora --rank 4 --delta_ckpt /path/to/best_model.pth --eval_mode tile_full_box --tile_size 512 --tile_overlap 256 --pred_threshold auto --val_thresholds 0.35 0.4 0.45 0.5 0.55 0.6 --is_savenii
```

For `HQ + LoRA`, add `--decoder_type hq` to both train and test so the same decoder is rebuilt before loading the delta checkpoint.

<h3>4. Engine / CLI inference:</h3>

`python -m segmentation.sam.finetune predict` now defaults to tiled full-box inference. If `best_threshold.txt` exists next to the delta checkpoint, `--threshold auto` will use it automatically.
