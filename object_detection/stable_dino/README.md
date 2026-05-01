# StableDINO Training Notes

This wrapper trains the ported StableDINO project under `projects/stabledino` on a custom damage dataset. For crack datasets, prefer fine-tuning from a detector checkpoint instead of starting from the default ImageNet R50 backbone.

## Recommended R50 Fine-tune

Download a detrex DINO/StableDINO R50 4-scale COCO checkpoint, then run:

```bash
python -m object_detection.stable_dino.train \
  --dataset /path/to/dataset.yaml \
  --output-dir /path/to/output/stable_dino_r50_coco_ft \
  --finetune-checkpoint /path/to/dino_r50_4scale_12ep.pth \
  --imgsz 512 \
  --batch-size 8 \
  --workers 2 \
  --max-iter 4000 \
  --eval-period 1000 \
  --checkpoint-period 1000 \
  --augmentation-profile balanced \
  --test-with-nms 0.8
```

The fine-tune loader skips `class_embed` and `label_enc` by default, plus tensors whose shape does not match the 1-class damage model. This keeps backbone, transformer, and box heads from the COCO checkpoint while reinitializing class-specific heads.


## Smoke Checks

Before a long run, validate checkpoint loading and data registration with a short train:

```bash
python -m object_detection.stable_dino.train \
  --dataset /path/to/dataset.yaml \
  --output-dir /path/to/output/smoke_300 \
  --finetune-checkpoint /path/to/dino_r50_4scale_12ep.pth \
  --max-iter 300 \
  --eval-period 300 \
  --checkpoint-period 300 \
  --batch-size 8 \
  --workers 2 \
  --test-with-nms 0.8
```

To check whether the model can fit the training set, evaluate a saved checkpoint on train split:

```bash
python -m object_detection.stable_dino.train \
  --dataset /path/to/dataset.yaml \
  --output-dir /path/to/output/eval_train \
  --init-checkpoint /path/to/model_best.pth \
  --eval-only \
  --eval-split train \
  --test-with-nms 0.8
```

## Quick Experiments

Run short experiments before a long train:

```bash
# Balanced augmentation, default LR.
python -m object_detection.stable_dino.train --dataset /path/to/dataset.yaml --output-dir /path/to/out_balanced --finetune-checkpoint /path/to/dino_r50_4scale_12ep.pth --max-iter 3000 --eval-period 1000 --checkpoint-period 1000 --batch-size 8 --workers 2 --augmentation-profile balanced

# Lighter augmentation for thin cracks.
python -m object_detection.stable_dino.train --dataset /path/to/dataset.yaml --output-dir /path/to/out_light --finetune-checkpoint /path/to/dino_r50_4scale_12ep.pth --max-iter 3000 --eval-period 1000 --checkpoint-period 1000 --batch-size 8 --workers 2 --augmentation-profile light

# Lower LR through LazyConfig override.
python -m object_detection.stable_dino.train --dataset /path/to/dataset.yaml --output-dir /path/to/out_light_lr5e5 --finetune-checkpoint /path/to/dino_r50_4scale_12ep.pth --max-iter 3000 --eval-period 1000 --checkpoint-period 1000 --batch-size 8 --workers 2 --augmentation-profile light optimizer.lr=5e-5
```

Use `model_best.pth` for comparison. `model_final.pth` can be worse if validation AP peaks before the last iteration.

## Notes

- Keep `--imgsz 512` for 512px source images unless you intentionally upsample.
- On one T4, use R50 as the main StableDINO baseline. Swin-T is only a secondary experiment with small batch size; Swin-S and larger are not practical for fair training on T4.
- If training looks suspicious, run `--eval-split train --eval-only --init-checkpoint /path/to/checkpoint.pth` to distinguish underfitting from validation/data issues.
