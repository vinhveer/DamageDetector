# Semi-Label Detector Training

Utilities for training object detectors from semi-labeled COCO datasets and validating predictions by semantic coverage rather than strict ground-truth mAP.

## Dataset

Expected input:

```text
semi_labeling_coco/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── images/
    ├── train/
    └── val/
```

Prepare framework-specific files:

```bash
python -m object_detection.semi_training prepare-dataset \
  --coco-root /path/to/semi_labeling_coco \
  --link-mode symlink
```

This writes:

- `yolo/data.yaml`
- `stable_dino_dataset.yaml`

StableDINO-only prepare:

```bash
python -m object_detection.semi_training prepare-stable-dino \
  --coco-root /path/to/semi_labeling_coco
```

## Prototype Crops

```bash
python -m object_detection.semi_training build-prototypes \
  --coco-root /path/to/semi_labeling_coco \
  --output-dir /path/to/prototypes \
  --samples-per-class 32
```

## Semantic Validation Smoke

Use COCO annotations as pseudo-predictions to validate the report path without running a detector:

```bash
python -m object_detection.semi_training semantic-val \
  --coco-root /path/to/semi_labeling_coco \
  --split val \
  --limit 20 \
  --output-dir /tmp/semi_semantic_smoke
```

Add `--prototype-dir` and `--dinov2-checkpoint` to run actual DINOv2 similarity checks.

Semantic validation writes:

- `semantic_validation_report.json`
- `predictions.csv` with `sim_<class>`, `predicted_class_similarity`, `class_margin`, `class_agreement`
- `review_queue.csv`
- `preview/accepted`, `preview/low_similarity`, `preview/class_mismatch`, `preview/no_detection_images`
- `prototype_bank.pt` in the prototype directory by default, so prototype embeddings are cached across runs

Useful flags:

```bash
--decision-mode coverage              # accept any damage-like nearest prototype
--decision-mode class-consistency     # require detector class == nearest prototype class
--threshold 0.75
--reject-threshold 0.50
--margin-threshold 0.05
--expand-ratio 0.05
```

## YOLO

```bash
python -m object_detection.semi_training yolo train -- \
  --coco-root /path/to/semi_labeling_coco \
  --model yolo26n.pt \
  --epochs 100 \
  --imgsz 1024
```

```bash
python -m object_detection.semi_training yolo val -- \
  --model /path/to/best.pt \
  --coco-root /path/to/semi_labeling_coco \
  --output-dir /path/to/val_report \
  --conf 0.05
```

`yolo val` runs two validation passes by default:

- YOLO native `model.val()` for pseudo-mAP on the semi-label validation split.
- Low-confidence detector inference followed by DINOv2 semantic validation when prototypes are provided.

Main outputs:

- `native_map_report.json`
- `semantic_validation/semantic_validation_report.json`
- `unified_validation_report.json`

Use `--no-map` to skip the native mAP pass, or `--map-conf` to override Ultralytics validation confidence.

## StableDINO

```bash
python -m object_detection.semi_training stable-dino train -- \
  --coco-root /path/to/semi_labeling_coco \
  --finetune-checkpoint /path/to/dino_r50_4scale_12ep.pth \
  --imgsz 1024 \
  --batch-size 16 \
  --amp \
  --scale-lr-schedule \
  --max-iter 30000
```

Validation emphasizes whether predictions are damage-like and covered by DINOv2 prototypes. Pseudo-label mAP can still be logged, but it should not be presented as fully verified ground-truth performance.
