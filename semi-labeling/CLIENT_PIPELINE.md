# Client semi-labeling pipeline

Goal: input image folder -> cleaned YOLO/COCO dataset.

## Recommended flow

```bash
cd DamageDetector/semi-labeling

# Run the automatic part: Step 1 Detect -> Step 2 Filter -> Step 3 Prepare Review
python -m client_pipeline recommended \
  --input-dir /content/HinhAnh \
  --output-dir /content/out \
  --run-id myrun \
  --device cuda

# Open the app, review labels, pick damage/reject prototypes, commit corrections.

# Optional after prototypes/review: refresh labels and clean/review queue.
python -m client_pipeline clean \
  --output-dir /content/out \
  --run-id myrun \
  --seed --policy

# Export final dataset.
python -m client_pipeline export \
  --input-dir /content/HinhAnh \
  --output-dir /content/out \
  --run-id myrun \
  --format both
```

Default DB: `<output-dir>/pipeline.sqlite3`.

Default dataset output: `<output-dir>/dataset/<run-id>`.

## Run step-by-step

```bash
# Step 1: GDINO detection
python -m client_pipeline detect --input-dir /content/HinhAnh --output-dir /content/out --device cuda

# Step 2: OpenCLIP semantic filter + initial clean/review tables
python -m client_pipeline filter --input-dir /content/HinhAnh --output-dir /content/out --run-id myrun --device cuda

# Step 3: crop + DINOv2 embedding + core clusters
python -m client_pipeline prepare --input-dir /content/HinhAnh --output-dir /content/out --run-id myrun --device cuda

# Status
python -m client_pipeline status --output-dir /content/out --run-id myrun

# Export
python -m client_pipeline export --input-dir /content/HinhAnh --output-dir /content/out --run-id myrun
```

## A100 GDINO speed tuning

The client `detect` step defaults to `--tile-batch-size 8` and `--image-workers 2`.
This keeps the same tiled+full-image detection logic but pushes multiple tiles
through one GroundingDINO forward pass, which uses large GPUs better.

If A100 VRAM/GPU utilization is still low, try:

```bash
python -m client_pipeline detect \
  --input-dir /content/HinhAnh \
  --output-dir /content/out \
  --device cuda \
  --tile-batch-size 12 \
  --image-workers 2
```

If VRAM is still comfortable, try `--tile-batch-size 16`. If you hit OOM, lower
to `8` or `4`.

Overlay images are off by default to keep the client pipeline clean and fast. If
you need visual debugging, add `--save-overlays` to the detect command.

## Simple client names

1. Detect: GDINO candidate boxes.
2. Filter: OpenCLIP + taxonomy + initial clean/review queue.
3. Prepare: crop thumbnails + DINOv2 embeddings + core clusters.
4. Review: app/manual prototype/reject corrections.
5. Export: YOLO/COCO dataset with train/val/test split.

Advanced internal steps (reliability, classifier, self-training) remain optional and are not required for the first clean dataset.