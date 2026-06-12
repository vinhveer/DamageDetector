# Semi-labeling client pipeline

Goal: **input = image folder**, **output = clean YOLO/COCO dataset**.

The client-facing pipeline has only one supported path:

```text
1. Detect   -> GDINO candidate boxes + seed labels
2. Filter   -> seed labels from GDINO detections + taxonomy + initial clean/review queues
3. Prepare  -> crop thumbnails + DINOv2 embeddings + visual domains/core clusters
4. Review   -> app/manual review + domain-first prototype/reject picks -> JSON handoff
5. Clean    -> apply prototype/review handoff + detector/prototype/core vote + strict policy
6. Export   -> YOLO/COCO dataset
```

OpenCLIP has been removed: step 2 seeds labels straight from the GroundingDINO
detector prompt instead of running a vision-language model.  Internal technical
modules still exist as implementation details, but the client path above is the
only supported way to create the dataset.

---

## Important files

```text
client_pipeline.py          # simple CLI: detect/filter/prepare/clean/export/status
tools/handoff.py            # apply JSON handoff requests from the app
tools/export_dataset.py     # export cleaned labels to YOLO/COCO
semilabel_app/              # review/prototype app
```

Default DB:

```text
<output-dir>/pipeline.sqlite3
```

Default dataset output:

```text
<output-dir>/dataset/<run-id>/
```

Default handoff JSON folder:

```text
<output-dir>/handoff/
```

---

## Setup

Run commands from:

```bash
cd /content/DamageDetector/semi-labeling
```

For Colab/GPU, allow model download if needed:

```bash
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
# optional, recommended if HuggingFace rate-limits:
export HF_TOKEN=<your_token>
```

Set common variables:

```bash
IMG=/content/HinhAnh
OUT=/content/out
RUN=myrun
```

---

## Recommended one-command automatic part

This runs:

```text
Detect -> Filter -> Prepare
```

```bash
python -m client_pipeline recommended \
  --input-dir $IMG \
  --output-dir $OUT \
  --run-id $RUN \
  --device cuda
```

After this, open the app for prototype/review, run `clean`, then export.

---

## Step-by-step commands

Use these if you want to run each step manually.

### Step 1 - Detect with GDINO

```bash
python -m client_pipeline detect \
  --input-dir $IMG \
  --output-dir $OUT \
  --device cuda
```

What it writes:

```text
$OUT/pipeline.sqlite3
```

Important defaults:

```text
--tiled-threshold 1024
--tile-size 1024
--tile-overlap 128
--tile-batch-size 8
--image-workers 2
--box-threshold 0.12
--nms-iou 0.45
--final-max-dets-per-class 400
--adaptive-duplicate-filter on
--duplicate-iou-threshold 0.0
```

For duplicate filtering, `0.0` means **auto-infer from the current image**. The
pipeline uses the image's own overlap distribution to suppress true overlapping
duplicate boxes by IoU only. Score is one part of the quality ranking;
shape/source/box-size priors are also used. Boxes that are merely contained
inside a larger box are **always kept** (a small box nested inside a broader
region is never dropped), so fine damage stays in the output even if the image
ends up densely filled with boxes.

To compare with the old fixed NMS-only behavior:

```bash
python -m client_pipeline detect \
  --input-dir $IMG \
  --output-dir $OUT \
  --device cuda \
  --no-adaptive-duplicate-filter
```

Overlay images are **off by default**. Only enable for debugging:

```bash
python -m client_pipeline detect \
  --input-dir $IMG \
  --output-dir $OUT \
  --device cuda \
  --save-overlays
```

### A100 tuning for Step 1

If A100 utilization is low, increase tile batch size:

```bash
python -m client_pipeline detect \
  --input-dir $IMG \
  --output-dir $OUT \
  --device cuda \
  --tile-batch-size 12 \
  --image-workers 2
```

If VRAM is still comfortable, try:

```bash
--tile-batch-size 16
```

If CUDA OOM happens, lower to:

```bash
--tile-batch-size 8
```

or:

```bash
--tile-batch-size 4
```

This tuning does not change prompt/threshold/NMS/tile size. It only runs more
tiles in one GDINO forward pass.

---

### Step 2 - Seed labels from GDINO detections + taxonomy

```bash
python -m client_pipeline filter \
  --input-dir $IMG \
  --output-dir $OUT \
  --run-id $RUN \
  --device cuda
```

What it does internally:

```text
Semantic Init reads GDINO detections directly (no OpenCLIP)
Seed labels come from the detector prompt (crack / mold / spall)
Taxonomy mapping: crack / mold / spall / reject
Initial cleaned_labels + review_queue
```

OpenCLIP has been removed. The seed label is the detector prompt label; the
later DINOv2 domain/core + human prototype steps are the main quality gate.
`--source-run-id` selects which damage_scan run to import (default: latest).

---

### Step 3 - Prepare review data

```bash
python -m client_pipeline prepare \
  --input-dir $IMG \
  --output-dir $OUT \
  --run-id $RUN \
  --device cuda
```

What it does internally:

```text
Crop generation: tight only
DINOv2 embedding
Core cluster mining
```

Default DINOv2 model:

```text
facebook/dinov2-giant
```

For quick test:

```bash
python -m client_pipeline prepare \
  --input-dir $IMG \
  --output-dir $OUT \
  --run-id $RUN \
  --device cuda \
  --dinov2-model facebook/dinov2-small \
  --embed-batch-size 128
```

---

### Step 4 - Review and prototype in the app

Open the semi-labeling app, then:

1. Review uncertain boxes.
2. Correct wrong labels.
3. Reject background/object/shadow boxes.
4. Pick prototypes for:

```text
crack
mold
spall
reject
```

Prototype selection is **domain-first**. The app groups candidates by DINOv2
visual domain/core cluster and spreads candidates across images. Score values are
shown only for audit/ranking hints; do not treat the reliability score as
truth. Prefer a few clean examples per domain over many near-duplicate examples
from the same image.

Recommended first pass:

```text
crack: 50-70 clean prototypes, spread across domains
mold : 70-90 clean prototypes, spread across domains
spall: 60-80 clean prototypes, spread across domains
reject: 100-120 true negatives/false positives, spread across images
```

#### JSON handoff rule

When the app commits review/prototype choices, it first creates a JSON request in:

```text
$OUT/handoff/
```

Then the app/tool reads that JSON and applies it.

This is intentional. It keeps human decisions clean and auditable.

Prototype JSON example:

```json
{
  "type": "prototype_request",
  "db": "/content/out/pipeline.sqlite3",
  "run_id": "myrun",
  "model_name": "facebook/dinov2-giant",
  "view_name": "tight",
  "prototypes": [
    {
      "resultId": 101,
      "label": "crack",
      "domainIndex": 0,
      "clusterId": "core_myrun_emb_ea96_tight_crack_000",
      "imageRelPath": "DSC0001.png"
    },
    {"resultId": 205, "label": "mold", "domainIndex": 1},
    {"resultId": 333, "label": "spall", "domainIndex": 2}
  ],
  "rejects": [
    {"resultId": 404, "label": "reject", "isReject": true, "imageRelPath": "DSC0002.png"}
  ],
  "run_seed": true,
  "run_policy": true
}
```

Review JSON example:

```json
{
  "type": "review_request",
  "db": "/content/out/pipeline.sqlite3",
  "run_id": "myrun",
  "reviewer": "vinh",
  "notes": "manual review batch 1",
  "decisions": [
    {
      "resultId": 10,
      "action": "manual_relabel",
      "previousLabel": "mold",
      "newLabel": "spall"
    },
    {
      "resultId": 11,
      "action": "manual_reject",
      "previousLabel": "crack",
      "newLabel": "reject"
    }
  ]
}
```

Apply a handoff JSON manually:

```bash
python -m tools.handoff \
  --request-json $OUT/handoff/prototype_${RUN}_YYYYMMDD_HHMMSS.json \
  --action prototype \
  --chain
```

For review:

```bash
python -m tools.handoff \
  --request-json $OUT/handoff/review_${RUN}_YYYYMMDD_HHMMSS.json \
  --action review
```

After prototype/review, run clean to refresh labels from seed + policy:

```bash
python -m client_pipeline clean \
  --output-dir $OUT \
  --run-id $RUN \
  --seed --policy
```

Clean is strict by default: low-priority/uncertain items remain in
`review_queue` instead of being exported as cleaned labels. The seed vote uses
detector + human prototype + DINOv2 core evidence.

---

### Step 5 - Export dataset

```bash
python -m client_pipeline export \
  --input-dir $IMG \
  --output-dir $OUT \
  --run-id $RUN \
  --format both
```

Default output:

```text
$OUT/dataset/$RUN/
```

Output layout:

```text
dataset/myrun/
  data.yaml
  classes.txt
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  annotations.coco.json
```

Default split:

```text
train,val,test = 0.8,0.1,0.1
```

Change split:

```bash
python -m client_pipeline export \
  --input-dir $IMG \
  --output-dir $OUT \
  --run-id $RUN \
  --format both \
  --split 0.9,0.1,0
```

Custom dataset dir:

```bash
python -m client_pipeline export \
  --input-dir $IMG \
  --output-dir $OUT \
  --run-id $RUN \
  --dataset-dir /content/final_dataset \
  --format both
```

---

## Status check

```bash
python -m client_pipeline status \
  --output-dir $OUT \
  --run-id $RUN
```

---

## Clean recommended command sequence

```bash
cd /content/DamageDetector/semi-labeling

IMG=/content/HinhAnh
OUT=/content/out
RUN=myrun

python -m client_pipeline detect  --input-dir $IMG --output-dir $OUT --device cuda
python -m client_pipeline filter  --input-dir $IMG --output-dir $OUT --run-id $RUN --device cuda
python -m client_pipeline prepare --input-dir $IMG --output-dir $OUT --run-id $RUN --device cuda

# Open app -> review -> create prototype/review JSON handoff -> app/tool applies it.

python -m client_pipeline clean --output-dir $OUT --run-id $RUN --seed --policy
python -m client_pipeline export --input-dir $IMG --output-dir $OUT --run-id $RUN --format both
```

---

## Single allowed path

The clean client path is the only supported path:

```text
CLI detect -> CLI filter -> CLI prepare -> app writes JSON handoff ->
tools.handoff/app applies JSON -> CLI clean -> CLI export
```

Do not run old free-form app steps, old overlay tools, or self-training/classifier
side flows for the client dataset pipeline.