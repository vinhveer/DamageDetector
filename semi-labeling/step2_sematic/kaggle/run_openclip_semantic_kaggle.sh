#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="${IMAGE_ROOT:?Set IMAGE_ROOT}"
DB_PATH="${DB_PATH:-/kaggle/working/damage_scan.sqlite3}"
MODEL_NAME="${MODEL_NAME:-ViT-H-14}"
PRETRAINED="${PRETRAINED:-laion2b_s32b_b79k}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_CROPS="${SAVE_CROPS:---no-save-crops}"

python DamageDetector/semi-labeling/step2_sematic/run_openclip_semantic.py \
  --db "$DB_PATH" \
  --image-root "$IMAGE_ROOT" \
  --limit "$LIMIT" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --model-name "$MODEL_NAME" \
  --pretrained "$PRETRAINED" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  "$SAVE_CROPS"
