#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="${IMAGE_ROOT:?Set IMAGE_ROOT}"
DB_PATH="${DB_PATH:-/kaggle/working/damage_scan.sqlite3}"
CHECKPOINT="${CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
SERVICE_WORKERS="${SERVICE_WORKERS:-2}"
SERVICE_QUEUE_SIZE="${SERVICE_QUEUE_SIZE:-32}"
SERVICE_BATCH_SIZE="${SERVICE_BATCH_SIZE:-8}"
SERVICE_DEVICE_IDS="${SERVICE_DEVICE_IDS:-0,1}"
IMAGE_WORKERS="${IMAGE_WORKERS:-2}"
LIMIT="${LIMIT:-0}"

python -m object_detection.damage_scan \
  --input-dir "$IMAGE_ROOT" \
  --db "$DB_PATH" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --recursive \
  --limit "$LIMIT" \
  --image-workers "$IMAGE_WORKERS" \
  --service-workers "$SERVICE_WORKERS" \
  --service-queue-size "$SERVICE_QUEUE_SIZE" \
  --service-batch-size "$SERVICE_BATCH_SIZE" \
  --service-device-ids "$SERVICE_DEVICE_IDS" \
  --store-image-path-mode name \
  --save-overlays
