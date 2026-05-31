# Kaggle Setup

This setup keeps the SQLite DB portable:

- `damage_scan` stores `images.path` as `name` by default.
- `step2_sematic` resolves real files from `--image-root` + `images.rel_path`.

## 1. Setup

```bash
bash DamageDetector/semi-labeling/step2_sematic/kaggle/setup_kaggle.sh
```

Environment overrides:

- `REPO_DIR` default: `/kaggle/working/DamageDetector`

## 2. Step 1: GroundingDINO scan

```bash
IMAGE_ROOT="/kaggle/input/my-images/HinhAnh" \
DB_PATH="/kaggle/working/damage_scan.sqlite3" \
CHECKPOINT="IDEA-Research/grounding-dino-base" \
bash DamageDetector/semi-labeling/step2_sematic/kaggle/run_damage_scan_kaggle.sh
```

Notes:

- The script uses `--store-image-path-mode name`.
- For multi-GPU DINO workers, set `SERVICE_DEVICE_IDS=0,1`.

## 3. Step 2: OpenCLIP semantic labeling

```bash
IMAGE_ROOT="/kaggle/input/my-images/HinhAnh" \
DB_PATH="/kaggle/working/damage_scan.sqlite3" \
MODEL_NAME="ViT-H-14" \
PRETRAINED="laion2b_s32b_b79k" \
BATCH_SIZE="16" \
bash DamageDetector/semi-labeling/step2_sematic/kaggle/run_openclip_semantic_kaggle.sh
```

## 4. Sharding for 2x T4

The current `step2_sematic` CLI supports:

- `--shard-index`
- `--num-shards`

Example logical split:

```bash
python DamageDetector/semi-labeling/step2_sematic/run_openclip_semantic.py \
  --db "/kaggle/working/damage_scan.sqlite3" \
  --image-root "/kaggle/input/my-images/HinhAnh" \
  --device cuda \
  --shard-index 0 \
  --num-shards 2
```

```bash
python DamageDetector/semi-labeling/step2_sematic/run_openclip_semantic.py \
  --db "/kaggle/working/damage_scan.sqlite3" \
  --image-root "/kaggle/input/my-images/HinhAnh" \
  --device cuda \
  --shard-index 1 \
  --num-shards 2
```

For concurrent runs, prefer one DB copy per shard to avoid SQLite write contention.

Recommended pattern:

1. Copy the base `damage_scan.sqlite3` to `shard0.sqlite3` and `shard1.sqlite3`
2. Run shard 0 against `shard0.sqlite3`
3. Run shard 1 against `shard1.sqlite3`
4. Merge semantic tables back into one final DB

Example merge:

```bash
python DamageDetector/semi-labeling/step2_sematic/merge_semantic_tables.py \
  --target-db "/kaggle/working/damage_scan_merged.sqlite3" \
  --source-db "/kaggle/working/shard0.sqlite3" \
  --source-db "/kaggle/working/shard1.sqlite3" \
  --latest-only \
  --clear-target
```
