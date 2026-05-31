# Step 3: DINOv2 Embedding Cache

Embed all successful Step 2 detections once with DINOv2 and persist normalized float32 vectors to SQLite. Downstream steps should read this cache instead of re-embedding crops.

## Run

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
python semi-labeling/step3_embedding/embed_detections.py --resume
```


Default paths:

```text
source DB: /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3
image root: /Users/nguyenquangvinh/Desktop/Lab/HinhAnh
output DB: /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step3_embedding/embeddings.sqlite3
model: facebook/dinov2-giant
```

Smoke test:

```bash
python semi-labeling/step3_embedding/embed_detections.py \
  --limit 100 \
  --output-db /tmp/step3_embeddings_test.sqlite3 \
  --force
```

## Resume Rules

- Same `model_name` + same resolved Step 2 `semantic_run_id` will not re-run by default.
- Use `--resume` to continue the latest matching run and skip existing `result_id` rows.
- Use `--force` to create a new embedding run.

## Downstream API

```python
from step3_embedding.cache_reader import load_embeddings

embs, ids = load_embeddings(Path("/path/to/embeddings.sqlite3"))
```

Returned embeddings are `np.float32`, L2-normalized, and ordered by `result_id` unless a `result_ids` filter is provided.
