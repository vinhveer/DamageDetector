# Step 4 Class-Aware Deduplication

Runs learned-style, class-aware duplicate removal over Step 2 semantic detections and Step 3 cached DINOv2 embeddings.

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
python semi-labeling/step4_class_aware_dedup/dedup_detections.py \
  --source-db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3 \
  --embedding-db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step3_embedding/embeddings.sqlite3 \
  --output-db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step4_class_aware_dedup/dedup.sqlite3 \
  --semantic-run-id latest \
  --embedding-run-id latest
```

Main outputs are `dedup_runs`, `dedup_results`, and `dedup_pair_scores` in `dedup.sqlite3`.

Use `eval_dedup.py` with labeled pair/box CSV files after Phase 1 labeling.
