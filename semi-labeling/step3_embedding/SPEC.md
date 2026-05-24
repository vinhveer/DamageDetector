# Step 3 — Detection Embedding Cache

## 1. Mục đích

Embed **toàn bộ detection** từ Step 2 bằng DINOv2-giant **một lần duy nhất**, lưu vào SQLite. Mọi step downstream (Step 4 dedup, Step 5 clustering, Step 6 prototype review) chỉ đọc cache — không tự embed.

**Lợi ích:**
- Một lần embed, dùng cho 3 step → tiết kiệm ~2× GPU work.
- Step 4-6 không phụ thuộc PyTorch/HuggingFace, chỉ cần `numpy` + `sqlite3`.
- Đổi embedding model chỉ cần rerun Step 3.
- Cache là first-class artifact → có thể version, audit, share.

## 2. Đầu vào / Đầu ra

**Input**
- `infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3` (Step 2 output)
- Folder ảnh gốc (`HinhAnh/` hoặc tương đương)

**Output**
- `infer_results/semi-labeling/step3_embedding/embeddings.sqlite3`

## 3. CLI

```bash
python embed_detections.py \
  --source-db ../../infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3 \
  --semantic-run-id latest \
  --image-root /Users/nguyenquangvinh/Desktop/Lab/HinhAnh \
  --output-db ../../infer_results/semi-labeling/step3_embedding/embeddings.sqlite3 \
  --model-name facebook/dinov2-giant \
  --device auto \
  --batch-size 16 \
  --padding-ratio 0.05 \
  --min-confidence-pct 0.0 \
  --labels crack,spall,mold \
  --log-every 256
```

Tất cả flag có default — chạy không cần override gì cho dataset hiện tại.

| Flag | Default | Mô tả |
|---|---|---|
| `--source-db` | `step2_sematic/damage_scan.sqlite3` | Step 2 output |
| `--semantic-run-id` | `latest` | Run ID của Step 2 (semantic verify) |
| `--image-root` | `HinhAnh/` | Folder ảnh gốc |
| `--output-db` | `step3_embedding/embeddings.sqlite3` | Output |
| `--model-name` | `facebook/dinov2-giant` | HuggingFace model |
| `--device` | `auto` | cuda / mps / cpu (auto select) |
| `--batch-size` | `16` | Batch size cho embedder |
| `--padding-ratio` | `0.05` | Padding 5% mỗi cạnh khi crop |
| `--min-confidence-pct` | `0.0` | Min OpenCLIP confidence (% scale) để embed |
| `--labels` | `crack,spall,mold` | Labels cần embed |
| `--log-every` | `256` | In progress mỗi N detection |
| `--resume` | `false` | Skip detection đã có embedding |
| `--force` | `false` | Cho phép embed lại dù đã có run cùng model + semantic_run_id |
| `--limit` | `0` | 0 = không giới hạn; > 0 = chỉ embed N đầu tiên (test) |

## 4. DB Schema

```sql
CREATE TABLE embedding_runs (
    embedding_run_id        TEXT PRIMARY KEY,    -- uuid4 hex
    created_at_utc          TEXT NOT NULL,
    source_db_path          TEXT NOT NULL,
    source_semantic_run_id  TEXT NOT NULL,
    model_name              TEXT NOT NULL,        -- "facebook/dinov2-giant"
    dim                     INTEGER NOT NULL,     -- 1536
    device                  TEXT NOT NULL,        -- "cuda" | "mps" | "cpu"
    padding_ratio           REAL NOT NULL,
    total_detections        INTEGER NOT NULL,     -- tổng input từ Step 2
    embedded_count          INTEGER NOT NULL,
    skipped_count           INTEGER NOT NULL,
    options_json            TEXT NOT NULL         -- full args để reproduce
);

CREATE TABLE detection_embeddings (
    embedding_run_id   TEXT NOT NULL,
    result_id          INTEGER NOT NULL,           -- FK -> step2.detections.result_id
    image_rel_path     TEXT NOT NULL,              -- redundant, tiện query
    predicted_label    TEXT NOT NULL,              -- redundant
    embedding_blob     BLOB NOT NULL,              -- float32 LE, `dim` values, L2-normalized
    PRIMARY KEY (embedding_run_id, result_id)
);

CREATE INDEX idx_embeddings_image ON detection_embeddings (embedding_run_id, image_rel_path);
CREATE INDEX idx_embeddings_label ON detection_embeddings (embedding_run_id, predicted_label);

CREATE TABLE skipped_detections (
    embedding_run_id  TEXT NOT NULL,
    result_id         INTEGER NOT NULL,
    reason            TEXT NOT NULL,                -- "image_not_found" | "invalid_bbox" | "decode_error"
    detail            TEXT,                         -- optional message
    PRIMARY KEY (embedding_run_id, result_id)
);
```

## 5. Pseudo-code

```python
def main(args):
    semantic_run_id = resolve_semantic_run_id(args.source_db, args.semantic_run_id)
    detections = read_detections(
        args.source_db,
        semantic_run_id=semantic_run_id,
        min_confidence_pct=args.min_confidence_pct,
        labels=args.labels,
        limit=args.limit,
    )

    if args.resume:
        already_done = read_existing_result_ids(args.output_db, args.model_name)
        detections = [d for d in detections if d.result_id not in already_done]

    embedder = DinoV2Embedder(model_name=args.model_name, device=args.device)

    ensure_schema(args.output_db)
    embedding_run_id = uuid4().hex
    insert_run_metadata(args.output_db, embedding_run_id, args, embedder.dim)

    skipped, embedded_count = [], 0
    for batch in chunks(detections, args.batch_size):
        crops_ok, dets_ok, skipped_batch = robust_crop_batch(
            batch, args.image_root, padding_ratio=args.padding_ratio
        )
        skipped.extend(skipped_batch)

        if not crops_ok:
            continue

        embeddings = embedder.embed(crops_ok, batch_size=args.batch_size)
        # embeddings: torch.Tensor (N, 1536), L2-normalized

        rows = []
        for det, emb in zip(dets_ok, embeddings.cpu().numpy().astype(np.float32)):
            rows.append((
                embedding_run_id, det.result_id, det.image_rel_path,
                det.predicted_label, emb.tobytes(),
            ))
        bulk_insert(args.output_db, "detection_embeddings", rows)
        embedded_count += len(rows)

        if embedded_count % args.log_every < len(rows):
            print(f"[embed] {embedded_count}/{len(detections)}")

    if skipped:
        bulk_insert(
            args.output_db, "skipped_detections",
            [(embedding_run_id, s.result_id, s.reason, s.detail) for s in skipped],
        )

    update_run_counts(args.output_db, embedding_run_id, embedded_count, len(skipped))
    print(f"Done: {embedded_count} embedded, {len(skipped)} skipped")
```

## 6. Hành vi quan trọng

| Behavior | Spec |
|---|---|
| **Crop padding** | 5% mỗi cạnh, clip về `[0, W]` và `[0, H]` ảnh gốc |
| **Bad images** | Skip individual + log `skipped_detections`, không crash batch |
| **Resume** | Đọc `detection_embeddings` đã có cho cùng `model_name`, skip những `result_id` đã embed |
| **L2 normalize** | Output đã normalize → cosine similarity = dot product |
| **Float32** | Đủ precision, half storage so float64 |
| **Reproducibility** | Mọi args lưu trong `options_json` → rerun cùng kết quả |
| **No re-embed** | Nếu cùng model + cùng semantic_run_id đã có run, dừng + warn, trừ khi `--force` |
| **Atomic commit** | Mỗi batch commit 1 transaction → crash giữa chừng không corrupt |

## 7. API cho downstream đọc cache

File `cache_reader.py`:

```python
from pathlib import Path
import numpy as np

def load_embeddings(
    db_path: Path,
    *,
    model_name: str = "facebook/dinov2-giant",
    result_ids: list[int] | None = None,
    embedding_run_id: str | None = None,   # default: latest cho model_name
) -> tuple[np.ndarray, list[int]]:
    """
    Returns (embeddings, result_ids):
      embeddings: float32 array shape (N, dim)
      result_ids: list[int] length N, parallel to embeddings rows

    Nếu result_ids filter được cung cấp, trả về đúng thứ tự yêu cầu.
    """

def latest_embedding_run(db_path: Path, model_name: str) -> str:
    """Trả embedding_run_id mới nhất cho model_name."""

def embedding_run_metadata(db_path: Path, embedding_run_id: str) -> dict:
    """Trả full metadata của run (model, dim, options_json, counts)."""
```

Step 4-6 import qua:
```python
from step3_embedding.cache_reader import load_embeddings
embs, ids = load_embeddings(STEP3_DB)
```

## 8. Kích cỡ kỳ vọng

| Item | Số |
|---|---|
| Detection input | ~87,495 (sau filter confidence) |
| Embedded | ~85,000 (skip vài ảnh missing) |
| DB size | ~520 MB (87k × 1536 × 4 byte + overhead) |
| GPU time (CUDA) | ~12 phút |
| GPU time (MPS, Mac M-series) | ~25 phút |
| CPU fallback | nên tránh |

## 9. Test plan

1. **Smoke**: `--limit 100` → embed 100 detection, kiểm tra DB schema + BLOB shape.
2. **L2 norm**: random sample 50 embedding → `np.linalg.norm == 1.0 ± 1e-5`.
3. **Cache reader**: `load_embeddings(filter=[id1, id2, id3])` → shape `(3, 1536)`, đúng thứ tự.
4. **Resume**: chạy 2 lần, lần 2 với `--resume` → log "skipped X already-embedded".
5. **Bad image**: xóa 1 file ảnh → run → `skipped_detections` có 1 row reason="image_not_found".
6. **Force**: chạy 2 lần cùng config → lần 2 dừng + warn; thêm `--force` → tạo run mới.

## 10. File structure

```
DamageDetector/semi-labeling/step3_embedding/
├── __init__.py
├── SPEC.md                       # file này
├── README.md                     # quickstart
├── embed_detections.py           # main entry (~250 dòng)
├── embedder.py                   # DinoV2Embedder class (~100 dòng)
├── cropper.py                    # robust_crop_batch + resolve_image_path (~80 dòng)
├── source_store.py               # read_detections từ Step 2 (~70 dòng)
├── output_store.py               # SQLite writer (~120 dòng)
└── cache_reader.py               # load_embeddings cho downstream (~80 dòng)
```

Tổng ~700 dòng. Có thể reuse `DinoV2Embedder`, `resolve_image_path`, `crop_box` từ `_archive_step3_2026-05-22/code_step3_spatial_filter/filter_duplicates.py`.

## 11. Out of scope

- Embedding cho ảnh ngoài dataset (chỉ embed những detection có trong Step 2).
- Multi-model ensemble (1 run = 1 model).
- Re-projection / dim reduction (Step 5 sẽ tự PCA nếu cần).
- Embedding cho mask SAM (segmentation pipeline khác).

## 12. Tham chiếu

- DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023.
- Hugging Face model: `facebook/dinov2-giant`.
