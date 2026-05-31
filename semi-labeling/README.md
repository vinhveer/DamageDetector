# Semi-labeling — Commands

Pipeline bán nhãn hư hỏng (crack / mold / spall). Toàn bộ code gom trong
`semi-labeling/`:

- `steps/gdino_scan/` — khoanh box (wrapper GDINO).
- `steps/openclip_semantic/` — chấm nhãn semantic (input cho resemi).
- `steps/step01_semantic … step09_self_train/` — 9 bước resemi (mỗi bước `main.py`).
- `shared/` — module dùng chung (runtime / db / taxonomy / crop).
- `tools/` — schema_audit, review_commit, render_bbox_overlays.
- `run_pipeline.py` — orchestrator cho 9 step resemi.
- `app/` — Electron review app (layout shell).

Tất cả lệnh dưới chạy từ thư mục `DamageDetector/`:

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
PY=.venv/bin/python      # interpreter dự án
```

Quy ước biến dùng lại:

```bash
IMG=/Users/nguyenquangvinh/Desktop/Lab/data/HinhAnh                 # ảnh nguồn
OUT=/Users/nguyenquangvinh/Desktop/Lab/model_with_inference/semi_labeling
```

---

## Pipeline 3 bước đầu (GDINO → OpenCLIP → DINOv2)

3 bước nối nhau, **ghi chung 1 file SQLite** (`pipeline.sqlite3`).
Đã smoke-test 5 ảnh chạy sạch, không lỗi.

```bash
DB=$OUT/step1_grounding_dino/pipeline.sqlite3
```

### Step 1 — GroundingDINO (khoanh box, recall cao)

Sinh bounding box hư hỏng. Mặc định `damage_scan` chạy **full-image, threshold/NMS
gốc** (không đổi hành vi repo). Để bật chế độ recall cao cho semi-labeling, **truyền
flag tường minh**: tiled lưới 1024 + full pass, nới NMS, hạ box threshold.

```bash
$PY -m steps.gdino_scan.main \
  --input-dir $IMG \
  --db $DB \
  --device mps \
  --tiled-threshold 1024 --tile-size 1024 --tile-overlap 128 \
  --nms-iou 0.45 --box-threshold 0.12 \
  --final-max-dets-per-class 400 \
  --save-overlays
```

Các flag recall (chỉ bật khi truyền; bỏ đi = hành vi gốc):
- `--tiled-threshold 1024` — ảnh max-dim > 1024 chạy tiled lưới + 1 pass full. `0` = luôn full-image (mặc định).
- `--tile-size 1024 --tile-overlap 128` — kích thước ô và overlap khi tiled.
- `--nms-iou 0.45` — nới NMS, giữ box chồng/sát nhau. `0` = dùng giá trị theo prompt spec.
- `--box-threshold 0.12` — hạ ngưỡng box (bắt thêm box yếu). `0` = dùng giá trị theo prompt spec.

Cờ khác:
- `--limit N` — chỉ N ảnh đầu (smoke test).
- `--no-save-overlays` — không xuất overlay PNG (nhanh hơn).
- `--checkpoint IDEA-Research/grounding-dino-base` — dùng model HF thay vì folder local.

Output: `detections`, `images`, `runs` trong DB + overlay ở `<db_parent>/overlays/`.

### Step 2 — OpenCLIP semantic (chấm lại nhãn từng box)

Đọc box step1, crop, chấm % crack/mold/spall bằng OpenCLIP. Ghi vào **cùng DB**.

```bash
$PY -m steps.openclip_semantic.main \
  --db $DB \
  --image-root $IMG \
  --source-run-id latest \
  --stage final \
  --model-name ViT-B-32 --pretrained laion2b_s34b_b79k \
  --device mps
```

- Bản nhẹ/nhanh: `--model-name ViT-B-32` (mặc định).
- Bản chất lượng cao: `--model-name ViT-H-14 --pretrained laion2b_s32b_b79k`.

Output: `openclip_semantic_results`, `openclip_semantic_scores`, `openclip_semantic_runs`.

### Step 3 — DINOv2 embedding (cache vector cho các bước sau)

Đây chính là **resemi step03** (đọc semantic step2, embed crop bằng DINOv2, ghi
`detection_embeddings`). Chạy qua orchestrator hoặc trực tiếp:

```bash
cd semi-labeling
$PY -m steps.step03_embed.main \
  --db $DB \
  --view-name tight \
  --model-name facebook/dinov2-small \
  --device mps --batch-size 8 --resume
```

- Bản nhẹ: `facebook/dinov2-small`. Bản nặng: `facebook/dinov2-giant`.
- `--resume` chạy tiếp run dở; `--force` tạo run embedding mới.

Output: `detection_embeddings`, `embedding_runs`.

### Kiểm tra DB

```bash
$PY - <<'PY'
import sqlite3
c = sqlite3.connect("DB_PATH")  # thay DB_PATH
for (t,) in c.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"):
    print(f"{t:32} {c.execute(f'select count(*) from {t}').fetchone()[0]}")
PY
```

---

## Chạy trên Google Colab

Colab là máy sạch → model không có sẵn local. step2 (open_clip) và step3
(DINOv2 HF) tự tải. Riêng step1 GDINO phải:

1. Truyền repo id `--checkpoint IDEA-Research/grounding-dino-base`.
2. **Tắt offline mode** (code mặc định ép offline) bằng env:

```bash
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_TOKEN=<token>      # nên có, tránh rate-limit khi tải model

IMG=/content/data/HinhAnh
DB=/content/out/pipeline.sqlite3
cd /content/DamageDetector/semi-labeling   # các lệnh -m chạy từ đây

# Step 1 — GDINO
python -m steps.gdino_scan.main \
  --input-dir $IMG --db $DB \
  --checkpoint IDEA-Research/grounding-dino-base \
  --device cuda \
  --tiled-threshold 1024 --tile-size 1024 \
  --nms-iou 0.45 --box-threshold 0.12 --final-max-dets-per-class 400

# Step 2 — OpenCLIP
python -m steps.openclip_semantic.main \
  --db $DB --image-root $IMG --source-run-id latest --stage final \
  --model-name ViT-H-14 --pretrained laion2b_s32b_b79k --device cuda

# Step 3 — DINOv2 embedding (resemi step03)
python -m steps.step03_embed.main \
  --db $DB --view-name tight --model-name facebook/dinov2-giant \
  --device cuda --batch-size 16 --resume
```

Đổi `--device mps` → `--device cuda` cho mọi bước trên Colab.

---

## resemi v2 (tinh chỉnh nhãn, tùy chọn — sau 3 bước trên)

Cấu trúc code (đã tách theo tính năng để dễ đọc):

```
semi-labeling/
├── run_pipeline.py          # orchestrator (status / list-runs / run)
├── shared/                  # module dùng chung nhiều bước
│   ├── runtime/  paths.py, bootstrap.py
│   ├── db/       schema.py, source_store.py, embedding_cache.py
│   ├── taxonomy/ label_taxonomy.py
│   └── crop/     crop_generation.py
├── steps/                   # mỗi bước = 1 thư mục, main.py = CLI riêng + logic của bước
│   ├── step01_semantic/  main.py + pipeline, semantic_ensemble, decision_policy, bbox_quality
│   ├── step02_crops/     main.py
│   ├── step03_embed/     main.py
│   ├── step04_core/      main.py + core_mining.py
│   ├── step05_proto/     main.py + prototype_bank.py
│   ├── step06_reliability/ main.py + reliability_scoring.py
│   ├── step07_decision/  main.py + decision_policy_v1.py
│   ├── step08_classifier/ main.py + lightweight_classifier.py
│   └── step09_self_train/ main.py + self_training.py
└── tools/                   # schema_audit, review_commit, render_bbox_overlays
```

Chạy từ `semi-labeling/`:

```bash
cd semi-labeling
PY=../.venv/bin/python

$PY -m run_pipeline status --run-id <id>      # xem run ở bước nào
$PY -m run_pipeline list-runs                 # liệt kê run
$PY -m run_pipeline run step01 --run-id <id>  # chạy 1 bước
$PY -m run_pipeline run all --run-id <id>     # step01→02→03 bắt buộc

# chạy thẳng 1 step / tool (mỗi step là package, CLI ở main.py)
$PY -m steps.step04_core.main --run-id <id>
$PY -m tools.schema_audit
```

resemi đọc input từ `infer_results/semi-labeling/` (step2 + step4 cũ),
ghi output gộp 1 file `resemi.sqlite3` vào `model_with_inference/semi_labeling/`.

9 step resemi: step01 Semantic Init · step02 Crop · step03 Embedding (cổng chặn
step04→09) · step04 Core · step05 Prototype (🖐 pick tay) · step06 Reliability ·
step07 Decision · step08 Classifier · step09 Self-train (🖐 audit trước
`--apply-promotions`).
