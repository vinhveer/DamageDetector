# Semi-labeling

Code: `DamageDetector/semi-labeling/`. Kết quả: `infer_results/semi-labeling/`,
chia theo step, mỗi step một file `.sqlite3` riêng.

Có **hai tầng**. Xem bản đồ trực quan: mở `PIPELINE_MAP.html` bằng trình duyệt.

---

## Tầng 1 — Pipeline gốc (tạo dữ liệu)

Mỗi bước là một thư mục, ghi ra một artifact riêng:

| Thư mục | Vai trò | Output |
|---|---|---|
| `step1_gdino_labeling/` | GroundingDINO sinh bounding box | `step1_grounding_dino/damage_scan.sqlite3` |
| `step2_sematic/` | OpenCLIP chấm điểm ngữ nghĩa | `step2_sematic/damage_scan.sqlite3` |
| `step3_embedding/` | DINOv2 embedding cache (cho dedup) | `step3_embedding/embeddings.sqlite3` |
| `step4_class_aware_dedup/` | Lọc box trùng theo class | `step4_class_aware_dedup/dedup.sqlite3` |
| `step5_clustering/` | Gom cụm (cho app review) | `step5_clustering/clusters.sqlite3` |
| `step6_classifier/` | Classifier (cho app review) | `step6_classifier/` |
| `step7_label_review/` | Review nhãn (app Electron) | `step7_label_review/*.sqlite3` |

`app/` là app Electron review nhãn (đọc các artifact trên).

---

## Tầng 2 — `resemi/` (tinh chỉnh & làm sạch nhãn)

Gói v2, đọc output step2 (+ tùy chọn step4) rồi làm sạch nhãn. Tất cả ghi vào
**một** `infer_results/semi-labeling/resemi/resemi.sqlite3`.

### Cấu trúc (đã dọn)

```
resemi/
├── run_pipeline.py     ← CỬA VÀO DUY NHẤT (run + status + list-runs)
├── lib/                ← logic thuần, không CLI
│   ├── paths.py  bootstrap.py  schema.py  source_store.py  pipeline.py
│   ├── crop_generation.py  semantic_ensemble.py  bbox_quality.py
│   ├── core_mining.py  prototype_bank.py  reliability_scoring.py
│   ├── decision_policy.py  decision_policy_v1.py  label_taxonomy.py
│   ├── embedding_cache.py  lightweight_classifier.py  self_training.py
├── steps/              ← 9 BƯỚC, mỗi bước 1 file CLI thật
│   ├── step01_semantic.py … step09_self_train.py
└── tools/              ← tiện ích ngoài luồng
    ├── schema_audit.py  review_commit.py  render_bbox_overlays.py
```

Quy tắc: `lib/` chứa logic, `steps/` là 9 bước chạy, `tools/` là phụ trợ.
Không còn cặp `run_X.py` + `stepNN_X.py` trùng nhau — mỗi bước chỉ một file.

### Chạy

```bash
cd DamageDetector/semi-labeling

# xem một run đang ở bước nào
python -m resemi.run_pipeline status --run-id <id>

# liệt kê tất cả run
python -m resemi.run_pipeline list-runs

# chạy 1 bước
python -m resemi.run_pipeline run step01 --run-id <id>
python -m resemi.run_pipeline run step03 --run-id <id> --view-name tight --resume

# chạy 3 bước bắt buộc (step01→02→03)
python -m resemi.run_pipeline run all --run-id <id>
```

Chạy thẳng một bước/tool (tương đương):

```bash
python -m resemi.steps.step04_core --run-id <id>
python -m resemi.tools.schema_audit
```

### Thứ tự & phụ thuộc

| Step | Bắt buộc | Phụ thuộc | Ghi chú |
|---|---|---|---|
| step01 Semantic Init | ✅ | step2 (tầng 1) | không cần embedding |
| step02 Crop Generation | ✅ | step01 | |
| step03 DINOv2 Embedding | ✅ | step02 | **cổng chặn** cho step04→09 |
| step04 Core Mining | — | step03 | |
| step05 Prototype Bank | — | step03 | 🖐 người chọn prototype |
| step06 Reliability | — | step04+05 | |
| step07 Decision Policy | — | step06 | chia auto_accept vs review_queue |
| step08 Classifier | — | step07 | |
| step09 Self-Training | — | step08 | 🖐 audit trước `--apply-promotions` |

### 3 chốt làm tay

1. **Pick prototype** (sau step04, trước step05): chọn 20–50 ảnh/class + reject, truyền qua `--prototype id:label`.
2. **Review suspect** (sau step07): chỉ duyệt phần trong `review_queue`.
3. **Audit self-training** (step09): mặc định chỉ ghi audit; cần `--apply-promotions` mới sửa nhãn.
