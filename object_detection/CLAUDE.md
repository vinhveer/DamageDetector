# CLAUDE.md — object_detection

This file provides guidance to Claude Code when working in `object_detection/`.

## Vai trò của module

Module object_detection gồm tất cả code liên quan đến **phát hiện vùng hư hỏng** (bounding boxes). Phần cốt lõi là `dino/` — engine dùng GroundingDINO để phát hiện theo text query. Các submodule còn lại là wrappers, training pipeline, hoặc baseline comparison.

---

## Cấu trúc submodule

| Submodule | Vai trò |
|-----------|---------|
| `dino/` | **Engine chính**: GroundingDINO inference + tiled detection + DINOv2 reranking |
| `dinov2/` | Classifier và few-shot prototype matcher dùng DINOv2 embeddings |
| `grounding_dino/` | High-level wrapper đơn giản cho GroundingDINO (single image / folder) |
| `stable_dino/` | Training và inference cho StableDINO (baseline so sánh) |
| `yolo/` | YOLOv8 wrapper (training + inference) |
| `damage_scan/` | Multi-prompt pipeline chạy DINO qua nhiều loại hư hỏng, lưu kết quả vào SQLite |
| `infer_compare.py` | Evaluation harness: so sánh YOLO vs StableDINO, tính mAP, Precision, Recall |

---

## dino/engine.py — DinoRunner (cốt lõi)

Đây là file phức tạp nhất trong toàn repo. Phải đọc kỹ trước khi sửa.

### Model loading

```python
# DinoRunner.ensure_model_loaded() hỗ trợ 3 dạng checkpoint:
# 1. HuggingFace folder local (có config.json)
# 2. HuggingFace model ID (online, cache tại ~/.cache/huggingface)
# 3. .pth weights + config_id riêng (legacy format)
#
# Offline-first: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 được set trong process_client.py
```

### Simple predict (ảnh nhỏ, thường < 1024px)

```
predict(image_path, params)
  → load image
  → _build_valid_mask()        # xác định vùng không phải nền đen
  → run_text_boxes()           # GroundingDINO text → boxes
  → filter by:
      _box_is_fully_valid()    # box phải nằm trong valid mask
      _box_has_any_pure_black_pixels()  # reject nếu có pixel đen
      _box_black_ratio() < 0.40        # reject nếu >40% box là đen
  → return {detections: [...], image_path: ...}
```

### Recursive/Tiled detect (ảnh lớn, ảnh drone, TIFF)

Đây là phần quan trọng nhất — chạy DINO trên nhiều tile để xử lý ảnh lớn:

```
predict_recursive(image_path, params)
  → load image + valid_mask
  → _compute_oriented_roi()    # optional: xoay ảnh để tối ưu coverage
  → 3 tile scales song song:
      SMALL:  tile=512,  overlap=64,  min_coverage=0.30
      MEDIUM: tile=1024, overlap=160, min_coverage=0.15
      LARGE:  tile=1536, overlap=256, min_coverage=0.05
  → Với mỗi tile:
      _generate_valid_tiles()  # skip tiles quá nhiều nền đen
      run_text_boxes(tile_crop)
      shift boxes về tọa độ gốc
  → Collect tất cả detections từ mọi tiles
  → _map_box_from_rotated_to_original()  # undo rotation nếu có
  → Filter: min size, valid coverage, black ratio
  → target_label_filter → parent_box_filter → NMS → top-k
  → return {detections: [...], display_detections: [...]}
```

**Tại sao 3 scales?** Small tiles bắt chi tiết nhỏ; large tiles bắt hư hỏng rộng. Min coverage thấp hơn ở tiles lớn vì drone images thường có border đen.

### Tile refinement

Khi một tile có coverage thấp (nhiều nền đen), thay vì bỏ qua, engine sẽ tìm **connected components** trong valid mask và tạo "refined tiles" bao quanh từng component. Đây là cơ chế quan trọng cho ảnh GeoTIFF có nodata areas.

### Valid mask

```python
_build_valid_mask(image_path, image)
  # Ưu tiên rasterio cho TIFF:
  #   → dataset.dataset_mask() — chính xác nhất, dùng nodata values
  # Fallback OpenCV cho JPEG/PNG:
  #   → tìm vùng gần đen (threshold=12) → morphological close → invert
  # Output: bool mask shape (H, W), True = pixel hợp lệ
```

### DINOv2 reranking (rank_boxes)

Sau khi DINO phát hiện boxes, DINOv2 classifier hoặc prototype matcher được dùng để lọc false positives:

```
rank_boxes(image_path, params)
  → recursive_detect() hoặc predict()
  → crop mỗi box từ ảnh
  → DinoV2ClassifierRunner.classify(crops)    # top-k labels + confidence
  → _resolve_classifier_decision()            # accept/reject dựa trên rules
  → HOẶC DinoV2PrototypeRunner.match(crops)   # cosine sim với prototype images
  → sort by refined_score, return top-k
```

Classifier rules là dict JSON: `{"positive": "crack", "negative": null}` — "positive" map sang "crack", "negative" → reject.

---

## dino/client.py — DinoServicePool

Service pool quản lý nhiều worker subprocesses cho parallel inference:
- Mỗi worker được pin vào 1 CUDA device
- Requests được distribute qua Queue
- `get_dino_service()` trả về singleton DinoServicePool
- Dùng trong workflows.py: `ctx.call_service("dino", get_dino_service, "predict", params)`

---

## damage_scan/ — Multi-prompt Pipeline

Dùng khi cần quét toàn bộ folder ảnh qua nhiều loại hư hỏng:

```python
DamageScanConfig(
    checkpoint="IDEA-Research/grounding-dino-base",
    prompts=["surface crack", "spalling", "exposed rebar", ...],
    final_max_dets_per_class=5,
    max_box_fraction_of_image=0.6,
    image_workers=4,       # thread pool
    service_workers=2,     # DINO subprocess pool
)

DamageScanPipeline(config).run(input_dir, db_path)
# → SQLite: tables runs, images, detections
# → overlay images với colored boxes per prompt
```

**Pipeline flow:** với mỗi ảnh × mỗi prompt → detect → NMS → lưu SQLite + render overlay.

**PROMPT_SPECS** và **PROMPT_ORDER** trong `prompts.py` định nghĩa thứ tự và tên các loại hư hỏng được quét.

---

## dinov2/ — Classifier & Prototypes

### DinoV2ClassifierRunner
- Model: DINOv2 fine-tuned cho damage classification (surface_crack_image_detection hoặc custom)
- Input: list PIL images (crops từ DINO boxes)
- Output: top-k `[{"label": "crack", "confidence": 0.92}]`

### DinoV2PrototypeRunner
- Few-shot matching: so sánh cosine similarity với prototype crops
- Prototype crops lưu tại `models/<class>/*.png`
- `build_prototypes_from_yolo_dataset()` trích xuất crops từ YOLO dataset để làm prototypes
- Phân biệt class vs background bằng `background_labels`

---

## Thêm prompt / text query mới

Chỉ cần truyền vào `params["text_queries"]` khi gọi DINO:
```python
params = {"text_queries": ["crack", "delamination", "spalling"], "box_threshold": 0.25}
```
Không cần sửa code nào khác.

---

## Hyperparameters quan trọng trong engine.py

```python
_RECURSIVE_TILE_SIZE = 512          # tile nhỏ
_RECURSIVE_MEDIUM_TILE_SIZE = 1024  # tile trung
_RECURSIVE_LARGE_TILE_SIZE = 1536   # tile lớn
_RECURSIVE_TILE_OVERLAP = 64        # overlap để tránh miss boxes ở biên
_RECURSIVE_MIN_VALID_COVERAGE = 0.30
_RECURSIVE_MEDIUM_MIN_VALID_COVERAGE = 0.15
_RECURSIVE_LARGE_MIN_VALID_COVERAGE = 0.05
_DINO_BOX_BLACK_RATIO_REJECT = 0.40  # % pixel đen để reject box
_DINO_BLACK_PIXEL_THRESHOLD = 12     # giá trị RGB coi là "đen"
```

---

## CLI commands

```bash
# Single image
python -m object_detection.dino predict --image img.jpg --checkpoint IDEA-Research/grounding-dino-base

# Batch
python -m object_detection.dino predict-batch --images img1.jpg img2.jpg --checkpoint ...

# Tiled recursive (cho ảnh lớn)
python -m object_detection.dino recursive-detect --image large.tif --checkpoint ...

# Rerank với DINOv2
python -m object_detection.dino rank-boxes --image img.jpg --checkpoint ... --classifier-checkpoint ...

# Damage scan pipeline
python -m object_detection.damage_scan --input-dir ./images/ --db-path results.db --checkpoint ...
```

---

## Lưu ý khi sửa code

- `_generate_valid_tiles()` dùng integral image (`_mask_integral`) để check coverage trong O(1) — đừng dùng numpy sum trực tiếp.
- Rotation trong `_compute_oriented_roi()` áp dụng affine transform cho cả ảnh lẫn valid mask — nếu sửa rotation, phải update cả `_map_box_from_rotated_to_original()`.
- `display_detections` và `detections` trong recursive output khác nhau: `display_detections` là trước NMS (dùng để hiện thị partial results), `detections` là sau NMS (dùng cho SAM box prompting).
- Worker subprocess được spawn bởi `JsonServiceProcess` với env var `HF_HUB_OFFLINE=1` — model phải đã được download về local trước.
