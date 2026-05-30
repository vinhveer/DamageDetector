# SPEC — Pipeline: `dino_cutout`

> **Mục tiêu:** Chạy GDINO tiled-detect trực tiếp trên thư mục ảnh đã crop sẵn (SAM cutouts).
> Không cần step detect bridge, không cần SAM segmentation — chỉ detect damage.
>
> **So với `sam_gdino`:** Pipeline này bỏ step 1-2 (bridge detect + SAM crop) và step 4-6
> (semantic label + dedup + segment). Chỉ thực hiện tương đương step 3 nhưng nhận thẳng
> `--input-dir` thay vì đọc từ step2 SQLite.

---

## 1. Bối cảnh & Lý do tách pipeline

| | `sam_gdino` step 3 | `dino_cutout` |
|---|---|---|
| Input | SQLite từ step 2 (bridge crops) | `--input-dir` trực tiếp |
| Ảnh | RGBA crop từ SAM mask (trong `infer_results/`) | RGBA cutout sẵn (ví dụ `model_with_inference/pineline_detect_damage/cutouts/`) |
| Tiled | `tiled_threshold=1600`, scales `small/medium/large` | **Bắt buộc tiled** — ảnh rộng ~5472 px, tile cố định 512 px |
| Output | Phụ thuộc run_id step 2 | Độc lập — chỉ cần `--input-dir` |

Cutout hiện tại: 102 ảnh RGBA, kích thước điển hình **5472 × 1000–2500 px**. Ở kích thước này:
- Luôn cần tiling (5472 >> bất kỳ tiled_threshold hợp lý nào)
- Tile 512 px → ~10 tiles theo chiều ngang, 2–5 tiles theo chiều dọc

---

## 2. Cấu trúc thư mục

```
DamageDetector/pineline/dino_cutout/
├── SPEC.md                     ← file này
├── __init__.py
├── common/
│   ├── __init__.py
│   └── paths.py                ← đường dẫn mặc định (output root, DB paths)
└── step1_gdino_detect/
    ├── __init__.py
    ├── run.py                  ← CLI entry point (argparse → gọi runner)
    ├── runner.py               ← logic chính: iter ảnh → convert → detect → store
    ├── overlay.py              ← vẽ bounding box lên ảnh (reuse từ step3_gdino_damage)
    ├── prompts.py              ← damage prompt groups (reuse từ step3_gdino_damage)
    ├── rgb_export.py           ← RGBA→RGB on black (reuse từ step3_gdino_damage)
    └── store.py                ← SQLite schema + insert helpers
```

> **Reuse:** `overlay.py`, `prompts.py`, `rgb_export.py` copy y chang từ
> `pineline/sam_gdino/step3_gdino_damage/`. Không import chéo — copy để pipeline tự chứa.

---

## 3. Đường dẫn mặc định (`common/paths.py`)

```python
LAB_ROOT    = Path("/Users/nguyenquangvinh/Desktop/Lab")
RESULTS_ROOT = LAB_ROOT / "infer_results" / "pineline" / "dino_cutout"

STEP1_DIR        = RESULTS_ROOT / "step1_gdino_detect"
STEP1_DB         = STEP1_DIR / "detections.sqlite3"
STEP1_RGB_DIR    = STEP1_DIR / "rgb"          # ảnh RGBA đã convert sang RGB
STEP1_OVERLAY_DIR = STEP1_DIR / "overlays"
STEP1_SUMMARY_CSV = STEP1_DIR / "summary.csv"

DEFAULT_INPUT_DIR = LAB_ROOT / "model_with_inference" / "pineline_detect_damage" / "cutouts"
```

---

## 4. Step 1: GDINO detect damage (`step1_gdino_detect`)

### 4.1 Luồng xử lý

```
for each image in --input-dir:
    1. rgba_to_rgb_on_black(image) → ghi vào rgb_dir/   [skip nếu đã có]
    2. max_dim = max(w, h)
    3. if max_dim > tiled_threshold (mặc định 400):
           service.call("recursive_detect", {tile_size: 512, ...})
       else:
           service.call("predict", {...})
    4. Lọc box:
       - Bỏ box chiếm > max_box_area_ratio (mặc định 0.50) diện tích ảnh
       - match_group_for_label(label, prompt_groups) → giữ box có group
    5. insert_detections(conn, rows)
    6. write_overlay(image_rgb, detections, overlay_dir/)
    7. log + commit
```

### 4.2 Prompt groups (mặc định)

| Group | Slug | Queries |
|-------|------|---------|
| crack | `01-crack` | crack, surface crack, concrete crack, wall crack, thin crack, long crack, hairline crack, fracture line |
| mold  | `02-mold`  | mold, mildew, moss, algae, algae stain, biological growth on wall, green stain on concrete, fungal growth, lichen on surface |
| stain | `03-stain` | water stain, damp stain, dark stain on wall, discoloration patch, dirty stain, rust stain, surface contamination, blackish patch on concrete |

### 4.3 Gọi GDINO service

```python
params = {
    "gdino_checkpoint": checkpoint,
    "gdino_config_id": "auto",
    "text_queries": combined_queries(prompt_groups),   # tất cả queries gộp lại
    "box_threshold": box_threshold,    # mặc định 0.16
    "text_threshold": text_threshold,  # mặc định 0.16
    "max_dets": max_dets,              # mặc định 80
    "device": device,
    "recursive_tile_scales": tile_scales,  # ["small", "medium"] — khớp ~512 px tile
}

# Luôn dùng recursive_detect vì ảnh ~5472 px
result = service.call("recursive_detect", {
    "image_path": str(rgb_path),
    "params": params,
    "target_labels": combined_queries(prompt_groups),
    "max_depth": recursive_max_depth,   # mặc định 2
    "min_box_px": min_box_px,           # mặc định 16
})
```

> **Về tile size 512 px:** `recursive_tile_scales=["small", "medium"]` cho tiles nhỏ hơn
> so với `["small", "medium", "large"]` trong step3. Nếu GDINO service hỗ trợ tham số
> `tile_size` trực tiếp thì dùng `tile_size=512`; nếu không thì dùng scales và document
> rõ trong `run.md`.

### 4.4 SQLite schema (`store.py`)

```sql
-- Bảng run metadata
CREATE TABLE IF NOT EXISTS run_info (
    run_id          TEXT PRIMARY KEY,
    created_at_utc  TEXT,
    input_dir       TEXT,
    checkpoint      TEXT,
    device          TEXT,
    box_threshold   REAL,
    text_threshold  REAL,
    tile_scales     TEXT,        -- JSON array
    max_dets        INTEGER,
    image_count     INTEGER,
    detection_count INTEGER
);

-- Bảng ảnh đã xử lý
CREATE TABLE IF NOT EXISTS images (
    run_id          TEXT,
    image_id        TEXT,        -- sha1[:12] của rel path
    image_path      TEXT,
    image_rel_path  TEXT,
    width           INTEGER,
    height          INTEGER,
    det_count       INTEGER,
    PRIMARY KEY (run_id, image_id)
);

-- Bảng detections
CREATE TABLE IF NOT EXISTS detections (
    run_id          TEXT,
    image_id        TEXT,
    det_idx         INTEGER,
    group_id        INTEGER,
    group_name      TEXT,        -- "crack" | "mold" | "stain"
    label           TEXT,        -- raw GDINO label
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    score           REAL,
    PRIMARY KEY (run_id, image_id, det_idx)
);

-- Bảng prompt groups (audit trail)
CREATE TABLE IF NOT EXISTS prompt_groups (
    run_id      TEXT,
    group_id    INTEGER,
    name        TEXT,
    slug        TEXT,
    queries_json TEXT,
    PRIMARY KEY (run_id, group_id)
);
```

### 4.5 CLI (`run.py`)

```bash
python -m pineline.dino_cutout.step1_gdino_detect.run \
  --input-dir /Users/nguyenquangvinh/Desktop/Lab/model_with_inference/pineline_detect_damage/cutouts \
  [--db PATH]                      # default: STEP1_DB
  [--rgb-dir PATH]                 # default: STEP1_RGB_DIR
  [--overlay-dir PATH]             # default: STEP1_OVERLAY_DIR
  [--summary-csv PATH]             # default: STEP1_SUMMARY_CSV
  [--no-overlay]                   # tắt ghi overlay
  [--checkpoint PATH]              # auto-detect từ DamageDetector/models/
  [--device auto]
  [--box-threshold 0.16]
  [--text-threshold 0.16]
  [--max-dets 80]
  [--tiled-threshold 400]          # ảnh > 400 px → dùng recursive_detect
  [--tile-scales small medium]     # maps to ~512 px tiles
  [--recursive-max-depth 2]
  [--min-box-px 16]
  [--max-box-area-ratio 0.50]
  [--limit 0]                      # 0 = không giới hạn (debug: --limit 5)
  [--source-run-id OVERRIDE]       # nếu muốn gộp vào run_id cụ thể
```

**Bảng tham số quan trọng:**

| Tham số | Mặc định | Ghi chú |
|---------|----------|---------|
| `--input-dir` | *(bắt buộc)* | thư mục chứa cutout PNG/JPEG |
| `--box-threshold` | `0.16` | ngưỡng raw GDINO — thấp để lấy pool rộng |
| `--text-threshold` | `0.16` | |
| `--tiled-threshold` | `400` | đặt thấp để ảnh 5472 px luôn tiled |
| `--tile-scales` | `small medium` | scale nhỏ → tile ~512 px |
| `--recursive-max-depth` | `2` | 1=1 lần chia, 2=chia đệ quy thêm |
| `--min-box-px` | `16` | bỏ box < 16 px bất kỳ chiều |
| `--max-box-area-ratio` | `0.50` | bỏ box chiếm > 50% diện tích ảnh |

---

## 5. Output

```
infer_results/pineline/dino_cutout/
└── step1_gdino_detect/
    ├── detections.sqlite3      ← toàn bộ kết quả (nhiều run_id, append)
    ├── rgb/                    ← ảnh RGBA đã convert → RGB
    │   ├── <image_id>.png
    │   └── ...
    ├── overlays/               ← ảnh RGB có vẽ bounding box
    │   ├── <image_id>.png
    │   └── ...
    └── summary.csv             ← flat CSV của run mới nhất
        columns: image_rel_path, image_id, det_idx, group_name, label,
                 x1, y1, x2, y2, score
```

---

## 6. Thứ tự implement

1. **`common/paths.py`** — khai báo paths, `ensure_dirs()`
2. **Copy** `rgb_export.py`, `overlay.py`, `prompts.py` từ `step3_gdino_damage/`
3. **`step1_gdino_detect/store.py`** — schema SQLite + helpers (`ensure_schema`, `insert_images`, `insert_detections`, `insert_prompt_groups`, `insert_run_info`)
4. **`step1_gdino_detect/runner.py`** — logic `run_step1(input_dir, ...)` theo đúng luồng §4.1
5. **`step1_gdino_detect/run.py`** — argparse → `run_step1` → print result dict
6. **`__init__.py`** (trống) ở mỗi package

---

## 7. Kiểm tra nhanh

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector

# Chạy thử 3 ảnh đầu để xem log + overlay
.venv/bin/python -m pineline.dino_cutout.step1_gdino_detect.run \
  --input-dir /Users/nguyenquangvinh/Desktop/Lab/model_with_inference/pineline_detect_damage/cutouts \
  --limit 3 \
  --device auto

# Xem overlay
open /Users/nguyenquangvinh/Desktop/Lab/infer_results/pineline/dino_cutout/step1_gdino_detect/overlays/
```

---

## 8. Không làm trong lần này

- **OpenCLIP semantic label** (step4 analog) — có thể thêm sau nếu cần phân biệt crack/mold/stain rõ hơn
- **DINOv2 dedup** (step5 analog) — thêm sau nếu detect trùng lắp nhiều
- **SAM segment** (step6 analog) — ngoài scope: pipeline này chỉ cần boxes
- **Batch resume** — nếu bị gián đoạn, chạy lại tạo `run_id` mới, không overwrite

---

**Ngày tạo:** 2026-05-29  
**Pipeline:** `DamageDetector/pineline/dino_cutout/`  
**Input mặc định:** `model_with_inference/pineline_detect_damage/cutouts/` (102 ảnh RGBA ~5472 px)
