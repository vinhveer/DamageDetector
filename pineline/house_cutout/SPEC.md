# SPEC — Pipeline: `house_cutout`

> **Mục tiêu:** Giống `dino_cutout` (GDINO tiled-detect damage trên cutout), **nhưng thêm
> một bước cắt nhà ở đầu** và **chạy được ảnh `.tif` lớn**.
>
> - **Step 1 (MỚI):** Mở ảnh gốc (kể cả `.tif` ~160 MB), dùng **GDINO + SAM** để cắt
>   riêng phần **nhà** ra thành cutout RGBA (nền trong suốt).
>   - Positive prompt: **house** → box vùng nhà → điểm dương cho SAM.
>   - Negative prompt: **window, door** → điểm âm cho SAM (mask nhà tự loại trừ cửa / cửa sổ).
> - **Step 2:** Nhận cutout RGBA của step 1, chạy GDINO tiled-detect crack / mold / stain.
>   Logic y hệt `dino_cutout/step1_gdino_detect`.

---

## 1. So sánh với `dino_cutout` và `sam_gdino`

| | `sam_gdino` | `dino_cutout` | `house_cutout` |
|---|---|---|---|
| Input | thư mục ảnh (bridge) | thư mục cutout RGBA có sẵn | **thư mục ảnh gốc, gồm `.tif`** |
| Cắt đối tượng | bridge (step1+2) | *không* (cutout có sẵn) | **house** (step1, GDINO+SAM) |
| Negative cho SAM | *không* | — | **window, door** (điểm âm) |
| Detect damage | step3 | step1 | **step2** (copy từ dino_cutout) |
| TIF lớn | không tối ưu | — | **convert + downscale "working RGB"** |

Khác biệt cốt lõi so với `sam_gdino/step2_sam_bridge_crop`:
- Có **điểm âm** (window/door) trong `predictor.predict(point_labels=...)`, không chỉ điểm dương.
- Bước `image_io` chuyển `.tif` → **working RGB** (PNG) và downscale nếu cạnh dài vượt
  `--work-max-side` (mặc định 4096), tránh nổ RAM khi `set_image` ảnh khổng lồ. Mọi tọa độ
  box/mask đều nằm trong **không gian working**; cutout xuất ra cũng ở độ phân giải working.

---

## 2. Cấu trúc thư mục

```
DamageDetector/pineline/house_cutout/
├── SPEC.md                       ← file này
├── __init__.py
├── run_pipeline.py               ← chạy step1 → step2 một lệnh (tiện lợi)
├── common/
│   ├── __init__.py
│   └── paths.py                  ← paths động (repo-relative), default SAM ckpt
├── step1_sam_house_crop/         ← BƯỚC MỚI: cắt nhà bằng GDINO + SAM
│   ├── __init__.py
│   ├── run.py                    ← CLI
│   ├── runner.py                 ← logic chính (§4)
│   ├── image_io.py               ← đọc tif/jpg/png → working RGB + downscale
│   ├── prompts.py                ← positive=house, negative=window/door
│   ├── gdino_house.py            ← gọi GDINO, tách box positive/negative
│   ├── point_sampler.py          ← lấy điểm dương/âm trong box
│   ├── sam_crop.py               ← SAM mask (pos+neg points) → RGBA cutout
│   ├── overlay.py                ← vẽ box + mask + crop bbox
│   └── store.py                  ← SQLite (run_info, images, detections, crops)
└── step2_gdino_detect/           ← copy logic dino_cutout/step1_gdino_detect
    ├── __init__.py
    ├── run.py
    ├── runner.py
    ├── prompts.py                ← crack/mold/stain (copy)
    ├── rgb_export.py             ← RGBA→RGB crop alpha bbox (copy)
    ├── overlay.py                ← vẽ box damage (copy)
    └── store.py                  ← schema detections (copy)
```

> **Reuse:** các file step2 copy gần như y hệt `dino_cutout/step1_gdino_detect`, chỉ đổi
> import path sang `pineline.house_cutout.step2_gdino_detect`. Không import chéo pipeline khác.

---

## 3. Đường dẫn (`common/paths.py`)

Dùng `_repo_root()` động (tìm thư mục chứa `object_detection/` và `inference_api/`) — **không
hardcode đường dẫn macOS** như `sam_gdino`. `LAB_ROOT = repo_root().parent`.

```python
RESULTS_ROOT = LAB_ROOT / "infer_results" / "pineline" / "house_cutout"

# Step 1 — cắt nhà
STEP1_DIR         = RESULTS_ROOT / "step1_sam_house_crop"
STEP1_DB          = STEP1_DIR / "house_crops.sqlite3"
STEP1_WORK_DIR    = STEP1_DIR / "work"        # working RGB (tif→png, đã downscale)
STEP1_CUTOUTS_DIR = STEP1_DIR / "cutouts"     # RGBA cutout nhà (input cho step2)
STEP1_MASKS_DIR   = STEP1_DIR / "masks"       # mask nhị phân
STEP1_OVERLAY_DIR = STEP1_DIR / "overlays"
STEP1_SUMMARY_CSV = STEP1_DIR / "summary.csv"

# Step 2 — detect damage (giống dino_cutout)
STEP2_DIR         = RESULTS_ROOT / "step2_gdino_detect"
STEP2_DB          = STEP2_DIR / "detections.sqlite3"
STEP2_RGB_DIR     = STEP2_DIR / "rgb"
STEP2_OVERLAY_DIR = STEP2_DIR / "overlays"
STEP2_SUMMARY_CSV = STEP2_DIR / "summary.csv"

DEFAULT_INPUT_DIR = LAB_ROOT / "data" / "HinhAnhThucTe"   # chứa "NTT - 16m Lan 3.tif"

def default_sam_checkpoint() -> Path:
    # ưu tiên vit_h, fallback vit_l/vit_b theo file có thật trong models/sam/
    ...
```

> **Lưu ý môi trường thật:** hiện `models/sam/` chỉ có `sam_vit_b_01ec64.pth`
> (CLAUDE.md ghi `sam_vit_h` nhưng chưa tải). `default_sam_checkpoint()` sẽ tự fallback
> sang file đang tồn tại; nếu muốn vit_h thì `python setup.py download_models --name sam_vit_h`.

---

## 4. Step 1 — cắt nhà (`step1_sam_house_crop`)

### 4.1 Luồng xử lý (mỗi ảnh)

```
for each image in --input-dir (đệ quy, gồm .tif):
    1. image_io.to_working_rgb(src, work_png, max_side)
         - đọc tif/jpg/png (cv2, fallback PIL) → RGB
         - nếu max(w,h) > max_side: resize giữ tỉ lệ; lưu scale = work/orig
         - ghi work_png; trả (work_rgb ndarray, scale, orig_w, orig_h)
    2. gdino_house.detect(service, work_png, positive=[house], negative=[window,door], ...)
         - ảnh nhà lớn → max_dim thường > tiled_threshold → recursive_detect, ngược lại predict
         - tách detections thành: house_boxes (>= score_floor) và neg_boxes (window/door)
    3. nếu không có house_box: log "no house"; skip (không tạo cutout)
    4. SAM:
         predictor.set_image(work_rgb)
         neg_points_all = sample_points trong từng neg_box  (label 0)
         for mỗi house_box:
             pos_points  = sample_points trong house_box     (label 1)
             neg_points  = neg_points_all ∩ nằm trong house_box
             mask = sam_crop.predict_mask(predictor, pos_points, neg_points)
             gom vào masks[]
         union = OR tất cả masks
    5. bbox = mask_bbox(union, pad=pad_px); crop RGBA (alpha=mask) → cutouts/<id>.png
       ghi mask nhị phân → masks/<id>.png
    6. overlay: house box (xanh), window/door box (đỏ), điểm dương/âm, mask, crop bbox
    7. store: insert image + house/neg detections + crop row; commit
```

### 4.2 SAM với điểm âm (`sam_crop.predict_mask`)

```python
coords = np.array(pos_points + neg_points, np.float32)
labels = np.array([1]*len(pos_points) + [0]*len(neg_points), np.int32)
masks, scores, _ = predictor.predict(
    point_coords=coords, point_labels=labels, multimask_output=True,
)
return masks[int(np.argmax(scores))].astype(bool)
```

> Đây là điểm khác chính so với `sam_gdino` (vốn `labels` toàn 1). Window/door trở thành
> ràng buộc "không thuộc nhà", giúp mask nhà không nuốt khung cửa kính.

### 4.3 Prompt mặc định (`prompts.py`)

| Vai trò | Queries mặc định |
|---|---|
| positive | `house`, `building`, `residential building`, `facade` |
| negative | `window`, `door`, `glass window`, `glass door` |

CLI cho phép override: `--positive house --positive building`, `--negative window --negative door`.
Việc gộp text query gửi GDINO = positive ∪ negative; sau đó map label → positive/negative theo
keyword (`window`/`door` → negative, còn lại → positive).

### 4.4 CLI (`run.py`)

```bash
python -m pineline.house_cutout.step1_sam_house_crop.run \
  --input-dir data/HinhAnhThucTe \
  [--db PATH] [--work-dir PATH] [--cutouts-dir PATH] [--masks-dir PATH]
  [--overlay-dir PATH] [--summary-csv PATH] [--no-overlay]
  [--positive house] [--negative window] [--negative door]   # repeat được
  [--checkpoint PATH]              # GDINO ckpt (auto-detect)
  [--sam-checkpoint PATH]          # default: default_sam_checkpoint()
  [--sam-model-type auto]
  [--device auto]
  [--work-max-side 4096]           # downscale working RGB nếu cạnh dài lớn hơn
  [--box-threshold 0.15] [--text-threshold 0.15] [--max-dets 50]
  [--tiled-threshold 2048]         # ảnh > ngưỡng → recursive_detect tìm house
  [--score-floor 0.20]             # bỏ house box dưới ngưỡng
  [--points-per-box 5]             # số điểm dương/âm mỗi box
  [--pad-px 16]                    # nới crop bbox
  [--limit 0] [--skip-existing]
```

### 4.5 SQLite (`store.py`)

```sql
CREATE TABLE run_info (
    run_id, created_at_utc, input_dir, gdino_checkpoint, sam_checkpoint,
    device, work_max_side, box_threshold, text_threshold, score_floor,
    points_per_box, pad_px, image_count, cutout_count
);
CREATE TABLE images (
    run_id, image_id PRIMARY KEY-part, image_path, image_rel_path,
    orig_width, orig_height, work_width, work_height, scale,
    house_box_count, neg_box_count, has_cutout
);
CREATE TABLE detections (              -- house + window/door trong KHÔNG GIAN working
    run_id, image_id, det_idx, role TEXT,   -- 'house' | 'negative'
    label, x1, y1, x2, y2, score
);
CREATE TABLE crops (                   -- cutout nhà tạo ra
    run_id, image_id, cutout_path, mask_path,
    crop_x1, crop_y1, crop_x2, crop_y2, mask_area_px, source_box_count
);
```

---

## 5. Step 2 — detect damage (`step2_gdino_detect`)

Copy nguyên `dino_cutout/step1_gdino_detect` (runner/prompts/overlay/rgb_export/store), chỉ:
- Đổi import → `pineline.house_cutout.step2_gdino_detect.*`
- `--input-dir` mặc định = `STEP1_CUTOUTS_DIR` (cutout nhà của step1)
- Output → `RESULTS_ROOT/step2_gdino_detect/`

Hành vi giữ nguyên: RGBA→RGB crop theo alpha bbox, tiled `recursive_detect` (ảnh rộng),
lọc box theo group crack/mold/stain + `max_box_area_ratio`, lưu DB/overlay/CSV, map box về
khung cutout gốc qua `offset_x/offset_y`.

---

## 6. Output

```
infer_results/pineline/house_cutout/
├── step1_sam_house_crop/
│   ├── house_crops.sqlite3
│   ├── work/        <id>.png         ← working RGB (tif đã convert/downscale)
│   ├── cutouts/     <id>.png         ← RGBA cutout nhà  → input step2
│   ├── masks/       <id>.png         ← mask nhị phân
│   ├── overlays/    <id>.png
│   └── summary.csv
└── step2_gdino_detect/
    ├── detections.sqlite3
    ├── rgb/         <id>.png
    ├── overlays/    <id>.png
    └── summary.csv
```

---

## 7. Chạy thử

```bash
cd DamageDetector

# B1: cắt nhà (thử 1 ảnh tif)
python -m pineline.house_cutout.step1_sam_house_crop.run \
  --input-dir "../data/HinhAnhThucTe" --limit 1 --device auto

# B2: detect damage trên cutout vừa tạo
python -m pineline.house_cutout.step2_gdino_detect.run --device auto

# Hoặc chạy cả hai một lệnh
python -m pineline.house_cutout.run_pipeline --input-dir "../data/HinhAnhThucTe" --device auto
```

---

## 8. Không làm trong lần này

- OpenCLIP semantic / DINOv2 dedup / SAM-LoRA segment (như dino_cutout §8).
- Multi-house tách riêng từng cutout: hiện **union** mọi house box thành 1 cutout/ảnh.
- Batch resume nâng cao: gián đoạn thì chạy lại tạo `run_id` mới (có `--skip-existing`).

---

**Ngày tạo:** 2026-06-02
**Pipeline:** `DamageDetector/pineline/house_cutout/`
**Input mặc định:** `data/HinhAnhThucTe/` (gồm `NTT - 16m Lan 3.tif` ~160 MB)
