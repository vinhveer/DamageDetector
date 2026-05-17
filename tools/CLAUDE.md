# CLAUDE.md — tools

This file provides guidance to Claude Code when working in `tools/`.

## Vai trò của module

`tools/` là tập hợp các **standalone scripts** và **GUI apps** phục vụ data preparation, evaluation, và visualization. Không có logic nghiệp vụ nào ở đây — tất cả đều delegate vào các module core (`object_detection`, `segmentation`, `data_split`).

---

## Scripts chính

### Data preparation

| Script | Mục đích | Input → Output |
|--------|---------|----------------|
| `clean_binary_mask_dataset.py` | Dọn dẹp noise trong binary masks (speckles, thin artifacts) | `--dataset-root` có `images/` + `masks/` → masks sạch in-place + JSON manifest |
| `convert_crack500_seg_to_detection.py` | Chuyển segmentation masks → YOLO detection boxes | `--src-root` → `--out-dir` có `labels/*.txt` + `data.yaml` |
| `extract_dataset_subset.py` | Trích xuất tập con của dataset theo keyword trong tên file | `--input-root`, `--match-token`, `--output-root` |
| `crop_images.py` | Mở PyQt cropper UI để tạo ROI crops thủ công | Interactive — xem `tools/ui/create_data_tools/cropper_app/` |

### Model download

```bash
python -m tools.download_models list                          # xem tất cả presets
python -m tools.download_models download --name sam_vit_h    # download preset
python -m tools.download_models download --name all          # download tất cả
python -m tools.download_models download-hf --repo-id IDEA-Research/grounding-dino-base
python -m tools.download_models download-url --url https://...
```

Các preset: `sam_vit_b`, `sam_vit_l`, `sam_vit_h`, `grounding_dino_base`, `dinov2_small`

### Inference tools

```bash
# Single image GroundingDINO
python tools/gdino_detect_image.py \
  --image-path img.jpg \
  --prompt "crack" \
  --box-threshold 0.25 \
  --output-overlay overlay.jpg

# Batch folder
python tools/gdino_detect_folder.py \
  --input-dir ./images/ \
  --output-dir ./detections/ \
  --prompt "crack . spalling . exposed rebar" \
  --recursive
```

### Evaluation → SQLite/Excel

```bash
# UNet evaluation
python tools/run_unet_eval_to_sqlite.py \
  --dataset-path ./val_data \
  --model-checkpoint ./best_model.pth \
  --output-db ./eval_results.db

# SAM finetune evaluation → Excel
python tools/run_sam_eval_to_excel.py \
  --project-root ./experiments \
  --dataset-path ./test_data \
  --model-dir ./sam_outputs \
  --sam-ckpt ./sam_vit_h.pth \
  --output-root ./eval_excel

# SAM finetune evaluation → SQLite
python tools/run_sam_eval_to_sqlite.py [similar args]
```

Kết quả lưu theo threshold sweep (0.1 → 0.9, step 0.05) với metrics: Dice, IoU, Precision, Recall, skeleton metrics.

### Semi-labeling

```bash
python tools/semi_label_gdino_sqlite.py \
  --dataset-path ./unlabeled_images \
  --output-db ./semi_labels.db \
  --prompt-groups "crack,surface crack" "spalling,delamination"
```

Chạy GroundingDINO trên ảnh chưa label, lưu predictions vào SQLite để review thủ công.

### Tiled damage scan

```bash
python tools/detect_damage_tiled_sqlite.py \
  --input-dir ./inspection_images \
  --db-path ./damage_scan.db \
  --checkpoint IDEA-Research/grounding-dino-base
```

Delegates trực tiếp vào `object_detection.damage_scan.cli.main()`.

---

## convert_crack500_seg_to_detection.py — Quan trọng

Script này có nhiều heuristics để quyết định một connected component có nên trở thành bounding box không:

```python
--max-box-len-ratio 0.35   # box không dài hơn 35% chiều dài ảnh → tránh "full-image" boxes
--max-aspect-ratio 25      # aspect ratio tối đa → tránh boxes cực kỳ dài/hẹp
--max-area-ratio 0.15      # box không chiếm quá 15% diện tích ảnh
--min-component-area 10    # bỏ qua components quá nhỏ (noise)
--tile-overlap 0.15        # overlap khi split ảnh lớn thành tiles
```

Output: YOLO format (`class_id cx cy w h` normalized) + `data.yaml` manifest.

---

## tools/ui/ — GUI Apps

### editor_app (PySide6)

Desktop app đầy đủ tính năng cho inference. Xem CLAUDE.md riêng tại:
`tools/ui/editor_app/CLAUDE.md`

### create_data_tools/cropper_app (PySide6)

Công cụ tạo ROI crops thủ công:
- Load ảnh, vẽ bounding boxes
- Lưu crops vào SQLite (`roi_db.py`) để track metadata
- Export crops theo folder structure

### streamlit_demo

Demo đơn giản dùng Streamlit để visualize model predictions:
```bash
python -m tools.ui.streamlit_demo
```

---

## Lưu ý khi sửa code

- Scripts trong `tools/` không có tests — khi sửa, test thủ công trên sample data nhỏ trước.
- `sitecustomize.py` được load tự động bởi Python khi `tools/` trong PYTHONPATH — nó setup môi trường (encoding, warnings). Không xóa file này.
- `plot_style.py` là shared styling cho matplotlib — import nó ở đầu mọi plotting script để đảm bảo consistent visuals.
- Evaluation scripts (run_*_eval_*) không idempotent — chạy lại sẽ append thêm rows vào SQLite. Xóa file `.db` trước khi chạy lại từ đầu.
