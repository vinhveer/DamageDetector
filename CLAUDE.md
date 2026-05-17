# CLAUDE.md — DamageDetector

This file provides guidance to Claude Code when working in the `DamageDetector/` project.

## Tổng quan

DamageDetector là hệ thống phát hiện và phân tích hư hỏng kết cấu (crack, spalling, v.v.) từ ảnh. Pipeline đầy đủ gồm:

```
Ảnh đầu vào
    ↓ object_detection/   → Phát hiện vùng hư hỏng (bounding boxes) qua text query
    ↓ segmentation/       → Phân vùng pixel-level (masks) từ boxes hoặc độc lập
    ↓ inference_api/      → Orchestration: quản lý jobs, workflows, IPC
    ↓ tools/ui/editor_app → Desktop app PySide6 để tương tác và xem kết quả
    ↓ app/                → Electron app để chạy workflows & xem data split results
```

## Install & setup

```bash
pip install -e .
python setup.py download_models          # download tất cả model weights
python setup.py download_models --name sam_vit_h   # hoặc specific model
```

## Run tests

```bash
python -m unittest tests/test_dino_valid_tiling.py
```

## Chỉ dẫn chi tiết theo module

Mỗi module có CLAUDE.md riêng — **đọc file tương ứng trước khi làm việc trong module đó**:

| Module | CLAUDE.md | Mô tả ngắn |
|--------|-----------|------------|
| `inference_api/` | [inference_api/CLAUDE.md](inference_api/CLAUDE.md) | Job orchestration, workflow routing, IPC protocol |
| `object_detection/` | [object_detection/CLAUDE.md](object_detection/CLAUDE.md) | GroundingDINO engine, tiled detection, DINOv2 reranking |
| `segmentation/` | [segmentation/CLAUDE.md](segmentation/CLAUDE.md) | UNet, SAM zero-shot, SAM-LoRA training & inference |
| `data_split/` | [data_split/CLAUDE.md](data_split/CLAUDE.md) | Embedding-based anti-leakage dataset splitting |
| `tools/` | [tools/CLAUDE.md](tools/CLAUDE.md) | Scripts evaluation, data prep, semi-labeling |
| `tools/ui/editor_app/` | [tools/ui/editor_app/CLAUDE.md](tools/ui/editor_app/CLAUDE.md) | PySide6 desktop app: canvas, controllers, stores |
| `app/` | [app/CLAUDE.md](app/CLAUDE.md) | Electron + React app: workflow runner, result viewer |
| `runtime_lib/` | [runtime_lib/CLAUDE.md](runtime_lib/CLAUDE.md) | SQLite metrics/logging store |

## Entry points (console scripts)

```bash
damage-editor              # PySide6 desktop app (main UI)
damage-dino                # DINO detection CLI
damage-unet                # UNet segmentation CLI
damage-unet-train          # UNet training
damage-sam                 # SAM zero-shot CLI
damage-sam-finetune        # SAM-LoRA finetune CLI
damage-sam-finetune-train  # SAM-LoRA training
damage-models              # Download model weights
damage-cropper             # ROI cropping tool (PySide6)
damage-streamlit-demo      # Streamlit demo
```

## Device detection

`device_utils.py` và `torch_runtime.py` ở root cung cấp:
```python
from device_utils import select_device_str
device = select_device_str("auto")  # → "cuda" | "mps" | "cpu"
```
Tất cả model params đều nhận `device: str` — luôn dùng hàm này thay vì hardcode.

## Models directory

```
models/
  grounding-dino-base/      # GroundingDINO checkpoint (HuggingFace format)
  dinov2-small/             # DINOv2 embedder
  surface_crack_image_detection/  # DINOv2 fine-tuned classifier
  sam_vit_h_4b8939.pth      # SAM ViT-H checkpoint
  sam_vit_b_01ec64.pth      # SAM ViT-B checkpoint
```
