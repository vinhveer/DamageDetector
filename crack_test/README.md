# crack_test

Run out-domain crack detection and box-guided segmentation for selected ROI images.

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
python crack_test/run_crack_test.py --device auto
```

Defaults:

- Images are resolved from `../data/HinhAnh/damage_scan.sqlite3` by `image_id`.
- Output is written to `../model_with_inference/crack_test`.
- Pipelines:
  - `stable_dino_sam_finetune`: Stable DINO + SAM-LoRA HQ coarse/refine.
  - `yolo_unet`: YOLO + UNet.

Each pipeline/image folder contains:

- `overlay_box_only.png`
- `overlay_box_label.png`
- `overlay_segmentation.png`
- `overlay_segmentation_box_label.png`
- `mask.png`
