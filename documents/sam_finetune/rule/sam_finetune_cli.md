# SAM Finetune CLI Documentation

Tài liệu này hướng dẫn cách sử dụng Command Line Interface (CLI) cho mô hình SAM Finetune trong dự án DamageDetector. Đây là phiên bản SAM đã được tinh chỉnh bằng các phương pháp như LoRA hoặc Adapter để phù hợp hơn với tác vụ phát hiện hư hỏng/vết nứt.

## Lệnh cơ bản
```bash
python -m segmentation.sam.finetune [COMMAND] [OPTIONS]
```

## Các lệnh hỗ trợ (Commands)
1. `warmup`: Khởi động và tải mô hình gốc cùng với các trọng số tinh chỉnh (delta) vào bộ nhớ.
2. `predict`: Chạy SAM Finetune tự động phân vùng toàn bộ hình ảnh.
3. `predict-batch`: Chạy phân vùng trên nhiều hình ảnh.
4. `segment-boxes`: Sử dụng box prompting để phân vùng.

## Tham số chung
- `--checkpoint` (Bắt buộc): Đường dẫn tới checkpoint SAM gốc (ví dụ: `sam_vit_b_01ec64.pth`).
- `--sam-model-type`: Loại mô hình SAM (`auto`, `vit_b`, `vit_l`, `vit_h`).
- `--delta-type` (Bắt buộc): Loại tinh chỉnh đã áp dụng. Các giá trị hợp lệ: `adapter`, `lora`, `both`.
- `--delta-checkpoint`: Đường dẫn tới checkpoint tinh chỉnh (mặc định: `auto`).
- `--middle-dim`: Kích thước chiều trung gian (dùng cho adapter/lora). Mặc định: `32`.
- `--scaling-factor`: Yếu tố mở rộng (scaling factor) cho LoRA. Mặc định: `0.2`.
- `--rank`: Hạng (rank) của ma trận trong LoRA. Mặc định: `4`.
- `--invert-mask`: Đảo ngược giá trị mask đầu ra.
- `--min-area`: Diện tích nhỏ nhất của mask để giữ lại.
- `--dilate`: Số vòng lặp giãn nở mask.
- `--device`: `auto`, `cpu`, `cuda`, `mps`.
- `--output-dir`: Thư mục lưu kết quả.
- `--predict-mode`: Chế độ dự đoán (`auto`, `tile_full_box`, `legacy_full_box`). Mặc định: `auto`.
- `--tile-size`: Kích thước mỗi ô (tile) khi dùng `tile_full_box` (Mặc định -1: dùng từ metadata hoặc 512).
- `--tile-overlap`: Độ chồng lấn giữa các ô (Mặc định -1).
- `--threshold`: Ngưỡng tạo mask (dạng float) hoặc `auto` để lấy từ tệp `best_threshold.txt`.
- `--roi X1 Y1 X2 Y2`: Vùng quan tâm (ROI).
- `--pretty`: In đầu ra JSON cho dễ đọc.

## Ví dụ sử dụng
```bash
python -m segmentation.sam.finetune predict \
    --checkpoint weights/sam_vit_b.pth \
    --delta-type lora \
    --delta-checkpoint weights/sam_lora.pt \
    --image data/test.jpg \
    --predict-mode tile_full_box
```
