# U-Net CLI Documentation

Tài liệu này hướng dẫn cách sử dụng Command Line Interface (CLI) cho mô hình U-Net trong dự án DamageDetector.

## Lệnh cơ bản
```bash
python -m segmentation.unet [COMMAND] [OPTIONS]
```

## Các lệnh hỗ trợ (Commands)
1. `warmup`: Khởi động và tải mô hình U-Net vào bộ nhớ.
2. `predict`: Chạy mô hình U-Net trên một hình ảnh duy nhất.
3. `predict-batch`: Chạy mô hình trên nhiều hình ảnh.
4. `run-rois`: Chạy mô hình U-Net trên các vùng ROI (Region of Interest) cụ thể của một hình ảnh thay vì toàn bộ hình ảnh.

## Tham số chung
- `--model` (Bắt buộc): Đường dẫn tới tệp checkpoint của U-Net.
- `--output-dir`: Thư mục chứa kết quả đầu ra. Mặc định: `results_unet`.
- `--threshold`: Ngưỡng nhị phân hóa xác suất để tạo mask. Mặc định: `0.5`.
- `--no-postprocessing`: Tắt các bước hậu xử lý (ví dụ: làm mượt, xóa vùng nhỏ).
- `--mode`: Phương pháp xử lý kích thước ảnh khi suy luận. Các lựa chọn: `tile`, `letterbox`, `resize`. Mặc định: `tile`.
- `--input-size`: Kích thước ảnh đầu vào cho mô hình (ảnh vuông). Mặc định: `512`.
- `--tile-overlap`: Độ chồng lấn (pixel) giữa các tile trong chế độ `tile`. Mặc định: `0` (sẽ dùng `input_size // 2`).
- `--tile-batch-size`: Số lượng tile xử lý cùng lúc. Mặc định: `4`.
- `--device`: Thiết bị chạy mô hình (`auto`, `cpu`, `cuda`, `mps`).
- `--roi X1 Y1 X2 Y2`: Vùng quan tâm (ROI) cắt ảnh trước khi chạy U-Net.
- `--pretty`: In kết quả JSON định dạng đẹp.

## Tham số cụ thể cho từng lệnh

### `predict`
- `--image`: Đường dẫn hình ảnh cần dự đoán.

### `predict-batch`
- `--images`: Danh sách hình ảnh phân cách bằng khoảng trắng.

### `run-rois`
Chạy suy luận riêng biệt cho từng vùng ROI trên ảnh và kết hợp chúng lại trên một mask chung.
- `--image`: Đường dẫn hình ảnh.
- `--roi-box X1 Y1 X2 Y2`: Tọa độ ROI. Có thể lặp lại nhiều lần để định nghĩa nhiều ROI.

## Ví dụ sử dụng
```bash
python -m segmentation.unet predict \
    --model weights/unet_resnet34.pth \
    --image data/test.jpg \
    --mode tile \
    --input-size 512 \
    --threshold 0.6
```
