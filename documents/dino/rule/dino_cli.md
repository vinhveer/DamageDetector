# DINO CLI Documentation

Tài liệu này hướng dẫn cách sử dụng Command Line Interface (CLI) cho mô hình DINO trong dự án DamageDetector.

## Lệnh cơ bản
Mô hình DINO có thể được gọi thông qua module Python:
```bash
python -m object_detection.dino [COMMAND] [OPTIONS]
```

## Các lệnh hỗ trợ (Commands)
DINO CLI hỗ trợ các lệnh sau:
1. `warmup`: Khởi động và tải mô hình vào bộ nhớ.
2. `predict`: Chạy dự đoán phát hiện trên một hình ảnh duy nhất.
3. `predict-batch`: Chạy dự đoán trên nhiều hình ảnh.
4. `recursive-detect`: Chạy phát hiện đệ quy trên các vùng hình ảnh (phù hợp cho các chi tiết nhỏ như vết nứt).

## Tham số chung
Các tham số sau có thể sử dụng với mọi lệnh:
- `--checkpoint` (Bắt buộc): Đường dẫn tới tệp checkpoint GroundingDINO hoặc ID mô hình HuggingFace.
- `--config-id`: ID cấu hình GroundingDINO hoặc thư mục cục bộ (mặc định: `auto`).
- `--queries`: Chuỗi các truy vấn văn bản phân tách bằng dấu phẩy (mặc định: `crack`).
- `--box-threshold`: Ngưỡng độ tin cậy cho bounding box (mặc định: `0.25`).
- `--text-threshold`: Ngưỡng độ tin cậy cho văn bản (mặc định: `0.25`).
- `--max-dets`: Số lượng box phát hiện tối đa (mặc định: `20`).
- `--device`: Thiết bị chạy mô hình (`auto`, `cpu`, `cuda`, `mps`).
- `--output-dir`: Thư mục chứa kết quả (mặc định: `results_dino`).
- `--roi X1 Y1 X2 Y2`: Vùng quan tâm (Region of Interest) để dự đoán.
- `--pretty`: Định dạng đầu ra JSON cho dễ đọc.

## Tham số cụ thể cho từng lệnh

### `predict`
Yêu cầu tham số:
- `--image`: Đường dẫn tới hình ảnh cần dự đoán.

### `predict-batch`
Yêu cầu tham số:
- `--images`: Danh sách các đường dẫn hình ảnh cần dự đoán (phân tách bằng khoảng trắng).

### `recursive-detect`
Phát hiện đệ quy bằng cách cắt ảnh thành các phần nhỏ hơn để tìm chi tiết.
Yêu cầu và tham số bổ sung:
- `--image`: Đường dẫn tới hình ảnh cần dự đoán.
- `--target-label`: Bộ lọc nhãn; có thể lặp lại hoặc dùng giá trị phân tách bằng dấu phẩy.
- `--max-depth`: Độ sâu tối đa của đệ quy (mặc định: `3`).
- `--min-box-px`: Kích thước hộp nhỏ nhất tính bằng pixel để dừng đệ quy (mặc định: `48`).

## Ví dụ sử dụng
```bash
python -m object_detection.dino predict --checkpoint weights/groundingdino.pth --image data/test.jpg --queries "crack, damage" --box-threshold 0.3
```
