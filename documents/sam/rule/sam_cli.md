# SAM CLI Documentation

Tài liệu này hướng dẫn cách sử dụng Command Line Interface (CLI) cho mô hình SAM (Segment Anything Model) trong dự án DamageDetector.

## Lệnh cơ bản
Mô hình SAM có thể được gọi thông qua module Python:
```bash
python -m sam [COMMAND] [OPTIONS]
```

## Các lệnh hỗ trợ (Commands)
SAM CLI hỗ trợ các lệnh sau:
1. `warmup`: Khởi động và tải mô hình vào bộ nhớ.
2. `predict`: Chạy SAM để phân vùng toàn bộ hình ảnh tự động (Auto-masking).
3. `predict-batch`: Chạy SAM tự động phân vùng trên nhiều hình ảnh.
4. `segment-boxes`: Chạy SAM kết hợp với box prompting (Sử dụng bounding box làm gợi ý để phân vùng).

## Tham số chung
Các tham số sau có thể sử dụng với mọi lệnh:
- `--checkpoint` (Bắt buộc): Đường dẫn tới tệp checkpoint của SAM.
- `--sam-model-type`: Loại mô hình SAM (`auto`, `vit_b`, `vit_l`, `vit_h`). Mặc định: `auto`.
- `--invert-mask`: Đảo ngược giá trị của mask đầu ra.
- `--min-area`: Diện tích nhỏ nhất của mask để giữ lại, các vùng nhỏ hơn sẽ bị loại bỏ.
- `--dilate`: Số vòng lặp giãn nở (dilation) áp dụng lên mask để làm dày vùng được phân đoạn.
- `--device`: Thiết bị chạy mô hình (`auto`, `cpu`, `cuda`, `mps`).
- `--output-dir`: Thư mục chứa kết quả (mặc định: `results_sam`).
- `--roi X1 Y1 X2 Y2`: Vùng quan tâm (Region of Interest) để dự đoán.
- `--pretty`: Định dạng đầu ra JSON cho dễ đọc.

## Tham số cụ thể cho từng lệnh

### `predict`
Chạy tự động phân đoạn (Everything mode).
Yêu cầu tham số:
- `--image`: Đường dẫn tới hình ảnh cần dự đoán.

### `predict-batch`
Yêu cầu tham số:
- `--images`: Danh sách các đường dẫn hình ảnh cần dự đoán (phân tách bằng khoảng trắng).

### `segment-boxes`
Phân đoạn dựa trên bounding box đầu vào.
Yêu cầu tham số:
- `--image`: Đường dẫn tới hình ảnh.
- `--boxes-json`: Tệp JSON chứa các thông tin detections/boxes dùng làm prompt cho SAM. Định dạng JSON cần chứa mảng các objects với trường `box` `[x1, y1, x2, y2]`.

## Ví dụ sử dụng
```bash
python -m sam segment-boxes --checkpoint weights/sam_vit_b.pth --image data/test.jpg --boxes-json results/boxes.json
```
