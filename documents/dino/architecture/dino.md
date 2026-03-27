# Kiến trúc Chi tiết Mô hình DINO (GroundingDINO)

Tài liệu này cung cấp cái nhìn chuyên sâu về kiến trúc, luồng xử lý dữ liệu và các thuật toán đặc tả được triển khai cho mô hình DINO trong dự án `DamageDetector`.

## 1. Tổng quan Hệ thống

Module `dino` đóng vai trò là "mắt thần" ban đầu của hệ thống. Nó sử dụng mô hình **GroundingDINO**, một mạng nơ-ron kết hợp giữa Transformer thị giác (Vision Transformer) và mô hình ngôn ngữ (Language Model) để phát hiện đối tượng dựa trên bất kỳ văn bản truy vấn nào (Zero-shot / Open-vocabulary Object Detection).

Mục tiêu chính trong dự án: Tìm kiếm và khoanh vùng sơ bộ các vị trí có khả năng chứa vết nứt (crack) hoặc hư hỏng (damage) trên hình ảnh, tạo tiền đề (Bounding Boxes) cho các mô hình phân vùng (như SAM, U-Net) hoạt động.

### Sơ đồ Luồng xử lý (Pipeline)



```javascript
graph TD
    subgraph Input
        A[Hình ảnh] 
        B[Text Queries]
    end
    C[DinoRunner / Processor]
    D[GroundingDINO Model]
    E[Post-processing Thresholds]
    
    A --> C
    B --> C
    C --> D
    D --> E
    
    subgraph Recursive Detect
        F[Cắt Crop theo Box]
        G{Kiểm tra điều kiện dừng}
        H[NMS & Lọc hộp chứa]
    end
    
    E -->|Phát hiện đối tượng| F
    F --> G
    G -->|Chưa dừng| C
    G -->|Đã dừng| H
    H --> Final[Bounding Boxes Cuối cùng]
```

## 2. Quản lý Trạng thái & Tải Mô hình (`DinoRunner`)

Lớp `DinoRunner` được thiết kế dưới dạng Singleton/Stateful Runner nhằm giữ mô hình trong bộ nhớ (VRAM/RAM), tránh việc phải tải lại mô hình trong mỗi lần gọi.

### Cơ chế Tải Ngoại tuyến (Offline Loading)

Dự án được thiết kế để hoạt động ổn định trong môi trường **Air-gapped** (không có internet). `DinoRunner.ensure_model_loaded` thực hiện:

- Thiết lập các biến môi trường: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- Hỗ trợ đa dạng định dạng tệp:
  - Thư mục mô hình HuggingFace hoàn chỉnh (chứa `config.json`, weights).
  - Tệp `.safetensors` hoặc `.pth`.
- Nếu chỉ cung cấp tệp `.pth` trần, hệ thống sử dụng tham số `--config-id` (hoặc tự động suy luận `grounding-dino-tiny` / `grounding-dino-base`) để tải bộ cấu hình (config) và bộ tiền xử lý (processor) tương ứng từ bộ nhớ cache cục bộ.
- **Tiền xử lý Checkpoint (\_strip\_prefix\_if\_present)**: Tự động loại bỏ các tiền tố như `module.` hoặc `model.` trong `state_dict` nếu checkpoint được huấn luyện bằng Distributed Data Parallel (DDP).

## 3. Pipeline Xử lý Dự đoán Tiêu chuẩn (`predict`)

Khi người dùng chạy ở chế độ `predict` cơ bản, luồng dữ liệu như sau:

1. **Chuẩn bị ảnh**: Hình ảnh được đọc bằng OpenCV (`cv2`), chuyển sang không gian màu RGB và định dạng `PIL.Image`.
2. **Tiền xử lý Text (normalize\_queries)**:
   - Tách các truy vấn (VD: "crack", "spalling").
   - Chuyển chữ thường, loại bỏ khoảng trắng thừa và truy vấn trùng lặp.
   - Nối thành một câu mô tả duy nhất phân cách bằng dấu chấm (VD: `"crack. spalling."`).
3. **Mã hóa (Encoding)**:
   - `processor` chuyển đổi ảnh và text thành `input_ids`, `attention_mask` và các tensor ảnh.
4. **Suy luận (Forward Pass)**: Tensor được đưa qua `GroundingDinoForObjectDetection`.
5. **Hậu xử lý (post\_process\_gdino)**:
   - Trích xuất bounding boxes và điểm số (scores).
   - Lọc bỏ các kết quả thấp hơn `box_threshold` và `text_threshold`.
   - Kết quả được gói gọn trong dataclass `Det(label, box_xyxy, score)`.

## 4. Thuật toán Phát hiện Đệ quy (`predict_recursive`)

Phát hiện các vết nứt cực nhỏ trên ảnh có độ phân giải khổng lồ (ví dụ 4K, 8K) là một thách thức lớn vì GroundingDINO thường nén ảnh đầu vào xuống kích thước nhỏ (ví dụ 800x1333), làm mất hoàn toàn chi tiết vết nứt.

Dự án giải quyết bằng thuật toán **Recursive Zoom Detect (\_recursive\_zoom\_detect)**:

### Bước 1: Tính toán Bridge ROI

Thay vì quét toàn bộ ảnh (thường chứa rất nhiều vùng đen vô ích), thuật toán dùng NumPy (`np.where(gray > nonblack_thresh)`) để tìm bounding box tối thiểu chứa toàn bộ các điểm ảnh có thông tin.

### Bước 2: Quét Đệ quy (Tree-based Search)

- Đưa vùng ảnh hiện tại (Crop) vào DINO.
- Nếu tìm thấy đối tượng (vết nứt):
  - Hệ thống ghi nhận tọa độ (được chiếu ngược về tọa độ gốc của ảnh lớn).
  - Trích xuất crop của vết nứt đó, tiếp tục đệ quy (phóng to) đi sâu vào trong (`current_depth + 1`) để tìm các nhánh nứt nhỏ hơn.
- Điều kiện dừng (Base Cases):
  - Đạt tới `max_depth` (mặc định: 3).
  - Kích thước crop nhỏ hơn `min_box_px` (mặc định: 48 pixels).
  - Không phát hiện thêm đối tượng nào.

### Bước 3: Lọc Hộp chứa (Parent Containment Filter)

Quá trình đệ quy sinh ra rất nhiều hộp lồng nhau (hộp mẹ bao quanh các hộp con). Hàm `_filter_parent_boxes` loại bỏ các hộp mẹ nếu phần diện tích giao với hộp con chiếm quá `parent_contain_threshold` (mặc định 0.7), giữ lại các hộp con có chi tiết bám sát vết nứt nhất.

### Bước 4: Non-Maximum Suppression (NMS)

Áp dụng hàm `_nms_boxes` tự viết (sử dụng IoU Threshold, mặc định 0.5) để gộp các hộp chồng chéo lẫn nhau do quá trình cắt lưới sinh ra, đảm bảo mỗi vị trí vết nứt chỉ có một Bounding Box duy nhất với `score` cao nhất.

## 5. Cấu trúc Tham số (`DinoParams`)

Toàn bộ cấu hình được đóng gói bất biến (frozen dataclass) `DinoParams` để đảm bảo tính an toàn thread-safe:

- `text_queries`: Các nhãn cần tìm.
- `box_threshold`, `text_threshold`: Ngưỡng độ tin cậy.
- `roi_box`: Giới hạn không gian tìm kiếm tĩnh (nếu có).
- Các tham số đệ quy: `nms_iou_threshold`, `parent_contain_threshold`, `recursive_min_box_px`, `recursive_max_depth`.
