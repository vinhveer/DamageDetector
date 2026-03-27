# Kiến trúc Chi tiết Mô hình SAM Finetune

Tài liệu này giải thích chi tiết cấu trúc, cách thức tích hợp các trọng số tinh chỉnh (Delta/PEFT) và chiến lược chia lưới (Tiled Inference) của module `sam_finetune`.

## 1. Cơ sở Lý thuyết (PEFT trong SAM)

Mô hình SAM gốc được huấn luyện trên tập dữ liệu SA-1B khổng lồ (hơn 11 triệu ảnh), sở hữu khả năng tổng quát hóa cực tốt. Tuy nhiên, nó đôi khi gặp khó khăn với các miền dữ liệu chuyên biệt như y tế hay khuyết tật vật liệu (bê tông, kim loại).

Thay vì fine-tune toàn bộ hàng trăm triệu tham số của SAM (tốn kém và dễ gây catastrophic forgetting), dự án áp dụng **Parameter-Efficient Fine-Tuning (PEFT)**.
Module này nhắm vào việc sửa đổi cấu trúc của bộ mã hóa hình ảnh (Image Encoder - ViT) bằng cách thêm các trọng số cực nhỏ gọn (gọi là **Delta weights**).

### Sơ đồ Luồng xử lý (Pipeline)

```javascript
graph TD
    A[Hình ảnh đầu vào] --> B{Chế độ Predict}
    
    B -- legacy_full_box --> C[Đưa toàn bộ ảnh vào Mô hình]
    B -- tile_full_box --> D[Cắt ảnh thành các Tile có Overlap]
    
    subgraph SAM Finetune Model
        E[ViT Image Encoder]
        F[LoRA / Adapter Delta Weights]
        G[Mask Decoder]
        E -.->|Áp dụng Delta| F
        E --> G
    end
    
    C --> E
    D -->|Từng Tile| E
    
    G -->|Score Map| H[Hòa trộn Tile - Blending]
    H --> I[Áp dụng Threshold tùy chỉnh]
    I --> J[Mask Nhị phân]
```

## 2. Quản lý và Nạp Trọng số Delta (`SamFinetuneRunner`)

Quá trình tải mô hình diễn ra theo 2 bước:

1. Tải SAM gốc (ví dụ: `sam_vit_b_01ec64.pth`) thông qua `load_sam_model`.
2. Áp dụng Delta thông qua hàm `apply_delta_to_sam`.

### Các loại Delta (Delta Types)

- **LoRA (Low-Rank Adaptation)**:
  - Bổ sung các ma trận hạng thấp (low-rank) vào các lớp chiếu $Q$ (Query), $K$ (Key), $V$ (Value) của cơ chế Multi-Head Attention trong các khối Transformer.
  - Các tham số chi phối: `rank` (hạng của ma trận, thường là 4 hoặc 8) và `scaling_factor`.
- **Adapter**:
  - Chèn một mạng Bottleneck nhỏ (gồm 2 lớp Linear giảm và tăng chiều) vào sau mỗi khối Transformer (hoặc song song với lớp MLP).
  - Tham số chi phối: `middle_dim` (chiều không gian trung gian, thường là 32).
- **Both**: Kết hợp cả cơ chế LoRA và Adapter vào cùng một kiến trúc.

*Cơ chế Cache*: Lớp runner lưu lại `_delta_sig` (signature gồm đường dẫn checkpoint, type, rank, dim). Nếu request mới có cùng signature, mô hình sẽ không phải load lại, tối ưu thời gian phản hồi.

## 3. Chiến lược Suy luận Toàn Ảnh (Inference Modes)

Khác với SAM gốc chỉ có Auto-Masking (rải điểm) và Box Prompting, bản SAM Finetune thường được huấn luyện để phân đoạn trực tiếp đối tượng mà không cần điểm (hoạt động giống Semantic Segmentation). Dự án cung cấp hai chế độ (`predict_mode`):

### Chế độ `legacy_full_box`

- Hệ thống tạo một Bounding Box giả định bao phủ toàn bộ bức ảnh: `box=[0, 0, width-1, height-1]`.
- Truyền vào mô hình qua `predictor.predict`.
- Phương pháp này chỉ phù hợp với các ảnh có kích thước nhỏ hoặc vuông vức. Đối với ảnh chữ nhật dài, tỷ lệ khung hình sẽ bị bóp méo khi đưa qua Image Encoder, làm giảm nghiêm trọng độ chính xác.

### Chế độ Tiled Inference (`tile_full_box`) - Khuyên dùng

Giải quyết bài toán ảnh lớn bằng cách chia nhỏ hình ảnh:

1. **Chia lưới (Tiling)**: Ảnh lớn được chia thành các ô nhỏ kích thước `tile_size` x `tile_size` (thường là 512x512 hoặc 1024x1024).
2. **Chồng lấn (Overlap)**: Các ô được đặt đè lên nhau một khoảng `tile_overlap` (thường là 50%). Việc này giúp tránh hiện tượng "đứt gãy" mask ở ranh giới giữa các ô.
3. **Suy luận Từng ô**: Hàm `tiled_score_map` đưa từng ô qua mô hình để lấy ra "bản đồ điểm số" (score map) dạng float32 thay vì mask nhị phân cứng.
4. **Hòa trộn (Blending)**: Các điểm số được dán lại vào một bản đồ lớn bằng kích thước ảnh gốc. Ở các vùng chồng lấn, hệ thống sử dụng trọng số hòa trộn (như cửa sổ Hanning - Hanning Window) để làm mượt điểm số, vùng trung tâm của ô có trọng số cao hơn vùng rìa.

## 4. Ngưỡng Phân vùng (Thresholding)

Điểm khác biệt quan trọng của SAM Finetune là **Threshold**.
SAM mặc định luôn binarize ở mức logit `0.0`. Nhưng mô hình đã finetune có thể dịch chuyển phân phối điểm số này.

- Cơ chế `resolve_predict_threshold`:
  - Nếu `threshold="auto"`, hệ thống sẽ tự động tìm tệp `best_threshold.txt` trong cùng thư mục với checkpoint delta (được sinh ra trong quá trình Validation lúc huấn luyện).
  - Score Map tổng hợp sẽ được áp dụng: `chosen = (score_map >= threshold).astype(np.uint8)`.

## 5. Box Prompting với Mô hình đã Finetune

Hàm `segment_boxes` trong `sam_finetune` gọi trực tiếp `_segment_boxes_with_predictor` của module `sam` gốc, nhưng truyền vào đối tượng `predictor` đã bị biến đổi kiến trúc (đã gắn Delta).
Nhờ vậy, nó kế thừa toàn bộ tính năng lọc, clipping, dilation của SAM gốc, nhưng chất lượng phân vùng cho vết nứt (hoặc domain chuyên biệt) được cải thiện vượt trội.
