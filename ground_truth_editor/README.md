# GroundTruthEditor (PredictTools)

Chọn ảnh gốc + chọn mask (0/1 hoặc 0/255), rồi tô/xoá bằng brush.

## Chạy

Từ thư mục `DamageDetector`:

```powershell
python ground_truth_editor/main.py
```

## Điều khiển

- Chuột trái: tô (mask = 1)
- Giữ `Ctrl` + chuột trái: xoá (mask = 0)
- Lăn chuột: cuộn ảnh
- Giữ `Shift` + lăn chuột: cuộn trái/phải (nếu chuột chỉ có cuộn dọc)
- Giữ `Ctrl` + lăn chuột: zoom ảnh
- Giữ `Ctrl` + `Shift` + lăn chuột: tăng/giảm brush size

## Phím tắt nhanh

- Thanh trên: `Predict ...` sẽ hỏi detect ảnh hiện tại hay cả folder.
- `PgUp` / `PgDn`: ảnh trước / ảnh sau (khi đã mở folder)
- `Ctrl+K`: focus ô lọc danh sách ảnh trong Explorer
- `Ctrl+1` / `Ctrl+2` / `Ctrl+3`: chuyển nhanh Overlay / Image / Mask
- `Ctrl+4`: focus tab Explorer (panel trái)
- `Ctrl+,`: mở dialog Settings (SAM / DINO / UNet)
- `Alt+1` / `Alt+2` / `Alt+3`: mở nhanh Settings SAM / DINO / UNet
- `Ctrl+.`: dừng tác vụ đang chạy (Stop)

## Ghi nhớ cài đặt

- Settings của SAM / DINO / UNet được lưu lại và tự khôi phục khi mở ứng dụng.
