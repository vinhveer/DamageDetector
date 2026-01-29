# GroundTruthEditor (PredictTools)

Chọn ảnh gốc + chọn mask (0/1 hoặc 0/255), rồi tô/xoá bằng brush.

## Chạy

Từ thư mục `DamageDetector`:

```powershell
python PredictTools/GroundTruthEditor/main.py
```

## Điều khiển

- Chuột trái: tô (mask = 1)
- Giữ `Ctrl` + chuột trái: xoá (mask = 0)
- Lăn chuột: cuộn ảnh
- Giữ `Shift` + lăn chuột: cuộn trái/phải (nếu chuột chỉ có cuộn dọc)
- Giữ `Ctrl` + lăn chuột: zoom ảnh
- Giữ `Ctrl` + `Shift` + lăn chuột: tăng/giảm brush size

