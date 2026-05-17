# CLAUDE.md — data_split

This file provides guidance to Claude Code when working in `data_split/`.

## Vai trò của module

`data_split` giải quyết **vấn đề data leakage** khi split dataset: trong các ảnh crack được cắt từ cùng một ảnh gốc (video frame, drone panorama), nếu split ngẫu nhiên thì train/val/test sẽ có ảnh từ cùng nguồn → model học cách nhận dạng nguồn ảnh thay vì crack.

**Giải pháp:** Nhóm ảnh theo `source_id` (nguồn gốc), cluster theo visual similarity (DINOv2 embeddings), rồi assign **cả nhóm** vào một split duy nhất.

---

## 5-bước pipeline

```
discover_samples()        → list[SampleRecord]  (images + masks)
       ↓
embed_images()            → (N, 768) float32    (DINOv2 CLS embeddings, cached)
       ↓
group_embeddings()        → DataFrame           (trung bình embedding theo source_id)
       ↓
cluster_sources()         → DataFrame + cluster_id  (MiniBatchKMeans)
       ↓
assign_sources()          → {source_id: split_name} + summary_df  (greedy cost)
       ↓
export_split_folders()    → train/val/test folders
write_workbook()          → split_assignments.xlsx
```

---

## File-by-file

### types.py
```python
@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    mask_path: str | None
    stem: str               # tên file không extension
    source_id: str          # key nhóm ảnh cùng nguồn
    positive_ratio: float   # % pixel trắng trong mask (độ dày crack)
```

### config.py
```python
@dataclass
class SplitConfig:
    input_root: Path           # thư mục có images/ và masks/ bên trong
    output_root: Path
    splits: dict[str, float]   # {"train": 0.7, "val": 0.15, "test": 0.15}
    num_clusters: int = 10
    checkpoint: str = "facebook/dinov2-small"  # hoặc local path
    batch_size: int = 32
    device: str = "auto"       # "auto" | "cpu" | "cuda" | "mps"
    mask_threshold: int = 128  # pixel >= threshold → positive
```

### dataset.py — Source ID extraction

`discover_samples()` suy ra `source_id` từ tên file theo quy tắc:
- Bỏ suffix `_dup\d+` (duplicate frames)
- Bỏ phần số cuối nếu có (e.g. `DSC01398__roi118` → source `DSC01398`)
- Kết quả: nhiều ROI crops từ cùng ảnh gốc có cùng `source_id`

**Quan trọng:** Dataset phải có cấu trúc `input_root/images/*.jpg` và `input_root/masks/*.png`. Tên file mask phải match tên file image.

### embedding.py — DINOv2 Embeddings

```python
embed_images(image_paths, checkpoint, batch_size, device_preference, cache_path)
```

- Dùng `transformers.AutoImageProcessor` + `AutoModel` để extract CLS token từ last hidden state
- L2-normalize về unit sphere trước khi cache
- Cache tại `output_root/embedding_cache.pkl` — tự động invalid nếu checkpoint hoặc image list thay đổi
- **Lần chạy đầu chậm** (download model + process tất cả ảnh); lần sau nhanh nhờ cache

### assign.py — Greedy Cost Assignment

**group_embeddings():** Trung bình embeddings của tất cả ảnh có cùng `source_id` → source-level embedding vector

**cluster_sources():** MiniBatchKMeans với `n_clusters = min(len_groups, max(3, requested))`

**assign_sources() — greedy cost minimization:**

Với mỗi source (theo thứ tự image_count giảm dần), tính cost nếu assign vào mỗi split:

```python
cost = size_cost           # (current_ratio - target_ratio)^2 của image count
     + 0.15 × mask_cost    # balance tỷ lệ crack pixels
     + 25  × overflow_penalty  # nếu vượt 103% target → penalty nặng
     + 0.02 × source_balance   # phân đều sources
     + 0.05 × cluster_penalty  # tránh fragmentation (cùng cluster → cùng split)
```

→ chọn split có cost nhỏ nhất.

### export.py — Output

**write_workbook()**: Tạo Excel `split_assignments.xlsx` với 4 sheets:
- `summary`: tổng hợp tỷ lệ thực tế vs target, image/source/cluster counts
- `train`, `val`, `test`: danh sách ảnh từng split với metadata đầy đủ

**export_split_folders()**: Copy ảnh+masks vào:
```
output_root/
  train/images/*.jpg
  train/masks/*.png
  val/images/*.jpg
  val/masks/*.png
  test/images/*.jpg
  test/masks/*.png
```

---

## Chạy từ CLI

```bash
python -m data_split.split_dataset \
  --input-root  ./my_dataset \
  --output-root ./my_dataset_split \
  --train 0.70 --val 0.15 --test 0.15 \
  --num-clusters 10 \
  --checkpoint  facebook/dinov2-small \
  --batch-size  32 \
  --device auto
```

Hoặc qua Python:
```python
from data_split.config import SplitConfig
from data_split.runner import run_split

config = SplitConfig(
    input_root=Path("./my_dataset"),
    output_root=Path("./my_dataset_split"),
    splits={"train": 0.70, "val": 0.15, "test": 0.15},
    num_clusters=10,
)
run_split(config)
```

---

## Lưu ý khi sửa code

- `runner.py` cứng check `split_names == ["train", "val", "test"]` — nếu muốn thay đổi tên splits phải sửa cả export logic.
- Cache path từ `embed_images()` dùng hash của `(checkpoint, sorted(image_paths))` — nếu rename ảnh thì cache sẽ invalid.
- `assign_sources()` là greedy (không phải global optimum) — kết quả phụ thuộc thứ tự source và random state của KMeans. Để kết quả reproducible: đảm bảo `random_state=42` (đã set sẵn).
- `data_sail_lib/` là external library — không sửa code bên trong đó.
- `tools/crack500_split.py` và `tools/prepare_crack500.py` là scripts cụ thể cho Crack500 dataset, không phải general-purpose.
