# CLAUDE.md — runtime_lib

This file provides guidance to Claude Code when working in `runtime_lib/`.

## Vai trò của module

`runtime_lib` cung cấp **persistent storage** cho metrics và logs của các long-running inference jobs. Đây là module nhỏ (1 file thực tế) nhưng được dùng rộng rãi bởi evaluation scripts trong `tools/`.

---

## sqlite_store.py

### SQLiteRunStore

SQLite-backed store với schema linh hoạt (bảng được tạo động):

```python
store = SQLiteRunStore(db_path="results.db")

# Tạo bảng và insert data
store.ensure_table("unet_metrics", columns=["filename", "dice", "iou", "threshold"])
store.insert_rows("unet_metrics", [
    ("img001.jpg", 0.87, 0.79, 0.5),
    ("img002.jpg", 0.91, 0.85, 0.5),
])
store.insert_row("unet_metrics", ("img003.jpg", 0.83, 0.76, 0.5))
```

**Type inference tự động:** Column name chứa "name", "path", "label", "message", "type" → TEXT; các column khác → REAL.

**Thread-safe:** `threading.Lock()` bảo vệ mọi write operations.

**WAL mode:** `PRAGMA journal_mode=WAL` cho phép concurrent reads từ nhiều processes.

### SQLiteLogHandler

Tích hợp với Python `logging` module:

```python
import logging
from runtime_lib import SQLiteLogHandler

handler = SQLiteLogHandler("job_logs.db")
logging.getLogger().addHandler(handler)

# Logs được ghi vào bảng "logs" với columns: created_at, level, logger, message
# Buffered: flush mỗi 20 records
```

---

## Conventions khi dùng

- Mỗi evaluation run nên tạo 1 SQLite file riêng (không share giữa các runs)
- Table name tự động được normalize: alphanumeric + underscore, prefix `t_` nếu bắt đầu bằng số
- Không hỗ trợ UPDATE/DELETE — append-only design (phù hợp cho metrics logging)
