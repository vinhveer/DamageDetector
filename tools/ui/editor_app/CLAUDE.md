# CLAUDE.md — tools/ui/editor_app

This file provides guidance to Claude Code when working in `tools/ui/editor_app/`.

## Vai trò của module

`editor_app` là **desktop application chính** (PySide6/Qt) cho inference workflow:
- Load ảnh, vẽ ROI, chạy prediction (DINO + SAM/UNet)
- Xem kết quả (masks, overlays, detections) ngay trên canvas
- Quản lý run history
- So sánh ground truth vs predictions
- Isolate/crop đối tượng từ ảnh

Chạy bằng: `damage-editor` (entry point) hoặc `python -m tools.ui.editor_app`

---

## Kiến trúc 5 lớp

```
Domain (models.py)          ← frozen dataclasses, không depend on Qt
    ↑
Stores (stores/)            ← QObject + Qt signals, state container
    ↑
Services (services/)        ← I/O, storage, metrics, không có Qt widgets
    ↑
Controllers (controllers/)  ← business logic, wire stores + services + inference_api
    ↑
UI (ui/)                    ← Qt widgets, nhận signals từ stores, gọi controllers
```

**Nguyên tắc:** Code ở lớp dưới không được import từ lớp trên. Services không biết gì về Qt widgets. Controllers không contain widget code.

---

## Stores — Reactive State

Stores là `QObject` có Qt signals. UI components kết nối vào signals để cập nhật tự động.

| Store | State chứa |
|-------|-----------|
| `WorkspaceStore` | Image path, mask, detections, ROI list, image list |
| `PredictionStore` | Active job list, logs per job, events |
| `UiStore` | Active workspace, layout flags, overlay opacity |
| `HistoryStore` | List run summaries từ disk |
| `CompareStore` | Ground truth vs prediction comparison data |
| `IsolateStore` | Isolation workflow results |

```python
# Pattern điển hình khi đọc/cập nhật store
workspace_store.image_path = "/path/to/img.jpg"
workspace_store.image_changed.emit()  # notify UI

# Kết nối trong UI
workspace_store.image_changed.connect(self._on_image_changed)
```

---

## Controllers — Business Logic

Controllers orchestrate stores + services. Không chứa widget code.

| Controller | Trách nhiệm |
|-----------|-------------|
| `PredictionController` | Submit job → poll events → cập nhật store → notify UI |
| `WorkspaceController` | Load image/folder, navigate images |
| `EditorController` | ROI management, detection editing |
| `HistoryController` | Refresh + load run details từ disk |
| `CompareController` | Load ground truth, compute comparison metrics |
| `IsolateController` | Submit isolate workflow, display results |

### PredictionController flow (quan trọng nhất)

```
user trigger PredictionActions.run_predict_dialog()
    ↓
PredictionController.submit(config, settings)
    ↓
build_prediction_request(config, settings, image_path, output_dir)  ← inference_api
    ↓
RunStorageService.plan_run() → tạo run directory structure
RunStorageService.materialize_run() → ghi run.json
    ↓
inference_api.submit(request) → job_id
PredictionStore.add(JobRecord(job_id=..., status="queued"))
    ↓
_poll_timer.start()  ← QTimer interval 180ms
    ↓
[mỗi 180ms] inference_api.drain_events(job_id)
  → log events → PredictionStore.append_log()
  → partial_result events → hiện thi DINO boxes trước SAM
    ↓
[job done] PredictionStore.set_final_payload()
→ MainWindow._on_prediction_job_completed()
→ load detections vào EditorWorkspace
→ Canvas hiện thị masks + bounding boxes
```

---

## Services

| Service | Vai trò |
|---------|---------|
| `RunStorageService` | Tạo run dirs, ghi/đọc run.json + result.json |
| `SettingsService` | Persist settings (model paths, thresholds) vào JSON file |
| `FileService` | Image/mask file I/O |
| `ExportService` | Export overlay images, detection boxes |
| `CompareService` | Load + compute comparison metrics |
| `metrics.py` | IoU, Dice, pixel accuracy |

### Run directory structure

```
{results_root}/
  {timestamp}_{workflow}_{short_id}/    ← run directory
    run.json       ← metadata: workflow, status, created_at, image_path
    request.json   ← InferenceRequest serialized
    result.json    ← InferenceResult payload
    data/          ← masks, overlays, per-image results
```

---

## UI Layer

### MainWindow

Root Qt widget. Wires tất cả controllers + stores + workspaces lại với nhau. Khởi tạo trong `app.py`.

### Workspaces (QStackedWidget)

6 workspaces được switch qua `TopBar`:

| Workspace | Nội dung |
|-----------|---------|
| `EditorWorkspace` | Canvas + ExplorerPanel + ImageToolsPanel |
| `RunsWorkspace` | Active job monitor + log viewer |
| `HistoryWorkspace` | Historical run browser |
| `CompareWorkspace` | GT vs prediction comparison viewer |
| `IsolateWorkspace` | Isolation results |
| `SettingsWorkspace` | Model paths, prediction config |

### ImageCanvas (canvas.py)

Custom Qt widget cho image viewing:
- Zoom + pan bằng mouse wheel + drag
- ROI drawing (box selection)
- Brush painting (mask editing)
- Overlay rendering (masks, detections, bounding boxes)
- OpenCV/numpy images → QPixmap conversion

### Prediction Dialog (ui/dialogs/predict.py)

Dialog cho user chọn:
- Task group: Crack Only | More Damage
- Segmentation model: SAM | SAM LoRA | UNet
- Detection model: DINO | None
- Scope: Current image | Whole folder

---

## config/prediction_settings.py

`DEFAULT_EDITOR_SETTINGS` là dict chứa tất cả default values:
```python
{
    "sam_checkpoint": "",
    "dino_checkpoint": "IDEA-Research/grounding-dino-base",
    "dino_config_id": "IDEA-Research/grounding-dino-base",
    "unet_model": "",
    "sam_lora_checkpoint": "",
    "more_damage_crack_mask_model": "off",   # "off" | "sam_lora" | "unet"
    ...
}
```

`migrate_editor_settings(old_settings)` handle backward compatibility khi thêm settings mới.

---

## Lưu ý khi sửa code

- **Không import Qt widgets vào controllers/services/stores** — phá vỡ testability (xem test trong `tests/test_dino_valid_tiling.py` test controllers mà không cần Qt display).
- **PredictionStore** là source of truth cho job status — UI không giữ state riêng về jobs.
- Poll timer (180ms) là trade-off: đủ nhanh để UI responsive nhưng không quá tốn CPU. Không cần giảm interval trừ khi có lý do rõ ràng.
- **RunStorageService.plan_run()** tạo run directory **trước** khi submit — nếu submit fail thì directory empty còn đó. Là intentional (dễ debug).
- Canvas sử dụng `Qt.SmoothTransformation` cho zoom — nếu sửa rendering logic, test với ảnh lớn (>10MP) để kiểm tra performance.
- Settings được lưu tại `~/.config/DamageDetector/settings.json` (qua `paths.py`).
