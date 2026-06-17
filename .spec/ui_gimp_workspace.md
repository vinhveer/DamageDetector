# UI — GIMP-style Image Processing Workspace

Spec cho app Qt mới ở [`DamageDetector/ui/`](../ui/). Mục tiêu: biến nó thành
**image processing workspace** giống GIMP / Photoshop / CVAT-lite, không chỉ là
một tool detect ROI đơn lẻ. App phải đủ tổng quát để vừa làm
detection (DINO/StableDINO/YOLO) vừa làm segmentation (SAM/SAM-LoRA/UNet) vừa
làm review/annotation, có quản lý layer, tool palette, undo/redo, project state,
và job manager.

App chuẩn để chạy bằng:

```bash
.venv/bin/python -m ui
```

Console script `damage-editor` trong [`setup.py`](../setup.py) hiện đang trỏ
vào `ui.editor_app:main` cũ — sẽ chuyển sang `ui.app.application:run` ở phase
cuối, sau khi parity với app hiện tại đủ chắc.

---

## 1. Mục tiêu

1. Một desktop app PySide6 duy nhất cho toàn bộ workflow ảnh damage:
   open → annotate → detect → segment → review → export.
2. Layout giống image editor chuyên nghiệp: menu bar, top toolbar, left tool
   palette, central canvas, right inspector + layers, bottom job/log/console.
3. Mọi thao tác trên canvas đi qua **Tool system** (1 tool active tại 1 thời
   điểm) — không hard-code logic chuột vào canvas như app cũ.
4. Mọi overlay (ROI, detection box, mask, measurement) là **Layer** có thể
   bật/tắt, đổi opacity, lock, reorder.
5. Mọi job ML chạy trong **Job Manager** — cancellable, có progress, có log,
   có thể chạy nhiều job song song.
6. Reuse [`inference_api/`](../inference_api/) cho detect/segment thay vì viết
   pipeline mới.
7. Undo/redo theo **Command pattern** áp dụng cho mọi action thay đổi state.
8. Lưu/load **Project** (.ddproj) chứa image references, ROI, masks, settings.

---

## 2. Layout tổng thể

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Menu: File | Edit | View | Image | Detect | Segment | Layer | Window | ?│
├──────────────────────────────────────────────────────────────────────────┤
│ Toolbar: Open Save | Undo Redo | Pan Zoom Fit | Run Cancel Export       │
├────────┬────────────────────────────────────────────────┬────────────────┤
│ Tools  │                                                │ Inspector tabs │
│ ┌────┐ │                                                │ ┌──┬──┬──┬──┐ │
│ │ ☞ │ │                                                │ │Im│De│Se│Ob│ │
│ │ □ │ │                                                │ ├──┴──┴──┴──┤ │
│ │ ▭ │ │            Image Canvas                        │ │           │ │
│ │ ◯ │ │            (zoom + pan + overlays)              │ │ inspector │ │
│ │ ✎ │ │                                                │ │ panel     │ │
│ │ ⌫ │ │                                                │ │           │ │
│ │ ⊕ │ │                                                │ │           │ │
│ │ 📏 │ │                                                │ ├───────────┤ │
│ └────┘ │                                                │ │ Layers    │ │
│        │                                                │ │ ☑ Image   │ │
│ Tool   │                                                │ │ ☑ ROIs    │ │
│ opts   │                                                │ │ ☑ Boxes   │ │
│ ┌────┐ │                                                │ │ ☑ Masks   │ │
│ │ …  │ │                                                │ │ □ Measure │ │
│ └────┘ │                                                │ └───────────┘ │
├────────┴────────────────────────────────────────────────┴────────────────┤
│ Bottom dock tabs: [Jobs] [Log] [History] [Console]                       │
└──────────────────────────────────────────────────────────────────────────┘
│ Statusbar: zoom%  | cursor xy | image size | active tool | job status   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Cấu trúc thư mục đề xuất

```
ui/
├── __init__.py
├── __main__.py
├── main.py                     # entry: parse args, gọi application.run()
├── app/
│   ├── application.py           # QApplication bootstrap, theme/translations
│   ├── main_window.py           # QMainWindow shell: menu/toolbar/dock layout
│   └── shortcuts.py             # global keyboard shortcuts
├── core/
│   ├── settings.py              # UiSettings dataclass + persistence
│   ├── signals.py               # WorkspaceSignals event bus (Qt signals)
│   ├── commands.py              # Command base + UndoStack
│   ├── workspace.py             # current image / project state
│   └── project.py               # Project I/O (.ddproj JSON)
├── canvas/
│   ├── canvas_view.py           # QGraphicsView subclass với pan/zoom/gestures
│   ├── scene.py                 # QGraphicsScene chứa các Layer items
│   ├── items/
│   │   ├── image_item.py        # base image pixmap layer
│   │   ├── roi_item.py          # ROI rectangle (movable, resizable, labeled)
│   │   ├── box_item.py          # detection box (read-only, color theo class)
│   │   ├── mask_item.py         # segmentation mask (alpha blend)
│   │   └── measurement_item.py  # đo khoảng cách / diện tích
│   └── tools/
│       ├── base.py              # Tool interface: mousePress/Move/Release/Key
│       ├── pan_tool.py
│       ├── select_tool.py       # chọn 1 hoặc nhiều graphics items
│       ├── rect_roi_tool.py     # vẽ ROI rectangle (đã có ở app cũ)
│       ├── polygon_roi_tool.py  # vẽ ROI dạng polygon
│       ├── brush_tool.py        # vẽ thêm/sửa mask
│       ├── eraser_tool.py
│       ├── point_prompt_tool.py # SAM positive/negative point prompt
│       ├── crop_tool.py
│       └── measure_tool.py
├── panels/
│   ├── tools_palette.py         # left dock: tool buttons + tool options
│   ├── layers_panel.py          # right dock tab: Layer tree, visibility, alpha
│   ├── inspector_panel.py       # right dock tab: Image / Detect / Segment / Object
│   ├── jobs_panel.py            # bottom dock tab: job manager view
│   ├── log_panel.py             # bottom dock tab: streaming log
│   ├── history_panel.py         # bottom dock tab: undo history
│   └── console_panel.py         # bottom dock tab: lệnh ad-hoc (optional)
├── services/
│   ├── job_manager.py           # quản lý jobs trên QThread(Pool)
│   ├── inference_client.py      # wrap inference_api InferenceApi singleton
│   ├── detect_process.py        # ROI detect via subprocess (đã có)
│   ├── segment_service.py       # wrap segmentation pipelines
│   ├── image_io.py              # load/save image, EXIF, color profile
│   └── export_service.py        # overlay PNG, CSV, COCO JSON, mask PNG
├── models/                      # data classes (Pydantic / dataclass)
│   ├── annotation.py            # ROI, Box, Mask, Measurement
│   ├── detection.py             # DetectionRow, DetectionRun
│   ├── layer.py                 # LayerNode tree
│   ├── job.py                   # JobSpec, JobStatus, JobUpdate
│   └── project.py               # ProjectFile schema
├── resources/
│   ├── icons/                   # SVG/PNG icons cho tool palette + toolbar
│   ├── translations/            # .qm files cho i18n (later)
│   └── qrc/                     # Qt resource collections
└── widgets/                     # small reusable Qt widgets
    ├── color_swatch.py
    ├── slider_with_input.py
    ├── path_picker.py
    └── badge.py
```

---

## 4. Workspace state

`ui/core/workspace.py` giữ state hiện tại của app:

```python
@dataclass
class WorkspaceState:
    project_path: Path | None
    image_path: Path | None
    image_size: tuple[int, int] | None
    image_dpi: tuple[float, float] | None

    layers: LayerTree
    active_layer_id: str | None

    active_tool: str = "pan"
    tool_options: dict[str, Any] = field(default_factory=dict)

    selection: list[str] = field(default_factory=list)   # graphics item ids

    jobs: list[JobSpec] = field(default_factory=list)
    active_job_id: str | None = None

    detection_runs: list[DetectionRun] = field(default_factory=list)
    segmentation_runs: list[SegmentationRun] = field(default_factory=list)

    settings: UiSettings = field(default_factory=UiSettings)
```

`Workspace` lớp wrapper expose Qt signals:

```python
class Workspace(QObject):
    imageChanged = Signal(Path)
    layersChanged = Signal()
    activeToolChanged = Signal(str)
    selectionChanged = Signal(list)
    jobsChanged = Signal()
    projectDirty = Signal(bool)
```

Mọi panel subscribe các signal này, không panel nào tự đi đọc state khác.

---

## 5. Layer system

`ui/models/layer.py`:

```python
class LayerKind(StrEnum):
    image = "image"
    rois = "rois"
    detections = "detections"
    masks = "masks"
    measurements = "measurements"
    overlay = "overlay"

@dataclass
class LayerNode:
    id: str
    kind: LayerKind
    name: str
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    z_order: int = 0
    children: list["LayerNode"] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)  # graphics items thuộc layer
    metadata: dict = field(default_factory=dict)
```

Default layer tree khi mở 1 ảnh:

```
- Image                (locked=True, opacity=1.0)
- Annotations
    - ROIs
    - Detection boxes
    - Segmentation masks
    - Measurements
- Overlays (custom)
```

`LayersPanel` render layer tree (`QTreeView` + custom delegate). Mỗi row có:
- checkbox visibility
- icon kind
- tên layer
- opacity slider (0–100%)
- lock toggle
- context menu: rename, delete, duplicate, merge down, export

Khi visibility đổi → set `setVisible()` cho mọi `QGraphicsItem` thuộc layer.
Khi opacity đổi → set `setOpacity()`.
Khi reorder → đổi `setZValue()` theo z_order mới.

---

## 6. Tool system

`ui/canvas/tools/base.py`:

```python
class Tool(QObject):
    name: str = ""
    icon: str = ""               # path / qrc id
    cursor: Qt.CursorShape = Qt.CursorShape.ArrowCursor
    options_widget: QWidget | None = None  # render trong Tool Options panel

    def activate(self, canvas: "CanvasView") -> None: ...
    def deactivate(self) -> None: ...
    def mousePress(self, event: QMouseEvent, scene_pos: QPointF) -> bool: ...
    def mouseMove(self,  event: QMouseEvent, scene_pos: QPointF) -> bool: ...
    def mouseRelease(self, event: QMouseEvent, scene_pos: QPointF) -> bool: ...
    def keyPress(self, event: QKeyEvent) -> bool: ...
```

`CanvasView` chỉ delegate event cho tool active hiện tại; không tự xử lý drag
draw/select/brush nữa.

Tool registry: `ui/canvas/tools/__init__.py` export `TOOL_REGISTRY` dict
`{name: Tool}`. `ToolsPalette` đọc registry để build buttons.

Tool options widget được swap vào panel "Tool Options" bên dưới palette mỗi
khi đổi tool active. Ví dụ:

- Brush tool: slider radius, opacity, hardness
- Rect ROI tool: snap-to-grid, fixed aspect, label preset
- SAM point prompt tool: positive/negative toggle, min mask area

---

## 7. Command / Undo-Redo

`ui/core/commands.py`:

```python
class Command(ABC):
    label: str
    @abstractmethod
    def redo(self) -> None: ...
    @abstractmethod
    def undo(self) -> None: ...

class UndoStack(QObject):
    pushed = Signal(Command)
    cursorChanged = Signal(int)
    def push(self, cmd: Command) -> None: ...
    def undo(self) -> None: ...
    def redo(self) -> None: ...
    def history(self) -> list[Command]: ...
```

Mọi mutation đáng kể đều phải gói thành Command:

- AddRoiCommand / DeleteRoiCommand / MoveRoiCommand / ResizeRoiCommand
- AddDetectionRunCommand / RemoveDetectionRunCommand
- ToggleLayerVisibilityCommand
- SetLayerOpacityCommand
- BrushStrokeCommand (mask)
- ApplyDetectResultsCommand (sau khi job xong)
- ApplySegmentResultsCommand

`HistoryPanel` hiển thị stack list, click → `setIndex(...)` để jump.

Đảm bảo Command stateless để undo idempotent (lưu cả before/after).

---

## 8. Job manager

`ui/services/job_manager.py`:

- `JobSpec`: kind (detect/segment/export), params, callbacks, source layer
- `JobStatus`: queued | running | completed | failed | cancelled
- `JobUpdate`: progress (0..1), message, partial result
- Nội bộ dùng `QThread` hoặc `QThreadPool` + `QRunnable`
- Tạo bridge tới [`inference_api.api.InferenceApi`](../inference_api/api.py)
  qua `inference_client.py`

API:

```python
class JobManager(QObject):
    jobAdded = Signal(JobSpec)
    jobUpdated = Signal(str, JobUpdate)
    jobFinished = Signal(str, JobStatus, object)  # (job_id, status, result)

    def submit(self, spec: JobSpec) -> str: ...
    def cancel(self, job_id: str) -> None: ...
    def terminate(self, job_id: str) -> None: ...
    def jobs(self) -> list[JobSpec]: ...
```

`JobsPanel` (bottom dock):

| col      | nội dung                        |
|----------|--------------------------------|
| #        | id ngắn                         |
| Kind     | detect / segment / export       |
| Source   | tên layer / image               |
| Status   | queued / running / done / fail  |
| Progress | progress bar nhỏ                |
| Time     | elapsed                         |
| Actions  | Cancel / Open result / Re-run   |

Double-click row → focus log của job đó ở `LogPanel`.

Run nhiều job song song được, mỗi job có cancel riêng.

---

## 9. Inspector panel

Right dock có 4 tab:

### 9.1. Image
- File path, name, size, color mode
- Zoom % (đồng bộ canvas)
- Quick actions: Fit, Actual Size, Rotate 90°, Flip H/V

### 9.2. Detect
- Detector: gdino / stabledino / yolo (combobox)
- Box / text threshold
- Max dets
- Tile scales (multi-select chips)
- Class filter / prompt groups (link tới
  [`pineline.lib.step_gdino_detect.prompts`](../pineline/lib/step_gdino_detect/prompts.py))
- Show score ≥
- Run / Cancel buttons → dùng `JobManager`
- Sau khi run: hiện table detection (giống app cũ)

### 9.3. Segment
- Backend: SAM zero-shot / SAM-LoRA / UNet
- Model checkpoint picker
- Device (auto/cpu/cuda/mps)
- Mode-specific options:
  - SAM: multimask, min area, expand box
  - UNet: input_size, overlap, threshold, post-processing
- Mask overlay opacity
- Run / Cancel

### 9.4. Object properties
Hiện khi user select ≥ 1 item trên canvas:
- Item kind (ROI / box / mask)
- Class / label (editable)
- Score (read-only)
- Bbox xyxy
- Layer parent
- Delete / Lock / Hide / Send to layer …

---

## 10. Menu bar

```
File
  New Project        Cmd+N
  Open Image…        Cmd+O
  Open Project…      Cmd+Shift+O
  Open Folder…
  Recent Files       ▶
  ────────
  Save Project       Cmd+S
  Save Project As…   Cmd+Shift+S
  Export Results…    Cmd+E
  Export Mask PNG…
  Export COCO JSON…
  ────────
  Quit               Cmd+Q

Edit
  Undo               Cmd+Z
  Redo               Cmd+Shift+Z
  ────────
  Cut / Copy / Paste
  Delete Selection   Del
  Select All         Cmd+A
  Clear ROIs
  Preferences…       Cmd+,

View
  Fit to Window      F
  Actual Size        Cmd+1
  Zoom In / Out      Cmd+= / Cmd+-
  ────────
  Toggle Layers Panel
  Toggle Inspector
  Toggle Jobs Panel
  Toggle Log
  Reset Layout

Image
  Image Info…
  Rotate / Flip
  Crop to Selection
  Resize…
  Adjust ▶  (Brightness, Contrast, Levels)

Detect
  Run Detection      R
  Cancel Detection
  Re-run Last
  Detector Settings…

Segment
  Run Segmentation   Shift+R
  Cancel Segmentation
  Clear Masks

Layer
  New Layer
  Duplicate Layer
  Delete Layer
  Merge Down
  Show / Hide
  Lock / Unlock
  Reorder ▶

Window
  Tools
  Layers
  Inspector
  Jobs / Log / History / Console
  Reset Layout

Help
  Documentation
  About DamageDetector
```

---

## 11. Reuse [`inference_api/`](../inference_api/) thay vì code mới

Đây là điểm rất quan trọng. Hiện tại `ui/services/detect_process.py` đang gọi
trực tiếp [`pineline/roi_detect_app/detect_job.py`](../pineline/roi_detect_app/detect_job.py).
Phải refactor để mọi job ML đi qua `InferenceApi`:

```python
from inference_api.api import get_inference_api
from inference_api.contracts import InferenceRequest

class InferenceClient(QObject):
    eventReceived = Signal(str, object)   # (job_id, JobEvent)

    def __init__(self) -> None:
        super().__init__()
        self._api = get_inference_api()
        self._poller = QTimer(self)
        self._poller.setInterval(60)
        self._poller.timeout.connect(self._drain_events)
        self._poller.start()

    def submit(self, request: InferenceRequest) -> str:
        return self._api.submit(request)

    def cancel(self, job_id: str) -> None:
        self._api.cancel(job_id)

    def terminate(self, job_id: str) -> None:
        self._api.terminate(job_id)

    def _drain_events(self) -> None:
        for event in self._api.drain_events():
            self.eventReceived.emit(event.job_id, event)
```

Workflow ánh xạ:

| User action               | InferenceRequest.workflow |
|---------------------------|---------------------------|
| Detect with DINO          | `sam_only` thay = `dino_only`*|
| Detect + Segment SAM      | `sam_dino`                |
| Detect + Segment SAM-LoRA | `sam_dino_ft`             |
| Segment only SAM          | `sam_only`                |
| Segment only UNet         | `unet_only`               |
| Detect + UNet             | `unet_dino`               |
| Isolate                   | `isolate`                 |

\* hiện chưa có `dino_only` → khi UI cần "Detect only" có thể dùng `sam_dino` nhưng
skip SAM, hoặc thêm workflow mới (xem mục 13.1).

`detect_process.py` chỉ giữ lại làm fallback cho ROI-rectangle quick detection,
không dùng cho luồng chính.

---

## 12. Project file (.ddproj)

```json
{
  "version": 1,
  "image": {
    "path": "/abs/path/img.jpg",
    "size": [4032, 3024],
    "checksum_sha1": "..."
  },
  "layers": [
    {"id":"L_image","kind":"image","name":"Image","visible":true,"opacity":1.0,"locked":true},
    {"id":"L_rois","kind":"rois","name":"ROIs","visible":true,"opacity":1.0,
     "items":[{"id":"R_1","x1":..,"y1":..,"x2":..,"y2":..,"label":"crack"}]},
    {"id":"L_dets","kind":"detections","name":"Detections","visible":true,
     "items":[{"id":"D_1","x1":..,"y1":..,"x2":..,"y2":..,
               "label":"crack","score":0.71,"detector":"gdino"}]},
    {"id":"L_masks","kind":"masks","name":"Masks","visible":true,"opacity":0.45,
     "items":[{"id":"M_1","mask_path":"masks/M_1.png","detection_id":"D_1"}]}
  ],
  "settings": {
    "dino": {"box_threshold":0.25,"text_threshold":0.25,"max_dets":20},
    "sam":  {"checkpoint":".../sam_vit_b_01ec64.pth","model_type":"auto"},
    "unet": {"checkpoint":".../best_model.pth","threshold":0.5}
  },
  "history": {
    "detection_runs": [{"run_id":"...","timestamp":"...","detector":"gdino"}],
    "segmentation_runs":[{"run_id":"...","timestamp":"...","backend":"sam"}]
  }
}
```

Lưu nhị phân nặng (mask PNG) ra cùng thư mục `<project>.ddproj.assets/`.

`ProjectIO`:
- `load(path) -> Project`
- `save(project, path)`
- `migrate(project, from_version, to_version)`

---

## 13. Settings & preferences

Settings nằm trong:
- `~/.damagedetector/ui/settings.json` (per user, persistent)
- Project-level settings nằm trong `.ddproj`

Reuse format hiện có ở `.editor_app.json` (model checkpoints, thresholds), nhưng
dịch sang dạng có namespace:

```json
{
  "models": {
    "dino_checkpoint": "...",
    "sam_checkpoint": "...",
    "sam_lora_checkpoint": "...",
    "unet_checkpoint": "..."
  },
  "thresholds": {
    "box_threshold": 0.25,
    "text_threshold": 0.25,
    "unet_threshold": 0.5
  },
  "device": "auto",
  "ui": {
    "theme": "system",
    "show_grid": false,
    "default_zoom": "fit"
  },
  "recent_files": [],
  "window": {
    "geometry": "...",
    "state": "..."
  }
}
```

`Preferences` dialog (Cmd+,) là tabbed dialog:
- General (theme, locale, shortcuts)
- Models (paths, defaults)
- Detection (default thresholds, max dets)
- Segmentation (default backend, opacity)
- Performance (device, threads, cache size)

---

## 14. Statusbar

```
[ zoom% 47% ▾ ]   [ x: 1234, y: 567 ]   [ 4032×3024 ]   [ Tool: Rect ROI ]   [ Job: detect 3/8 ]
```

- Zoom dropdown để jump (10%, 25%, 50%, 100%, 200%, Fit)
- Cursor xy theo image coords
- Image size
- Active tool tên
- Job status: tóm tắt job đang chạy

---

## 15. Keyboard shortcuts (mặc định)

| Action            | Mac          | Win/Linux    |
|-------------------|--------------|--------------|
| Open Image        | Cmd+O        | Ctrl+O       |
| Save Project      | Cmd+S        | Ctrl+S       |
| Undo / Redo       | Cmd+Z / Shift| Ctrl+Z / Y   |
| Pan tool          | H            | H            |
| Select tool       | V            | V            |
| Rect ROI          | R            | R            |
| Polygon ROI       | P            | P            |
| Brush             | B            | B            |
| Eraser            | E            | E            |
| Point prompt SAM  | T            | T            |
| Crop              | C            | C            |
| Measure           | M            | M            |
| Zoom in / out     | Cmd+= / -    | Ctrl+= / -   |
| Fit               | F            | F            |
| Actual size       | Cmd+1        | Ctrl+1       |
| Run detection     | R            | R            |
| Run segmentation  | Shift+R      | Shift+R      |
| Cancel job        | Cmd+.        | Esc          |
| Toggle layers     | F7           | F7           |
| Toggle inspector  | F8           | F8           |
| Toggle log/jobs   | F9           | F9           |

R bị overload (Rect ROI vs Run detection) → cho phép chỉnh trong Preferences;
mặc định Rect ROI dùng R, Run detection dùng F5.

---

## 16. Style / theming (không QSS)

Theo yêu cầu, **không dùng QSS** ở phase đầu. Theme dựa vào:
- `QApplication.setStyle("Fusion")`
- `QPalette` mặc định, app respect system dark/light
- Icons SVG có 2 phiên bản (light/dark) chọn theo palette luminance
- Spacing/padding cấu hình qua `UiSettings.metrics` (margin, padding, icon size)

Sau này nếu cần look custom, mở 1 phase riêng cho QSS — nhưng phải nằm trong
`ui/resources/styles/*.qss`, app load tùy chọn.

---

## 17. Kế hoạch triển khai (phase từng commit, mỗi bước test được)

### Phase 0 — Skeleton hiện tại
Đã có ở [`ui/`](../ui/). App mở được, có toolbar/dock, ROI rectangle vẽ được,
detect chạy thật qua [`detect_process.py`](../ui/services/detect_process.py).

### Phase 1 — Tool system
- Tạo `ui/canvas/tools/base.py` + `pan_tool.py` + `select_tool.py` +
  `rect_roi_tool.py`
- Refactor `ImageCanvas` để delegate event cho tool active
- Thêm `ToolsPalette` ở left dock (thay nút Draw ROI cũ)
- Verify: switch giữa Pan / Select / Rect ROI mượt, ROI vẽ hoạt động

### Phase 2 — Layer system
- Tạo `ui/models/layer.py`, `LayerTree`
- Tạo `ui/panels/layers_panel.py` (QTreeView với visibility/opacity/lock)
- Refactor scene: ROI / Box / Mask được gắn vào layer
- Verify: tắt/bật layer, đổi opacity, lock layer

### Phase 3 — Inspector tabs
- Tạo `ui/panels/inspector_panel.py` (QTabWidget với 4 tab Image/Detect/
  Segment/Object)
- Migrate `DetectionPanel` cũ vào tab Detect
- Verify: switch tab hoạt động, table detection vẫn render

### Phase 4 — Undo/redo
- `ui/core/commands.py` + `UndoStack`
- Convert ROI add/delete/move/resize sang Command
- Edit menu nối Cmd+Z / Cmd+Shift+Z
- `HistoryPanel` ở bottom dock
- Verify: vẽ 5 ROI, undo từng cái, redo lại

### Phase 5 — Job manager
- `ui/services/job_manager.py` + `JobsPanel`
- Refactor detect button → submit job vào JobManager
- Verify: chạy detect, theo dõi tiến độ ở Jobs panel, cancel hoạt động

### Phase 6 — Inference API integration
- `ui/services/inference_client.py` poll `InferenceApi`
- Map detect button → `InferenceRequest` workflow
- Map segment button → `sam_only` / `sam_dino` / `unet_only`
- Verify: dùng cùng worker subprocess như editor_app cũ, không spawn detect_job
  riêng nữa

### Phase 7 — Project I/O
- `ui/core/project.py` + `models/project.py`
- File menu: New / Open / Save / Save As / Recent
- Verify: vẽ ROI, lưu project, đóng app, mở lại → ROI khôi phục

### Phase 8 — Segmentation panel
- Tab Segment trong inspector
- Mask layer + `MaskItem` blend alpha
- Run SAM zero-shot, SAM-LoRA, UNet qua JobManager
- Verify: 1 ảnh có boxes → run SAM → mask hiện overlay

### Phase 9 — Brush / eraser
- `BrushTool`, `EraserTool` cho mask layer
- BrushStrokeCommand undo/redo
- Verify: tô thêm vùng, undo, redo hoạt động

### Phase 10 — Export
- Export overlay PNG
- Export COCO JSON
- Export mask PNG / mask binary
- Reveal in Finder/Explorer

### Phase 11 — Polish
- Preferences dialog
- Recent files
- Reset layout
- Crash handling: log to `~/.damagedetector/ui/logs/`
- Update [`setup.py`](../setup.py) console script `damage-editor` →
  `ui.app.application:run`
- Update [`CLAUDE.md`](../CLAUDE.md) thêm row `ui/` vào bảng module

---

## 18. Files cần đụng

### Tạo mới
- Toàn bộ thư mục mới ở mục 3 (canvas/, panels/, services/, models/, core/,
  resources/, widgets/)
- `.ddproj` schema docs trong `ui/models/project.py`

### Sửa
- [`ui/app/main_window.py`](../ui/app/main_window.py): thay shell hiện tại
  bằng layout đầy đủ menu + 4 dock
- [`ui/app/application.py`](../ui/app/application.py): wire workspace +
  inference client singleton
- [`ui/widgets/canvas.py`](../ui/widgets/canvas.py): rút logic vẽ ROI ra
  `RectRoiTool`, chỉ giữ pan/zoom + delegate event
- [`ui/services/detect_process.py`](../ui/services/detect_process.py): giữ làm
  fallback, không phải đường chính
- [`setup.py`](../setup.py): cập nhật console script entry point ở Phase 11

### Reuse (không sửa)
- [`inference_api/`](../inference_api/) — toàn bộ
- [`object_detection/`](../object_detection/), [`segmentation/`](../segmentation/) —
  qua inference_api
- [`device_utils.py`](../device_utils.py) — chọn device

---

## 19. Rủi ro & câu hỏi mở

1. **Chia tab vs chia dock**: Inspector có 4 tab dễ chật trên màn nhỏ. Có thể
   cho user kéo tab ra dock riêng (Qt MDI tab tear-off). Phase đầu giữ tab.
2. **Mask render performance**: 1 ảnh 4000×3000 với 50 mask → blend trong
   `QGraphicsView` có thể chậm. Giải pháp: cache mask compositing thành 1
   `QPixmap` khi `MasksLayer` thay đổi, không re-blend mỗi paint.
3. **Brush tool trên mask lớn**: cần work directly trên `QImage` của mask, không
   spam scene update. Áp dụng dirty rect.
4. **Undo cho job kết quả**: kết quả từ `InferenceApi` đến bất đồng bộ. Wrap
   thành `ApplyDetectResultsCommand` lúc job xong; user undo sẽ remove kết quả
   khỏi layer (nhưng job đã chạy xong → chấp nhận).
5. **Cancel job chạy lâu**: `InferenceApi.terminate()` kill subprocess; UI phải
   chịu downtime warmup lại lần sau. Cần show toast "Service restarted".
6. **Project file size**: nếu nhúng mask PNG inline dưới base64 sẽ to. Chọn
   cách tham chiếu file ngoài ở `<project>.ddproj.assets/`.
7. **Multi-image (folder/batch)**: phase đầu single image. Phase sau thêm
   "Image Browser" panel, mỗi ảnh là 1 project entry trong file `.ddproj`.
8. **Console script overlap**: `damage-editor` đang trỏ tới
   [`tools/ui/editor_app/`](../tools/ui/editor_app/) cũ. Phase 11 chuyển sang
   `ui/`. Trong giai đoạn chuyển tiếp, tạm thêm `damage-editor-next` →
   `ui.app.application:run` để chạy song song không đụng app cũ.
9. **PySide6 version**: lock minor version trong [`requirements.txt`](../requirements.txt)
   để tool registry / Qt6 native gestures không vỡ giữa các bản.
10. **Drag-drop ảnh từ Finder**: feature dễ; thêm ở Phase 1 cùng tool system.
11. **Coordinate convention**: image coords (top-left origin) khắp nơi; chuyển
    sang scene coords chỉ ở `CanvasView`. Tránh mismatch.

---

## 20. Test plan tối thiểu

- Mở 1 ảnh từ [`HinhAnh/`](../../data/HinhAnh) bằng File → Open Image → ảnh
  hiện trong canvas, layer "Image" auto-add.
- Chuyển tool sang Rect ROI, vẽ 3 ROI, kiểm tra layer "ROIs" có 3 item, undo
  3 lần → ROIs biến mất, redo 3 lần → hiện lại.
- Chạy Detect (DINO) → Job Manager hiện job running, log streaming, sau xong
  layer "Detections" có boxes, table inspector populate.
- Tắt visibility layer "Detections" → boxes biến mất khỏi canvas, table
  vẫn còn.
- Chạy Segment SAM trên các box → layer "Masks" có mask, alpha slider hoạt
  động.
- Save Project → Quit → Open Project → state khôi phục đầy đủ (ROIs, boxes,
  masks).
- Cancel detect khi đang chạy → job status = cancelled, không có result được
  apply.
- Chuyển model checkpoint trong Preferences → run lại detect → output đổi.
- Drag-drop 1 ảnh khác từ Finder → confirm dialog "Save current project?",
  chọn No → app load ảnh mới.

---

## 21. Quyết định đã chốt

- **Stack**: PySide6 + QGraphicsView/Scene; không Qt Quick / QML.
- **Theme**: không QSS phase đầu; `Fusion` style + system palette.
- **Orchestration**: reuse [`inference_api/`](../inference_api/); không viết
  pipeline mới trong UI.
- **Job execution**: thread-based via `JobManager`, IPC qua subprocess đã có.
- **State**: workspace singleton + Qt signals; không Redux/store ngoài.
- **Persistence**: project file `.ddproj` JSON + asset folder bên cạnh.
- **Undo**: Command pattern, `UndoStack` toàn cục, history panel để jump.
- **Layer**: tree (parent → children), mỗi layer chứa graphics items theo id.
- **Tools**: tool registry, 1 tool active, options widget swap.
- **Console script**: thêm `damage-editor-next` → `ui.app.application:run` ở
  giai đoạn parity; rename thành `damage-editor` khi sẵn sàng.

---

## 22. Backend mapping chi tiết

### 22.1. Map UI action → InferenceRequest

Mọi nút Run trong app phải build đúng `InferenceRequest` để [`InferenceApi`](../inference_api/api.py) submit. Bảng tham chiếu:

| UI action | workflow | image_path | roi_box | params keys | Notes |
|-----------|----------|-----------|---------|-------------|-------|
| Detect (DINO) | `sam_dino` | image hiện tại | None hoặc bbox của ROI active | `dino` | nếu UI cần "DINO only" thì thêm workflow `dino_only` (xem 22.4) |
| Detect (StableDINO) | `sam_dino` | như trên | như trên | `dino.detector="stabledino"`, `dino.checkpoint=...` | |
| Segment SAM (zero-shot, từ box) | `sam_only` | image hiện tại | None | `sam.boxes=[…]`, `sam.checkpoint`, `sam.model_type` | boxes là list xyxy lấy từ Layer Detections hoặc Layer ROIs |
| Segment SAM-LoRA (từ box) | `sam_only_ft` | như trên | None | `sam.checkpoint`, `sam.lora_checkpoint`, `sam.lora_*` | |
| Detect + Segment SAM | `sam_dino` | như trên | None | `dino`, `sam` | DINO ra box, SAM segment luôn |
| Detect + Segment SAM-LoRA | `sam_dino_ft` | như trên | None | `dino`, `sam` (LoRA) | |
| Segment UNet only | `unet_only` | image hiện tại | None | `unet.checkpoint`, `unet.threshold`, `unet.input_size`, `unet.mode` | mode = `tile` / `full` |
| Detect + Segment UNet | `unet_dino` | như trên | None | `dino`, `unet` | |
| Isolate damage | `isolate` | image hiện tại | None | `isolate.labels`, `isolate.crop`, `isolate.outside_white` | |
| Run scan với prompt groups | `sam_dino` | image hiện tại | None | `dino.prompt_groups=[{name,prompt}]` | dùng `pineline.lib.step_gdino_detect.prompts` |

### 22.2. UI side responsibilities

- `request_builder.py` đã tồn tại trong [`inference_api/request_builder.py`](../inference_api/request_builder.py): UI build `PredictionConfig` rồi gọi builder, không tự ráp dict params.
- Ánh xạ `Inspector → Detect/Segment` form trực tiếp ra `PredictionConfig` (giữ shape giống editor cũ trong [`tools/ui/editor_app/config/prediction_settings.py`](../tools/ui/editor_app/config/prediction_settings.py)).
- Khi user chọn "Run" trên Layer ROIs có nhiều ROI rectangle: app không submit nhiều job, mà submit nhiều `InferenceRequest` (mỗi ROI một request) qua `JobManager`. Job manager group thành 1 "batch job" hiển thị 1 dòng trong Jobs panel với progress = sum.

### 22.3. Reuse từ [`tools/ui/editor_app/`](../tools/ui/editor_app/)

| Code path cũ | Vai trò | Reuse mode |
|--------------|---------|-----------|
| [`controllers/prediction_controller.py`](../tools/ui/editor_app/controllers/prediction_controller.py) | submit job, drain events | refactor → `services/inference_client.py` |
| [`controllers/workspace_controller.py`](../tools/ui/editor_app/controllers/workspace_controller.py) | manage current image / settings | refactor → `core/workspace.py` |
| [`controllers/history_controller.py`](../tools/ui/editor_app/controllers/history_controller.py) | run history (DB) | move sang `services/run_storage.py` (file `services/run_storage.py` cũ giữ nguyên ở [`tools/ui/editor_app/services/run_storage.py`](../tools/ui/editor_app/services/run_storage.py)) |
| [`controllers/isolate_controller.py`](../tools/ui/editor_app/controllers/isolate_controller.py) | isolate workflow | gắn vào tab Inspector → Segment, dùng workflow `isolate` |
| [`controllers/compare_controller.py`](../tools/ui/editor_app/controllers/compare_controller.py) | compare runs | port sang `panels/jobs_panel.py` "Compare runs" sub-action |
| [`services/file_service.py`](../tools/ui/editor_app/services/file_service.py) | image / mask IO | reuse trực tiếp |
| [`services/export_service.py`](../tools/ui/editor_app/services/export_service.py) | save overlay/CSV | reuse trực tiếp |
| [`services/settings_service.py`](../tools/ui/editor_app/services/settings_service.py) | load/save .editor_app.json | port sang `core/settings.py` (xem 28) |
| [`stores/*`](../tools/ui/editor_app/stores) | Qt signal bus per-domain | gộp vào `core/signals.py` + `core/workspace.py` |
| [`canvas.py`](../tools/ui/editor_app/canvas.py) | canvas + ROI items | refactor split ra `canvas/canvas_view.py`, `canvas/items/*`, `canvas/tools/rect_roi_tool.py` |
| [`color_utils.py`](../tools/ui/editor_app/color_utils.py) | class color | reuse |

### 22.4. Workflow mới đề xuất

Nếu UI có "Detect only" thật sự không cần SAM/UNet, thêm workflow:

- `dino_only` — chỉ chạy DINO, trả `DetectionResult` (không gọi SAM).
- File: thêm handler trong [`inference_api/workflows.py`](../inference_api/workflows.py) + đăng ký trong [`inference_api/workflow_resolver.py`](../inference_api/workflow_resolver.py).

Workflow này không bắt buộc cho v1. v1 dùng `sam_dino` rồi UI ẩn mask layer.

---

## 23. Annotation model

`ui/models/annotation.py`:

```python
class AnnotationKind(StrEnum):
    roi = "roi"            # vùng do user vẽ tay
    box = "box"            # detection từ model
    mask = "mask"          # mask segmentation
    point = "point"        # SAM prompt point
    polygon = "polygon"    # ROI dạng polygon
    measurement = "measurement"
    text = "text"

@dataclass(frozen=True)
class BBox:
    x1: float; y1: float; x2: float; y2: float   # image coords, top-left origin

@dataclass
class RoiAnnotation:
    id: str                    # "R_<uuid>"
    bbox: BBox
    label: str = ""
    color: str = "#FFC629"
    layer_id: str = "L_rois"
    locked: bool = False
    metadata: dict = field(default_factory=dict)

@dataclass
class BoxAnnotation:
    id: str                    # "D_<uuid>"
    bbox: BBox
    label: str
    score: float
    detector: str              # "gdino" | "stabledino" | "yolo" | ...
    run_id: str
    layer_id: str = "L_dets"
    metadata: dict = field(default_factory=dict)

@dataclass
class MaskAnnotation:
    id: str                    # "M_<uuid>"
    mask_path: str             # asset relative path
    bbox: BBox                 # bounding box của mask trong image coords
    score: float = 0.0
    source_box_id: str | None = None
    backend: str = "sam"       # "sam" | "sam_lora" | "unet"
    run_id: str = ""
    layer_id: str = "L_masks"
    area: int = 0
    metadata: dict = field(default_factory=dict)

@dataclass
class PointPromptAnnotation:
    id: str
    x: float; y: float         # image coords
    polarity: int              # +1 positive, -1 negative
    layer_id: str = "L_points"

@dataclass
class MeasurementAnnotation:
    id: str
    points: list[tuple[float, float]]
    kind: str                  # "length" | "area"
    layer_id: str = "L_measure"
```

Convention bắt buộc:

- **Toạ độ luôn là image space** (top-left origin, pixel-aligned). `CanvasView` chuyển sang scene coords chỉ ở lớp render.
- **bbox format**: xyxy (`x1<x2`, `y1<y2`).
- **Mask lưu**: PNG grayscale 8-bit (0/255) trong `<project>.ddproj.assets/masks/<mask_id>.png`. Dùng `bbox` để biết mask đặt ở đâu, không lưu mask full-image (trừ khi UNet trả full-image — khi đó `bbox` = full image rect).
- **ID stable**: tạo bằng `uuid4().hex[:8]` prefix theo kind. Không dựa vào index để link giữa command và item.
- **Class labels**: chuẩn hoá lowercase ASCII (`crack`, `mold`, `spall`). UI hiển thị label thân thiện qua bảng map (xem 28.4).

Mapping annotation ↔ layer ↔ graphics item:

```
Annotation (data)  ──►  Layer (group)  ──►  QGraphicsItem (visual)
  RoiAnnotation         L_rois              RoiRectItem
  BoxAnnotation         L_dets              BoxRectItem
  MaskAnnotation        L_masks             MaskPixmapItem
  PointPromptAnn        L_points            PointMarkerItem
```

Repository `AnnotationStore` (đặt trong `core/workspace.py`) giữ dict id→annotation; `Layer.item_ids` chỉ là chỉ mục con. Khi serialize project, mỗi layer dump kèm `items` đầy đủ annotation đó.

---

## 24. Layer model — bổ sung

Mở rộng từ mục 5:

### 24.1. Operations

| Op | Behavior |
|----|----------|
| add | append vào parent, sinh id `L_<kind>_<uuid8>` |
| remove | xoá layer + tất cả annotation con (hỏi confirm nếu có item) |
| rename | đổi `name`, không đổi `id` |
| duplicate | clone deep, tất cả annotation con sinh id mới |
| merge_down | merge với layer ngay dưới cùng kind; mask + annotations gộp |
| reorder | thay đổi `z_order`, các sibling re-index liên tục |
| solo | chỉ layer này visible, các sibling tạm tắt (lưu visibility cũ vào metadata) |
| toggle_visibility | flip `visible` |
| set_opacity | 0.0–1.0 |
| set_locked | True khoá tương tác (no select / no move) |
| send_to | chuyển 1 annotation sang layer khác cùng kind |

### 24.2. Blend mode

Phase đầu chỉ `normal` (alpha blend). Phase sau hỗ trợ:

- `multiply` — tô mask đỏ trên ảnh sáng
- `screen` — overlay sáng
- `difference` — debug compare runs

Blend mode lưu trong `LayerNode.metadata.blend_mode`.

### 24.3. Layer rendering

Render được implement như sau:

- mỗi layer có 1 `QGraphicsItemGroup` parent
- items con đặt vào group
- `setVisible/setOpacity` apply trên group → tự propagate
- z_order = `setZValue(layer.z_order * 1000 + item_index)`
- mask layer dùng `MaskCompositeItem` cache 1 `QPixmap` (giải pháp ở mục 19.2)

---

## 25. Tool state machine

Mỗi tool có state machine rõ ràng. Định nghĩa chuẩn:

```
Tool: <name>
- States: <list>
- Inputs: mouse(press/move/release/wheel), key(press/release), focus(enter/leave)
- Transitions: <state> --[event]--> <state>
- Visual feedback: cursor / overlay / handles
- Commits: emit Command nào khi nào
```

### 25.1. PanTool

```
States: idle → panning
idle --[LMB press]--> panning
panning --[mouse move]--> panning  (translate viewport)
panning --[LMB release]--> idle
Cursor: open hand → closed hand khi panning
Commits: none (không tạo Command)
```

### 25.2. SelectTool

```
States: idle → marquee → moving
idle --[LMB press on item]--> moving
idle --[LMB press empty]--> marquee
marquee --[move]--> marquee  (rubber band rect)
marquee --[release]--> idle  (chọn item trong rect)
moving --[move]--> moving  (translate selected items)
moving --[release]--> idle  (commit MoveAnnotationsCommand)
Keys: Esc cancel, Del DeleteAnnotationsCommand, Cmd+A SelectAll
Visual: 8 handle resize quanh selection
Commits: MoveAnnotationsCommand, ResizeAnnotationCommand, DeleteAnnotationsCommand
```

### 25.3. RectRoiTool

```
States: idle → drawing
idle --[LMB press in image]--> drawing
drawing --[move]--> drawing  (rubber rect)
drawing --[release]--> idle  (commit AddRoiCommand nếu w*h ≥ MIN_AREA)
drawing --[Esc]--> idle  (abort)
Cursor: crosshair
Visual: dashed yellow rubber rect, label preview "#N"
Settings: min_size, snap_to_pixel, fixed_aspect, default_label
Commits: AddRoiCommand
```

### 25.4. PolygonRoiTool

```
States: idle → adding_points → closing
idle --[LMB click]--> adding_points  (anchor first vertex)
adding_points --[move]--> adding_points  (preview line tới chuột)
adding_points --[LMB click]--> adding_points  (push vertex)
adding_points --[double-click | Enter | click first vertex]--> idle  (close + commit)
adding_points --[Esc]--> idle  (abort)
adding_points --[Backspace]--> adding_points  (pop last vertex)
Visual: vertices markers + preview polyline
Commits: AddPolygonRoiCommand
```

### 25.5. BrushTool

```
States: idle → painting
idle --[LMB press on mask layer]--> painting
painting --[move]--> painting  (paint stroke vào QImage mask, dirty rect)
painting --[release]--> idle  (commit BrushStrokeCommand với before/after diff)
Settings: radius, hardness, opacity, mode (paint | erase), color = white/black
Visual: cursor circle preview với radius
Commits: BrushStrokeCommand (1 command per stroke, không per move)
```

### 25.6. EraserTool

Như BrushTool nhưng force `mode=erase`.

### 25.7. PointPromptTool (SAM)

```
States: idle
idle --[LMB click]--> idle  (add positive point, commit AddPointCommand)
idle --[Shift+LMB click]--> idle  (add negative point)
idle --[Alt+click on point]--> idle  (delete point)
Visual: green dot positive, red dot negative, số thứ tự
Commits: AddPointCommand, DeletePointCommand
Side effect: sau mỗi click, optionally trigger live SAM preview job (debounced 300ms)
```

### 25.8. CropTool

```
States: idle → adjusting → committed
idle --[click on image]--> adjusting  (rect 8-handle xuất hiện covering image)
adjusting --[drag handle]--> adjusting  (resize crop rect)
adjusting --[Enter]--> committed  (commit CropImageCommand thay image)
adjusting --[Esc]--> idle  (abort)
Visual: dim ngoài crop rect
Commits: CropImageCommand (replaces image, undoable nhưng nặng — lưu original ref)
```

### 25.9. MeasureTool

```
States: idle → measuring
idle --[LMB click]--> measuring  (anchor point 1)
measuring --[move]--> measuring  (preview line + length label)
measuring --[LMB click]--> idle  (commit AddMeasurementCommand)
Settings: unit (px | mm if DPI known), kind (length | area)
Commits: AddMeasurementCommand
```

### 25.10. Cross-cutting rules

- Mọi tool tôn trọng `active_layer.locked` — nếu locked, abort thao tác và show toast.
- `Esc` ở mọi tool → abort thao tác hiện tại, không chuyển tool.
- `Space` giữ tạm Pan tool, thả ra trở về tool cũ (giống Photoshop).
- Tool option widget swap qua `ToolsPalette.set_active_tool(name)` → emit `Workspace.activeToolChanged`.

---

## 26. Command pattern — chi tiết

### 26.1. Base

```python
class Command(ABC):
    label: str = ""              # hiển thị trong History panel
    timestamp: float = 0.0
    merge_key: str | None = None # để merge với command kế tiếp cùng key
    @abstractmethod
    def redo(self, ws: Workspace) -> None: ...
    @abstractmethod
    def undo(self, ws: Workspace) -> None: ...
    def merge_with(self, other: "Command") -> bool:
        """Trả True nếu hai command merge được, mutate self."""
        return False
```

### 26.2. Danh mục Command bắt buộc

| Command | label | reversible | merge | Notes |
|---------|-------|-----------|-------|-------|
| AddRoiCommand | "Add ROI" | yes | no | lưu RoiAnnotation snapshot |
| AddPolygonRoiCommand | "Add polygon ROI" | yes | no | |
| MoveAnnotationsCommand | "Move N item(s)" | yes | yes (drag liên tục) | merge_key = "move:<ids>" |
| ResizeAnnotationCommand | "Resize ROI" | yes | yes | |
| DeleteAnnotationsCommand | "Delete N item(s)" | yes | no | snapshot deleted items |
| EditLabelCommand | "Rename to <x>" | yes | yes | merge_key = "label:<id>" |
| BrushStrokeCommand | "Brush stroke" | yes | no | lưu dirty rect + before/after PNG |
| EraseStrokeCommand | "Erase stroke" | yes | no | |
| AddPointCommand | "Add SAM point" | yes | no | |
| DeletePointCommand | "Delete SAM point" | yes | no | |
| AddMeasurementCommand | "Add measurement" | yes | no | |
| ApplyDetectResultsCommand | "Apply detect run <run_id>" | yes | no | snapshot box list |
| ApplySegmentResultsCommand | "Apply segment run <run_id>" | yes | no | snapshot mask refs |
| AddLayerCommand | "Add layer <name>" | yes | no | |
| RemoveLayerCommand | "Remove layer <name>" | yes | no | snapshot layer + items |
| ReorderLayerCommand | "Reorder layers" | yes | yes | merge khi drag |
| SetLayerVisibilityCommand | "Hide/Show <layer>" | yes | yes | merge_key = "visible:<id>" |
| SetLayerOpacityCommand | "Opacity <layer> = N%" | yes | yes | merge_key = "opacity:<id>" |
| LockLayerCommand | "Lock <layer>" | yes | yes | |
| CropImageCommand | "Crop image" | yes (heavy) | no | snapshot original pixmap path |
| ImportFromRunCommand | "Import run <id>" | yes | no | reverse remove imported items |

### 26.3. Không reversible

- Run job (chạy DINO, SAM, UNet) — bản thân job không vào undo stack. Chỉ
  `ApplyDetectResultsCommand` / `ApplySegmentResultsCommand` ở step "apply
  result vào layer" mới vào stack. Undo = remove kết quả khỏi layer (job đã
  tốn compute, không hoàn lại được).
- Save Project / Export — side effect ngoài, không undo.
- Preferences thay đổi — lưu trực tiếp, không vào undo.

### 26.4. Merge rule

- Drag liên tục một item → spam `MoveAnnotationsCommand`. UndoStack merge khi:
  - cùng `merge_key`
  - command kế trong vòng `MERGE_WINDOW_MS = 500`
- Brush stroke: 1 mouse press→release = 1 command (không merge across stroke).
- Slider opacity: thả slider → 1 command đại diện toàn bộ drag (start value lưu
  ở pressed, end value lưu ở released).

### 26.5. Limit & memory

- `UndoStack.MAX_DEPTH = 100`. Khi vượt, drop oldest.
- BrushStrokeCommand giữ before/after PNG dirty rect để khỏi giữ full mask.
- ApplySegmentResultsCommand không inline mask PNG, chỉ tham chiếu file asset
  đã ghi sẵn.

### 26.6. Persistence

- UndoStack KHÔNG persist sang session sau (project file).
- Nhưng `history.detection_runs` / `history.segmentation_runs` thì PERSIST
  trong `.ddproj` (mục 12) — dùng để hiển thị "Imports" trong History panel
  và cho phép tái-apply qua `ImportFromRunCommand`.

---

## 27. Job manager — bổ sung

### 27.1. Model `Job`

```python
@dataclass
class JobSpec:
    id: str                    # "J_<uuid8>"
    kind: str                  # "detect" | "segment" | "export" | "import"
    title: str                 # "Detect (gdino) on img.jpg"
    request: InferenceRequest | None
    layer_id: str | None       # layer kết quả sẽ apply vào
    parent_id: str | None      # batch parent

@dataclass
class JobStatusInfo:
    status: str                # "queued" | "running" | "completed" | "failed" | "cancelled"
    progress: float = 0.0      # 0..1
    message: str = ""
    error: str | None = None
    result: object | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    log_lines: list[str] = field(default_factory=list)
```

Lưu ý: `JobSpec` immutable, `JobStatusInfo` mutable do JobManager update.

### 27.2. Queue policy

- Queue mặc định: **serial** trong cùng kind (detect chỉ 1 job/lúc, segment chỉ
  1 job/lúc) vì service subprocess chỉ handle 1 call/lúc.
- Cross-kind: detect và segment có thể song song (2 subprocess khác).
- Export job: chạy parallel không giới hạn (CPU-bound, không phụ thuộc service).
- Khi user submit detect mới trong khi detect cũ đang chạy → hỏi: "Cancel
  current and start new?" / "Queue" / "Cancel".

### 27.3. Background vs UI-blocking

- Mọi job ML là **background** (Jobs panel, không block UI).
- Job nhỏ tức thời (apply result vào layer, render preview): chạy trên UI
  thread, không vào JobManager.
- Long export job: background, có progress bar.

### 27.4. Đóng app khi đang chạy

Khi user Cmd+Q với job running:

```
Dialog: "1 job đang chạy. Bạn muốn..."
  [Wait for jobs]   [Cancel jobs and quit]   [Force quit]
```

- Wait: app ẩn nhưng giữ process; quit khi tất cả xong.
- Cancel: gọi `terminate(job_id)` cho từng job, đợi tối đa 5s rồi quit.
- Force: kill subprocess ngay, có thể leave file tạm.

### 27.5. Job → layer mapping

Khi job xong:

- Detect → submit `ApplyDetectResultsCommand` thêm vào layer Detections (hoặc
  layer mới `Detect <run_id>` nếu user bật setting "isolate runs in layers").
- Segment → submit `ApplySegmentResultsCommand` thêm mask vào layer Masks.
- Mỗi run được track trong `WorkspaceState.detection_runs` / `segmentation_runs`
  để có thể re-apply / compare.

### 27.6. Compare runs

`JobsPanel` có sub-tab "Runs" liệt kê runs đã có (kế thừa
[`compare_controller`](../tools/ui/editor_app/controllers/compare_controller.py)):

- Toggle visibility per run
- Compare A vs B (overlay diff layer)
- Delete run

---

## 28. Settings & preferences — chi tiết

### 28.1. Layout settings

```
~/.damagedetector/ui/
├── settings.json           # (28.2) per-user
├── shortcuts.json          # custom keyboard shortcuts
├── recent_files.json
├── window_state.json       # geometry + dock layout
└── logs/
    └── ui_<date>.log
```

Trong project: `<project>.ddproj` mục `settings:` (28.3) override per-project.

Resolution rule: project > user > default.

### 28.2. settings.json (per-user, namespaced)

```json
{
  "version": 1,
  "models": {
    "dino_checkpoint": "IDEA-Research/grounding-dino-base",
    "dino_config_id": "IDEA-Research/grounding-dino-base",
    "stabledino_checkpoint": "",
    "sam_checkpoint": "/abs/path/sam_vit_b_01ec64.pth",
    "sam_model_type": "auto",
    "sam_lora_checkpoint": "/abs/path/best_model.pth",
    "unet_checkpoint": "/abs/path/best_model.pth"
  },
  "thresholds": {
    "box_threshold": 0.25,
    "text_threshold": 0.25,
    "max_dets": 20,
    "dino_nms_iou_threshold": 0.5,
    "unet_threshold": 0.5,
    "min_box_px": 4,
    "min_mask_area": 0
  },
  "device": "auto",
  "ui": {
    "theme": "system",
    "show_grid": false,
    "show_pixel_grid_above_zoom": 8.0,
    "default_zoom": "fit",
    "icon_size": 24,
    "panel_density": "comfortable"
  },
  "performance": {
    "max_jobs_parallel": 2,
    "mask_pixmap_cache_mb": 256,
    "image_decode_pyramid": true
  },
  "shortcuts_overrides": {},
  "recent_files": [],
  "language": "vi"
}
```

### 28.3. Project-level settings (subset trong .ddproj)

Chỉ chứa những setting relevant cho project (model checkpoint, threshold), không
chứa UI/window state.

### 28.4. Class label map

`ui/resources/class_map.json`:

```json
{
  "crack":  {"display": "Crack",  "color": "#3796FF"},
  "mold":   {"display": "Mold",   "color": "#34D399"},
  "stain":  {"display": "Stain",  "color": "#F59E0B"},
  "spall":  {"display": "Spall",  "color": "#F87171"}
}
```

User có thể override trong Preferences → Classes.

### 28.5. Migration từ `.editor_app.json`

App khi khởi động:

1. Tìm `~/.damagedetector/ui/settings.json`. Nếu chưa có:
2. Tìm `<repo>/.editor_app.json` (legacy [`.editor_app.json`](../.editor_app.json)).
3. Nếu có, chạy `migrate_legacy_settings(legacy: dict) -> dict`:
   - `sam_checkpoint` → `models.sam_checkpoint`
   - `sam_model_type` → `models.sam_model_type`
   - `sam_lora_checkpoint` → `models.sam_lora_checkpoint`
   - `dino_checkpoint` → `models.dino_checkpoint`
   - `box_threshold`, `text_threshold`, `max_dets` → `thresholds.*`
   - `unet_model` → `models.unet_checkpoint`
   - `unet_threshold` → `thresholds.unet_threshold`
   - `device` → `device`
   - `last_workspace` → `recent_files[0]`
4. Ghi `~/.damagedetector/ui/settings.json` v1, giữ legacy file.

### 28.6. Preferences dialog

Tabs:

- **General**: language, theme, panel density, default zoom.
- **Models**: paths cho DINO / SAM / SAM-LoRA / UNet với picker + "Test load" button.
- **Detection**: defaults thresholds, max dets, prompt groups manager.
- **Segmentation**: defaults SAM/UNet, mask opacity, expand box.
- **Performance**: device, max parallel jobs, cache size, image decode pyramid toggle.
- **Shortcuts**: bảng (action ↔ shortcut), reset to default.
- **Classes**: class map editor (display name + color).

Mỗi tab có Apply / Reset / Cancel.

---

## 29. Performance & resource

### 29.1. Ảnh lớn (>4K)

- Sử dụng [`QImageReader`](https://doc.qt.io/qt-6/qimagereader.html) với
  `setAutoTransform(True)` để đọc EXIF orientation.
- Bật image decode pyramid: tạo 4 mức (1×, 1/2×, 1/4×, 1/8×) cache `QPixmap`.
  Render mức theo zoom hiện tại, tránh decode full mỗi paint.
- Đặt `setCacheMode(QGraphicsItem.DeviceCoordinateCache)` cho image item.
- Lazy decode: chỉ decode khi user chuyển ảnh; giữ tối đa 3 ảnh trong cache LRU.

### 29.2. Mask compositing

- `MaskCompositeItem` cache 1 `QPixmap` ARGB ở image size, blend tất cả mask
  con vào pixmap cache. Tự invalidate khi:
  - mask added/removed/changed
  - layer opacity / visibility đổi
  - blend mode đổi
- Brush tool dirty rect: chỉ re-blend trong rect, không full image.

### 29.3. Annotation count

- Soft cap: 5,000 annotation/layer. Vượt → cảnh báo, nhưng không block.
- `LayersPanel` paginate khi 1 layer > 1,000 item.

### 29.4. GPU / device

- Trước khi submit job, hiển thị device đã chọn ở Inspector. Nếu device = cuda
  nhưng `select_device_str("auto") == "cpu"` → toast "CUDA unavailable, falling
  back to CPU".
- Job manager nắm trạng thái "service warm" qua heartbeat events từ
  `InferenceApi`. Nếu service chết, restart và show toast.

### 29.5. Cancel an toàn

- `JobManager.cancel(id)` → `InferenceApi.cancel(id)` (graceful: workflow next
  iteration sẽ thoát).
- `JobManager.terminate(id)` → `InferenceApi.terminate(id)` (kill subprocess).
- UI bind: nút Cancel trong Jobs panel mặc định graceful; long-press 1s →
  terminate hard.
- Sau terminate, mark service dirty → next call sẽ ensure_started lại.

---

## 30. Error handling & UX feedback

### 30.1. Phân loại lỗi

| Loại | Ví dụ | UX |
|------|-------|----|
| Recoverable user error | Mở file image lỗi format | Toast warning + Log dock entry |
| Recoverable system error | Job fail vì OOM | Toast error + Job row red + Retry button |
| Recoverable config error | Model checkpoint không tồn tại | Inline validation trên Inspector field, disable Run |
| Critical | Service crash, project corrupted | Modal dialog với 3 action: Retry / Reset / Quit |
| Background | log spam từ service | Chỉ vào Log panel, không toast |

### 30.2. Channels

- **Toast** (status bar góc): 3-6s, dismissible. Dùng cho event ngắn (success
  save, job cancelled, service restarted).
- **Status bar message**: 1 dòng, không block, thay khi có message mới.
- **Log dock**: stream log từ subprocess + UI events (filter by level).
- **Modal dialog**: chỉ khi cần user quyết định (load broken project, quit
  during job, model checkpoint thiếu khi run).
- **Inline validation**: trên form field (red border + tooltip).

### 30.3. Use cases cụ thể

- **Mở ảnh lỗi**: toast "Cannot load image: <path>". Log full traceback.
- **Model checkpoint thiếu**: Inspector → Detect / Segment, nút Run disabled +
  inline "Checkpoint not found: <path>" với link "Choose…".
- **Job fail mid-run**: Job row đỏ, message từ workflow, nút Retry submit job
  cùng request.
- **Service crash**: toast "Service crashed, restarting…", Job marked failed,
  next submit sẽ warm lại.
- **Save project lỗi**: dialog "Cannot save project: <reason>. Try Save As?".
- **Project version cao hơn app**: dialog "Project saved by newer version,
  open read-only?".

### 30.4. Crash report

Uncaught exception → ghi `~/.damagedetector/ui/logs/crash_<ts>.log` với
traceback + workspace state snapshot, hiển thị dialog "Something went wrong"
với link Reveal log file.

---

## 31. Accessibility & input

### 31.1. Keyboard navigation

- Toolbar / menu / dock đều `setFocusPolicy(Qt.StrongFocus)` để Tab navigate.
- Inspector tabs Ctrl+Tab cycle.
- Layer panel: ↑/↓ select, Space toggle visibility, Cmd+L lock.
- Canvas: arrow keys nudge selected items 1px (Shift+arrow = 10px).

### 31.2. Focus

- Khi switch tool, focus về canvas để keyboard shortcut tool tiếp tục hoạt động.
- Khi mở dialog, return focus về widget trước đó khi đóng.

### 31.3. Multi-monitor / DPI

- Bật `Qt.AA_EnableHighDpiScaling`, `Qt.AA_UseHighDpiPixmaps` trong
  `application.py`.
- Icon SVG scale đúng. PNG icon có @2x.
- Lưu `window_state.json` per-monitor-config (hash của tổ hợp screen) để
  khi cắm monitor khác không mở app ngoài màn hình.

### 31.4. Locale

- Phase đầu chỉ Vietnamese + English. `ui/resources/translations/*.qm`.
- Fallback: nếu key thiếu trong tiếng Việt → English.

---

## 32. Test plan — mở rộng

### 32.1. Unit tests (`tests/ui/`)

- `test_command_pattern.py`: AddRoiCommand undo/redo, MoveAnnotationsCommand
  merge, BrushStrokeCommand dirty rect.
- `test_layer_tree.py`: add/remove/move layer, visibility cascade, opacity
  cascade, solo/unsolo.
- `test_annotation_store.py`: ID stability, snapshot/restore.
- `test_settings_migration.py`: migrate `.editor_app.json` → `settings.json` v1.
- `test_project_io.py`: save/load .ddproj, missing asset graceful, version
  migration.
- `test_workflow_mapping.py`: UI action → InferenceRequest đúng workflow + params.

### 32.2. Integration tests (Qt headless với `pytest-qt`)

- `test_canvas_tools.py`: simulate mouse events qua `QTest`, kiểm tra
  RectRoiTool tạo đúng RoiAnnotation.
- `test_undo_redo_flow.py`: vẽ 5 ROI, undo 3 lần, redo 2 lần, kiểm tra
  workspace state.
- `test_job_manager.py`: submit fake job, simulate progress events, kiểm tra
  Jobs panel update.
- `test_layers_panel.py`: toggle visibility, opacity slider thay đổi command
  stack đúng.

### 32.3. Smoke tests (manual checklist)

- Mở ảnh từ `data/HinhAnh/` → canvas hiện ảnh.
- Vẽ ROI → undo → redo → state ổn.
- Run detect (DINO) trên ảnh thật → boxes hiện trên layer Detections.
- Run segment SAM trên boxes → mask hiện overlay.
- Save project → quit → reopen → state khôi phục.
- Cancel job đang chạy → status = cancelled, không apply result.
- Drag-drop ảnh khác từ Finder → confirm dialog.
- Đổi theme system → app respect.
- Cắm/rút monitor → window vẫn hiện đúng monitor.

### 32.4. Regression tests

- Sau mỗi phase, chạy lại smoke checklist.
- Ghi expected screenshots (golden) cho 3 view: empty workspace, 1 ROI,
  detection result. So sánh visual diff.

### 32.5. Long-running / load tests

- Mở 1 ảnh 8K (24MP) → đo: load time < 2s, scroll/zoom < 16ms/frame.
- Tạo 500 ROI → vẫn 60fps.
- Chạy detect 10 ảnh batch → JobManager xử lý đúng thứ tự, không leak.
- Brush 1000 stroke trên mask 4K → undo full không crash.

### 32.6. Test với job thật

Smoke run trong môi trường có model:

- DINO detect 1 ảnh → có ≥ 1 box.
- SAM zero-shot từ box → mask non-empty.
- SAM-LoRA → mask non-empty.
- UNet → mask full-image.
- Isolate → output isolate đúng class.

---

## 33. Migration roadmap từ editor_app cũ

### 33.1. Co-existence

Trong giai đoạn parity:

- App cũ: console script `damage-editor` → `ui.editor_app:main` (giữ nguyên).
- App mới: console script `damage-editor-next` → `ui.app.application:run`.
- Hai app dùng chung `inference_api`, `object_detection`, `segmentation`,
  `tools/ui/editor_app/services/run_storage.py` (DB lịch sử).
- Settings: app mới migrate đọc `.editor_app.json`; app cũ vẫn ghi file đó →
  app mới có thể bị overwrite. Giải pháp: app mới đọc 1 lần lúc đầu, sau đó
  ghi vào `~/.damagedetector/ui/settings.json` riêng.

### 33.2. Code disposition

| Cũ | Hành động |
|----|----------|
| [`tools/ui/editor_app/canvas.py`](../tools/ui/editor_app/canvas.py) | Refactor → split |
| [`tools/ui/editor_app/controllers/*`](../tools/ui/editor_app/controllers/) | Refactor → `core/`, `services/` |
| [`tools/ui/editor_app/services/*`](../tools/ui/editor_app/services/) | Reuse phần lớn |
| [`tools/ui/editor_app/stores/*`](../tools/ui/editor_app/stores/) | Drop, gộp vào `core/workspace.py` |
| [`tools/ui/editor_app/ui/main_window.py`](../tools/ui/editor_app/ui/main_window.py) | Drop, viết mới ở `ui/app/main_window.py` |
| [`tools/ui/editor_app/domain/models.py`](../tools/ui/editor_app/domain/models.py) | Reuse + mở rộng → `ui/models/annotation.py` |
| [`tools/ui/editor_app/config/prediction_settings.py`](../tools/ui/editor_app/config/prediction_settings.py) | Reuse trực tiếp (PredictionConfig) |
| [`tools/ui/editor_app/color_utils.py`](../tools/ui/editor_app/color_utils.py) | Reuse |
| [`tools/ui/editor_app/image_io.py`](../tools/ui/editor_app/image_io.py) | Reuse |
| [`tools/ui/editor_app/paths.py`](../tools/ui/editor_app/paths.py) | Reuse |

### 33.3. Cutover

Sau Phase 11:

- Update [`setup.py`](../setup.py) console script `damage-editor` → `ui.app.application:run`.
- Thêm deprecation note trong [`tools/ui/editor_app/CLAUDE.md`](../tools/ui/editor_app/CLAUDE.md).
- Giữ `ui.editor_app:main` ít nhất 2 release cycle với flag `--legacy-editor`.

---

## 34. Phạm vi MVP

### 34.1. v1 (MVP)

**Mục tiêu**: parity tối thiểu với app cũ + layout GIMP-style.

- App shell: menu / toolbar / left tools / canvas / right inspector + layers /
  bottom jobs+log.
- Image load/display, pan/zoom/fit (đã có).
- Tools: Pan, Select, Rect ROI.
- Layers: Image, ROIs, Detections, Masks (tree đơn giản, no nesting custom).
- Inspector: Image, Detect, Segment tab (Object tab tối thiểu).
- Job manager với detect (DINO) + segment (SAM zero-shot) + UNet.
- Inference API integration thật (bỏ detect_process subprocess riêng).
- Undo/redo cho ROI add/delete/move.
- Settings: migration từ `.editor_app.json`, Preferences dialog (General +
  Models tabs).
- Export: overlay PNG, CSV, mask PNG.
- Console script `damage-editor-next`.

**Không có trong v1**: brush, polygon ROI, point prompt SAM, crop tool,
measurement, project file, layer reorder, multi-image batch, blend mode,
locale, accessibility nâng cao.

### 34.2. v2

- Project file `.ddproj` (Phase 7 spec).
- Brush + Eraser cho mask (Phase 9).
- SAM-LoRA + Detect+Segment combined workflows.
- Polygon ROI + measurement.
- Layer reorder + duplicate + merge_down.
- Recent files + window state restore.
- Preferences đầy đủ tabs (Detection, Segmentation, Performance, Shortcuts).

### 34.3. v3

- Point prompt SAM tool (live preview).
- Crop tool (destructive, undoable).
- Multi-image / Image Browser panel.
- Compare runs (side-by-side, diff layer).
- COCO JSON export.
- Locale Vietnamese đầy đủ.
- Accessibility nâng cao (focus order, multi-monitor restore).
- Custom QSS theme support (optional).

### 34.4. Out of scope

- Plugin system.
- Cloud sync.
- Multi-user collaborative editing.
- Video / 3D.
