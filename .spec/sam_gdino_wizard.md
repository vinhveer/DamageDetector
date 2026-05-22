# SAM + Grounding DINO Detector (Wizard)

Spec cho tính năng mới trong `app/`: wizard 5 bước chạy pipeline semi-labeling
(GroundingDINO → OpenCLIP relabel → Spatial filter → SAM) cho 1 hoặc nhiều ảnh,
với UI cho phép back-and-forth giữa các bước.

---

## 1. Mục tiêu

Người dùng đi từ ảnh thô → mask SAM cuối cùng qua 5 màn hình tuần tự:

| Màn | Mục đích | Backend chính |
|-----|----------|----------------|
| 1 | Chọn ảnh / thư mục ảnh | (file dialog) |
| 2 | Cấu hình GroundingDINO + Semantic + Spatial filter | semi-labeling step 1, 2, 3 |
| 3 | Chạy detection, xem box overlay; back lại Màn 2 để re-config | step1+step2+step3 pipeline |
| 4 | Chọn model SAM (`vit_b/l/h` hoặc SAM-LoRA) + tinh chỉnh tham số | (config form) |
| 5 | Chạy SAM trên các box đã filter, hiển thị mask overlay | `segmentation.sam` |

Yêu cầu UX:
- Có thể back ở bất kỳ màn nào để chỉnh và re-run.
- Kết quả Màn 3 (boxes) được giữ trong state cho đến khi user re-run.
- Kết quả Màn 5 (masks) hiển thị cùng overlay, có toggle bật/tắt.

---

## 2. Bố cục trong app

Thêm 1 tab mới ở [`app/src/App.jsx`](../app/src/App.jsx):

```
NAV_MAIN = [
  Workflows,
  Image Viewer,
  Prototype Review,
  Segment,
  SAM+GDino Wizard,   ← thêm
]
```

- Icon đề xuất: `IconWand` hoặc `IconRoute` (`@tabler/icons-react`).
- Component gốc: `features/samGdinoWizard/SamGdinoWizard.jsx`.
- Redux slice: `features/samGdinoWizard/samGdinoWizardSlice.js`, register
  trong [`app/src/app/store.js`](../app/src/app/store.js).

---

## 3. Cấu trúc thư mục frontend mới

```
app/src/features/samGdinoWizard/
├── SamGdinoWizard.jsx           # Container: switch theo state.step
├── samGdinoWizardSlice.js       # Redux state + thunks
├── components/
│   ├── WizardShell.jsx          # Header có stepper + nút Back / Next / Re-run
│   ├── Step1PickImages.jsx      # Màn 1
│   ├── Step2GdinoConfig.jsx     # Màn 2
│   ├── Step3DetectResults.jsx   # Màn 3 (overlay boxes + table)
│   ├── Step4SamConfig.jsx       # Màn 4
│   ├── Step5SamResults.jsx      # Màn 5 (mask overlay + per-image grid)
│   ├── BoxOverlay.jsx           # Canvas render boxes lên ảnh
│   └── MaskOverlay.jsx          # Canvas render masks lên ảnh
└── README.md                    # Ghi chú dev (option)
```

---

## 4. Redux state

```js
{
  step: 1,                       // 1..5
  // Step 1
  inputs: {
    paths: [],                   // mảng file ảnh tuyệt đối
    recursive: false,            // nếu user chọn folder
  },
  // Step 2 – GroundingDINO + semantic + spatial filter
  gdino: {
    boxThreshold: 0.16,
    textThreshold: 0.16,
    maxDets: 80,
    tiledThreshold: 512,
    tileScales: ['small', 'medium', 'large'],
    promptGroups: [              // mặc định = DEFAULT_PROMPT_GROUPS
      { id: 1, name: 'crack',  prompt: 'crack, surface crack, ...' },
      { id: 2, name: 'mold',   prompt: 'mold, mildew, moss, ...' },
      { id: 3, name: 'spall',  prompt: 'spalling, broken concrete, ...' },
    ],
    checkpointPath: '',          // default: model default
    device: 'auto',
  },
  semantic: {
    enabled: true,
    modelName: 'ViT-B-32',
    pretrained: 'laion2b_s34b_b79k',
    batchSize: 16,
  },
  spatial: {
    enabled: true,
    iouThreshold: 0.5,
    containmentThreshold: 0.8,
    // các threshold khác lấy từ filter_duplicates.py
  },
  // Step 3 – output
  detection: {
    status: 'idle',              // idle|running|done|error
    sessionId: null,             // session ID của workflow đang chạy
    logs: [],                    // stdout/stderr lines
    runId: null,                 // run_id trả về từ damage_scan
    semanticRunId: null,
    dbPath: null,                // path tới sqlite của step3 (filtered.sqlite3)
    boxesByImage: {              // { absImagePath: [box, ...] }
      // box: { x1, y1, x2, y2, label, score, detection_id, semantic_label, semantic_prob }
    },
    suspectByImage: {},          // boxes thuộc suspect.sqlite3
  },
  // Step 4 – SAM
  sam: {
    backend: 'sam',              // 'sam' | 'sam_finetune' (LoRA)
    modelType: 'vit_h',          // 'vit_b' | 'vit_l' | 'vit_h'
    checkpointPath: '',
    loraCheckpointPath: '',      // khi backend = 'sam_finetune'
    device: 'auto',
    multimask: false,
    minMaskArea: 0,
    expandBoxPx: 0,
  },
  // Step 5 – output
  segmentation: {
    status: 'idle',
    sessionId: null,
    logs: [],
    masksByImage: {              // { absImagePath: [{ box, mask_png_b64, score, area }] }
    },
    outputDir: null,
  },
  // UI
  showBoxOverlay: true,
  showMaskOverlay: true,
  selectedImagePath: null,       // ảnh đang xem ở step 3 & 5
  error: null,
}
```

---

## 5. Backend – workflow Python mới

Thêm 1 workflow module trong [`workflows/`](../workflows/) (Python entry mà
Electron đang spawn). Tách 2 workflow tách biệt để Step 3 và Step 5 chạy độc lập:

### 5.1. `workflows/sam_gdino_wizard_detect/`

Cấu trúc giống các workflow hiện có (xem
[`gdino_detect_damage/`](../app/workflows/gdino_detect_damage/) làm template):

- `sam_gdino_wizard_detect.json` – metadata cho app
- Python entrypoint chạy 3 step semi-labeling tuần tự trên DB tạm:
  - Tạo `runs_root = repo_root/.tmp/sam_gdino_wizard/<session_id>/`
  - Step 1: `object_detection.damage_scan.cli.main` ghi vào
    `runs_root/step1.sqlite3`
  - Nếu `semantic.enabled`: copy step1 → step2, sau đó gọi
    `semi-labeling/step2_sematic/pipeline.Step2SemanticPipeline`
  - Nếu `spatial.enabled`: gọi
    `semi-labeling/step3_spatial_filter/filter_duplicates.py`
    với `--output-dir runs_root/step3/`
  - In `RESULT_JSON:<json>` ở cuối stdout. JSON chứa:
    ```json
    {
      "db_path": ".../filtered.sqlite3",
      "suspect_db_path": ".../suspect.sqlite3",
      "run_id": "...",
      "semantic_run_id": "...",
      "boxes_by_image": {
        "/abs/path/img.jpg": [
          {"x1":..,"y1":..,"x2":..,"y2":..,
           "label":"crack","score":0.71,
           "semantic_label":"crack","semantic_prob":0.83,
           "detection_id": 42}
        ]
      }
    }
    ```

Renderer parse dòng `RESULT_JSON:` để load vào state.

### 5.2. `workflows/sam_gdino_wizard_segment/`

- Nhận `db_path` (filtered.sqlite3) + `sam_*` config từ Step 4.
- Đọc boxes từ DB, gọi `segmentation.sam.no_finetune.predict` hoặc
  `segmentation.sam.finetune.predict` per-image với box prompts.
- Lưu mask PNG vào `runs_root/masks/<image_stem>/<detection_id>.png`.
- In `RESULT_JSON:` chứa `masks_by_image` với base64 thumbnail (resized).

> **Lý do tách JSON marker:** workflows hiện tại stream stdout qua IPC như log.
> Dùng prefix `RESULT_JSON:` để renderer phân biệt log vs payload kết quả mà
> không phải đổi `workflow:start` protocol.

---

## 6. IPC mới

Reuse hoàn toàn `workflow:start` / `workflow:stop` / `workflow:event` hiện có.
Không cần thêm IPC handler mới ở `electron/main.js`.

Renderer chỉ cần handler bổ sung trong `samGdinoWizardSlice.js`:

```js
window.electronAPI.onWorkflowEvent(({ sessionId, type, data }) => {
  if (sessionId !== state.detection.sessionId &&
      sessionId !== state.segmentation.sessionId) return;
  if (type === 'stdout' && data.startsWith('RESULT_JSON:')) {
    dispatch(receiveResultJson(JSON.parse(data.slice('RESULT_JSON:'.length))));
  } else if (type === 'closed') {
    dispatch(sessionClosed({ sessionId, code: data }));
  } else {
    dispatch(appendLog({ sessionId, line: data }));
  }
});
```

Subscription được đăng ký 1 lần ở `SamGdinoWizard.jsx` mount, hủy ở unmount.

---

## 7. UX chi tiết từng màn

### Màn 1 – Pick images
- Nút **Add file(s)** → `browsePath('file')` cho phép multi (cần mở rộng IPC:
  `dialog:browse-path` truyền `properties: ['multiSelections']`).
  - Sửa `electron/main.js`:
    `if (mode === 'files') properties.push('openFile', 'multiSelections')`.
- Nút **Add folder** → `browsePath('directory')`, set `recursive` checkbox.
- Thumbnail grid (`<img src={"file://" + path}>`) — reuse styling từ
  `features/resultViewer/components/ImageGrid.jsx`.
- Footer: `Next →` (disabled nếu mảng paths rỗng).

### Màn 2 – GDino config
- Section **Prompt Groups**: list editable, mặc định 3 group (crack/mold/spall);
  nút `+ Add group`.
- Section **DINO thresholds**: box_threshold, text_threshold, max_dets,
  tile_scales (multi-select chip), tiled_threshold (numeric).
- Section **Checkpoint**: path picker (empty = repo default).
- Section **Device**: select `auto|cpu|cuda|mps`.
- Section **Semantic relabel (OpenCLIP)**: toggle on/off, model_name, pretrained,
  batch_size. Disabled fields khi toggle off.
- Section **Spatial filter**: toggle on/off, iou_threshold, containment_threshold.
- Footer: `← Back` | `Run detection →` (disabled khi đang chạy).

### Màn 3 – Detection results
- Layout 2 cột: trái = sidebar log (giống `WorkflowTerminal.jsx`), phải = canvas.
- Top bar: image selector (dropdown hoặc thumbnail strip horizontal scroll).
- Canvas hiển thị ảnh + `BoxOverlay` (boxes màu theo label, score nhỏ ở góc box).
- Toggle "Show suspect boxes" hiển thị `suspectByImage` (màu vàng nhạt).
- Table dưới canvas: list detections của ảnh đang chọn (sort, filter theo label).
- Footer:
  - `← Back to config`  → step = 2 (giữ nguyên detection.boxesByImage)
  - `Re-run detection`  → reset detection state + chạy lại
  - `Next: SAM config →` (disabled nếu `boxesByImage` rỗng)

### Màn 4 – SAM config
- Section **Backend**: radio `SAM (zero-shot)` vs `SAM-LoRA finetuned`.
- Section **Model type**: select `vit_b/vit_l/vit_h` (chỉ khi backend = SAM).
- Section **Checkpoint**: path picker; default suggest từ `models/`:
  - `models/sam_vit_h_4b8939.pth`
  - `models/sam_vit_b_01ec64.pth`
- Section **LoRA checkpoint**: chỉ hiện khi backend = SAM-LoRA.
- Section **Device**.
- Section **Box tuning**: `Multimask output (checkbox)`, `Expand box (px)`,
  `Min mask area (px²)`.
- Footer: `← Back to results` | `Run SAM →`.

### Màn 5 – SAM results
- Layout giống Màn 3 nhưng overlay là masks.
- Per-image grid: ảnh gốc, overlay mask blend, toggle alpha slider 0.0–1.0.
- Click vào 1 box trong table → highlight mask tương ứng.
- Nút **Export**: zip output dir và mở Reveal in Finder/Explorer.
- Footer: `← Back to SAM config` | `← Start over (về Màn 1)` | `Re-run SAM`.

---

## 8. Stepper component

`WizardShell.jsx` render breadcrumb với 5 chip, click vào chip đã hoàn thành
được phép quay lại (state vẫn còn). Logic:

```
canGoTo(step):
  step=1: always
  step=2: inputs.paths.length > 0
  step=3: detection.status !== 'idle' || boxesByImage không rỗng
  step=4: detection.status === 'done'
  step=5: segmentation.status !== 'idle' || masksByImage không rỗng
```

Khi user click back và chỉnh tham số ở Màn 2, **không** auto reset state.
Chỉ reset khi user bấm `Re-run detection` (clear `detection`) hoặc
`Run SAM` lần đầu (clear `segmentation`).

---

## 9. Persistence

Giống các slice khác, lưu config vào `localStorage`:
- `damage-detector.sam-gdino.gdino` (JSON of gdino state)
- `damage-detector.sam-gdino.sam` (JSON of sam state)

Không persist `detection.boxesByImage` / `segmentation.masksByImage` (transient).

---

## 10. Kế hoạch triển khai

Thứ tự PR / commit, mỗi bước có thể build & test độc lập:

1. **Scaffold tab + slice rỗng**
   - Thêm `SamGdinoWizard` vào `App.jsx` nav.
   - Tạo slice với `step`, action `goToStep`, container render placeholder.
   - Verify: `npm run dev` → tab xuất hiện, switch step bằng dev button.

2. **Step 1 UI**
   - Implement `Step1PickImages.jsx` + multi-select dialog (sửa
     `electron/main.js:dialog:browse-path` thêm mode `files`).
   - Verify: chọn nhiều ảnh, thumbnail grid render đúng.

3. **Backend workflow `sam_gdino_wizard_detect`**
   - Tạo `workflows/sam_gdino_wizard_detect/` (Python module + JSON).
   - Wire chạy step 1 only trước, in `RESULT_JSON:` với boxes rỗng giả.
   - Verify: spawn từ terminal, kiểm tra JSON ở stdout.

4. **Step 2+3 UI + IPC**
   - Form Màn 2; thunk `runDetection` → `electronAPI.startWorkflow`.
   - Subscriber parse `RESULT_JSON:` → populate `detection.boxesByImage`.
   - Render `Step3DetectResults` với `BoxOverlay`.
   - Verify end-to-end: chọn 1 ảnh ở [`HinhAnh/`](../../HinhAnh) → run → boxes.

5. **Full pipeline trong workflow**
   - Workflow thêm step 2 (semantic) + step 3 (spatial filter).
   - Output `boxes_by_image` filter từ `filtered.sqlite3`,
     `suspect_by_image` từ `suspect.sqlite3`.
   - Verify: boxes giảm sau spatial filter, suspect toggle hoạt động.

6. **Step 4 UI**
   - Form Màn 4.
   - Default checkpoint từ `models/sam_vit_h_4b8939.pth` (auto-detect path).

7. **Backend workflow `sam_gdino_wizard_segment`**
   - Python entrypoint nhận `--db` + sam params.
   - Loop qua boxes, gọi `segmentation.sam.no_finetune.predict.SamPredictor`.
   - Save mask PNG + base64 thumbnail → `RESULT_JSON`.

8. **Step 5 UI**
   - `MaskOverlay` canvas blend với alpha slider.
   - Table per-image với selection sync.
   - Nút Export + Start over.

9. **Polish**
   - Persist config to localStorage.
   - Error toast cho process exit code ≠ 0.
   - Disable Next khi running.
   - Update `CLAUDE.md` của `app/` mô tả tab mới.

---

## 11. Files cần đụng tới

**Tạo mới**
- `app/src/features/samGdinoWizard/**` (toàn bộ)
- `workflows/sam_gdino_wizard_detect/__init__.py`
- `workflows/sam_gdino_wizard_detect/sam_gdino_wizard_detect.json`
- `workflows/sam_gdino_wizard_detect/run.py`
- `workflows/sam_gdino_wizard_segment/__init__.py`
- `workflows/sam_gdino_wizard_segment/sam_gdino_wizard_segment.json`
- `workflows/sam_gdino_wizard_segment/run.py`

**Sửa**
- [`app/src/App.jsx`](../app/src/App.jsx) – thêm nav item + render
- [`app/src/app/store.js`](../app/src/app/store.js) – register reducer
- [`app/electron/main.js`](../app/electron/main.js) –
  `dialog:browse-path` hỗ trợ multi-file (mode `files`)
- [`app/electron/preload.cjs`](../app/electron/preload.cjs) –
  `browsePath` đã tổng quát, không cần đổi
- [`app/CLAUDE.md`](../app/CLAUDE.md) – mô tả tab mới
- [`DamageDetector/CLAUDE.md`](../CLAUDE.md) – append entry workflow mới

---

## 12. Rủi ro & câu hỏi mở

1. **Performance step 1 trên nhiều ảnh**: GroundingDINO scan chậm; cần stream
   progress (số ảnh đã xong) qua stdout dạng `PROGRESS:done=3/10` để Màn 3 hiển
   thị progress bar thay vì spinner.
2. **SQLite path mode**: step1 mặc định lưu `images.path` ở mode `name`. Khi
   nhiều ảnh trùng tên ở thư mục khác nhau sẽ xung đột. Wizard này nên ép
   `--store-image-path-mode absolute`.
3. **Multimask SAM** trả nhiều mask/box — UI chọn mask nào? Đề xuất: chọn mask
   có score cao nhất, để toggle "show all masks" cho power user.
4. **Suspect boxes**: step 3 spatial filter chia thành `filtered` và `suspect`.
   Wizard này tính suspect là "có thể bị duplicate" — mặc định ẨN, toggle để
   bật xem.
5. **LoRA backend**: cần verify `segmentation.sam.finetune` có API tương đương
   `predict_by_box(image, box)` không. Nếu chưa, fallback bỏ option `SAM-LoRA`
   ở phase đầu, để sau.
6. **Stop / cancel**: cần hook `electronAPI.stopWorkflow(sessionId)` vào nút
   Cancel ở Màn 3 & Màn 5 khi đang chạy.

---

## 13. Test plan tối thiểu

- 1 ảnh có nhiều crack: chạy full pipeline, verify boxes → masks reasonable.
- Folder 5 ảnh (mix crack/mold/spall): verify suspect filter giảm box trùng,
  SAM mask đúng class.
- Back từ Màn 3 → Màn 2, đổi `box_threshold`, re-run → state cập nhật.
- Back từ Màn 5 → Màn 4, đổi `vit_h` → `vit_b`, re-run → mask thay đổi.
- Đóng app khi đang chạy → child process bị kill (đã có sẵn ở
  `electron/main.js` qua `app.on('before-quit')`).
