# PLAN — Semi-labeling Reviewer: Electron → PySide6/Qt

Bản kế hoạch triển khai để một agent khác implement. Đọc kỹ phần "Bối cảnh" và
"Bài học bê nguyên" trước khi code — nhiều thứ đã giải xong, **không phát hiện lại**.

---

## 0. Bối cảnh

Pipeline bán nhãn hư hỏng (crack/mold/spall) sống trong `DamageDetector/semi-labeling/`.
Dữ liệu gom trong **`model_with_inference/semi_labeling/resemi.sqlite3`** (run hiện
tại `run_id = "myrun"`). App review hiện tại là **Electron/React** ở
`semi-labeling/app/` — ta sẽ **thay thế 100% bằng PySide6/Qt** rồi xoá Electron.

App mới: **`semi-labeling/semilabel_app/`**, mirror kiến trúc `tools/ui/editor_app/`
(đọc `tools/ui/editor_app/CLAUDE.md` để nắm pattern 5 lớp + threading + canvas).

Nguồn logic để PORT sang Python (đã validate, là source-of-truth):
- `semi-labeling/app/electron/labeling/queries.js` — tất cả truy vấn đọc + ghi.
- `semi-labeling/app/electron/labeling/db.js` — `connectRo` (WAL checkpoint rồi mở read-only).
- `semi-labeling/app/electron/labeling/sampling.js` — diverse sampling (FPS) cho review queue.
- `semi-labeling/app/electron/labeling/corrections.js` — `buildReviewDecisions`, `makeStampedId`.
- `semi-labeling/app/electron/labeling/pybridge.js` — chạy step qua subprocess (allow-list).
- `semi-labeling/app/src/features/labeling/exportLabel.js` — map nhãn → export label.
- `semi-labeling/app/src/features/labeling/Labeling.jsx`, `CleanedLabels.jsx`,
  `Prototype.jsx`, `Distribution.jsx`, `BoxImage.jsx` — tham chiếu UX/flow.

### Quyết định đã chốt (KHÓA — không đổi)
1. **App mới độc lập** `semi-labeling/semilabel_app/`, mirror `editor_app` (domain / stores=QObject+signals / services / controllers / ui). Entry point `damage-semilabel` (console_script trong pyproject/setup).
2. **Chạy step bằng QProcess (subprocess)** — `python -m steps.<step>.main` + flag form, stream stdout về log. DB read trên QThreadPool. Decode ảnh trên QThreadPool.
3. **Tab gọn:** Prototype · Review (gộp Labeling queue + Cleaned) · Phân bố/QA · Chạy bước · Export. **Bỏ Versions + Metrics.**
4. **Step 01–03 KHÔNG có trong app** (chạy trên Colab). Allow-list runner = `step04, step05, step06, step07, step08, step09, export_dataset`.
5. **Pipeline chain 1-nút từ Prototype:** step05 → 06 → 07 → 08 (nối tiếp, dừng nếu lỗi). **step09 KHÔNG trong chuỗi** (nút riêng, audit trước, `--apply-promotions` là hành động riêng có hộp xác nhận). **step04 chạy lẻ**.

---

## 1. Cấu trúc thư mục

```
semi-labeling/semilabel_app/
  __init__.py
  __main__.py              # cd semi-labeling && python -m semilabel_app
  app.py                   # QApplication, dựng MainWindow, wire stores+controllers
  paths.py                 # default resemi db, image root, settings path (reuse editor_app/paths.py style)
  config/defaults.py       # DEFAULT_SETTINGS: db_path, image_root, dinov2 model, ngưỡng reject...
  domain/
    models.py              # frozen dataclasses (KHÔNG import Qt): QueueItem, CleanedItem, Candidate, ClassDist, StepSpec, ChainStep
  services/
    db_service.py          # PORT read query từ queries.js (read-only sqlite)
    write_service.py       # commit_session / commit_corrections / update_cleaned_label (append-only)
    step_runner.py         # 1 QProcess: chạy 1 step, stream stdout, resolve python
    pipeline_runner.py     # chạy nối tiếp nhiều step (chain), dừng khi lỗi
    image_service.py       # resolve path + load crop/ảnh → QImage/QPixmap
    settings_service.py    # JSON persist
  stores/                  # QObject + Signal
    review_store.py        # queue/cleaned items, decisions đang chờ, index hiện tại
    prototype_store.py     # candidates (đã nhóm), picks
    run_store.py           # trạng thái chain/step + log
  controllers/
    review_controller.py
    prototype_controller.py
    run_controller.py
    export_controller.py
  ui/
    main_window.py         # sidebar nav + QStackedWidget (5 workspace)
    widgets/
      box_image.py         # ảnh + overlay bbox (port BoxImage.jsx; có thể tái dùng editor_app/canvas.py cho zoom/pan)
      thumb_grid.py        # lưới thumbnail lazy: QListView + QAbstractListModel + async decode + cache
      step_log.py          # chip trạng thái step + log viewer (QPlainTextEdit)
    workspaces/
      prototype_ws.py
      review_ws.py
      distribution_ws.py
      runsteps_ws.py
      export_ws.py
  tests/
    test_db_service.py     # test query không cần Qt (như editor_app test controllers)
```

**Nguyên tắc lớp:** domain ⊥ Qt; services không import widget; controllers không chứa widget code; UI nối signal của store + gọi controller. (Giống editor_app.)

---

## 2. Mô hình đa luồng

| Việc | Cơ chế | Lưu ý |
|---|---|---|
| DB read | `QThreadPool` + `QRunnable`; **mỗi worker mở connection sqlite read-only riêng** | sqlite không chia sẻ connection giữa thread. Kết quả trả về UI qua signal (dùng object emit, không đụng widget trong worker). |
| Decode crop/ảnh | `QThreadPool` worker: đọc PNG → `QImage` → signal; **lazy** chỉ ô đang nhìn + buffer; cache `dict[result_id]→QPixmap` (giới hạn kích thước, LRU) | đây là điểm chống lag cho lưới hàng nghìn crop |
| Chạy step | `QProcess` async; `readyReadStandardOutput`/`Error` → signal | resolve `python` (xem §6); env `PYTHONPATH=<semi-labeling dir>`, `cwd=<semi-labeling dir>`, `PYTHONUNBUFFERED=1`, `PYTHONIOENCODING=utf-8` |
| Ghi (commit) | QThread worker ngắn; `PRAGMA busy_timeout=60000` | append-only |

`QThreadPool.globalInstance().setMaxThreadCount(min(8, cpu-2))` cho decode.

---

## 3. domain/models.py (phác)

```python
@dataclass(frozen=True)
class QueueItem:
    result_id: int; image_rel_path: str; crop_path: str; image_uri: str; crop_uri: str
    initial_label: str; suggested_label: str; queue_type: str; reliability_score: float
    reasons: tuple[str, ...]; box: tuple[float,float,float,float] | None
    decided_action: str; decided_label: str
    pred_label: str | None; pred_prob: float; pred_margin: float; second_label: str
    disagrees_with_policy: bool; policy_label: str; defer_reasons: tuple[str, ...]

@dataclass(frozen=True)
class CleanedItem:
    result_id: int; image_rel_path: str; crop_path: str; final_label: str; export_label: str
    decision_type: str; reliability_score: float; box: tuple|None; crop_uri: str; image_uri: str

@dataclass(frozen=True)
class Candidate:
    result_id: int; label: str; predicted_label: str; reliability_score: float
    crop_uri: str; image_uri: str; box: tuple|None
    cluster_id: str; domain_index: int | None; cluster_size: int; centroid_similarity: float | None

@dataclass(frozen=True)
class ClassDist:
    total: int; by_label: list[tuple[str,int,float]]; by_decision_type: list[tuple[str,int,float]]

@dataclass(frozen=True)
class ChainStep:
    key: str; module: str; flags: dict   # module ∈ allow-list
```

---

## 4. services/db_service.py — PORT từ queries.js

Port nguyên các hàm (1:1, giữ SQL + logic), bỏ các hàm cho tab đã cắt:

**Giữ:** `list_runs`, `list_queue` (kèm diverse sampling, xem §5), `list_cleaned` (có `limit`),
`cleaned_distribution`, `list_prototype_candidates`, `latest_prototype`, `get_run_resources`,
`get_selftrain_promotions` (cho tóm tắt sau chain, tuỳ chọn).
**Bỏ:** `get_run_metrics`, `list_sessions`, `list_selftrain_runs`, `get_session_decisions` (thuộc Versions/Metrics đã cắt).

⚠️ **Bài học PHẢI giữ khi port** (đã validate trong queries.js hiện tại):
- `list_prototype_candidates`: **KHÔNG** LEFT JOIN `core_cluster_members`/`core_clusters` trong query chính (chưa index → ~18 phút). Nạp membership riêng vào dict rồi gắn trong Python. Lấy mẫu **stratified theo dải điểm** (`PARTITION BY eff_label, band`, cap `perBand`), vì `reliability_score` bão hoà ở 1.0. Nhãn hiệu dụng: `reliability < rejectBelow → 'reject'`.
- `connectRo`: nếu có file `*-shm` thì mở RW chạy `PRAGMA wal_checkpoint(TRUNCATE)` rồi đóng, sau đó mở read-only (để thấy row mới nhất).
- `list_queue` sắp xếp `reliability_score ASC` (box mơ hồ lên trước) và join `classifier_prediction_summary` (pred + cờ `disagrees_with_policy`) + `self_training_promotions` (defer reasons).
- `list_cleaned`: thêm tham số `limit` (0 = all) cho lưới.

DB read worker (QRunnable) bọc mỗi hàm; controller gọi async, store nhận signal.

---

## 5. Diverse sampling (port sampling.js)

`select_diverse_sample(rows, ratio)`: stratified theo lớp, mỗi lớp chạy **farthest-point
sampling** trên embedding DINOv2 tight (cosine distance = 1 - dot, vector đã L2-norm).
Dùng cho Review queue khi user chọn % mẫu < 100. Cài bằng numpy. Đọc blob embedding từ
`crop_embeddings` (chunk IN nếu cần — nhưng ở đây chỉ lấy theo embedding_run_id+view nên SELECT thẳng).

---

## 6. services/step_runner.py + pipeline_runner.py (port pybridge.js)

**Allow-list module:**
```python
STEP_MODULES = {
  "step04": "steps.step04_core.main",
  "step05": "steps.step05_proto.main",
  "step06": "steps.step06_reliability.main",
  "step07": "steps.step07_decision.main",
  "step08": "steps.step08_classifier.main",
  "step09": "steps.step09_self_train.main",
  "export_dataset": "tools.export_dataset",
}
```
**resolve_python():** thử `$SEMI_LABELING_PYTHON`, `<repo>/.venv/Scripts/python.exe`,
`<repo>/.venv/bin/python`, cuối cùng `"python"` (⚠️ KHÔNG `python3` — trên Windows máy này
`python3` trỏ Microsoft Store alias, exit 9009).
`semi_labeling_dir = .../DamageDetector/semi-labeling`; `cwd` và `PYTHONPATH` = thư mục này.

**StepRunner (1 step):** QProcess, `flags_to_argv` (validate `^--[a-z0-9-]+$`, bool→cờ trần,
bỏ None/False), signals `output(str)`, `finished(code)`. Có `stop()` (kill).

**PipelineRunner (chain):** nhận `list[ChainStep]`, chạy QProcess lần lượt; `finished(code)`:
code==0 → step kế; khác 0 → dừng + `chain_finished(False)`. Signals:
`step_started(i,label)`, `output(i,str)`, `step_finished(i,code)`, `chain_finished(ok)`.

---

## 7. Workspaces

### 7.1 Prototype (`prototype_ws.py`) — TRỌNG TÂM
- Setup: db path, image root, **DINOv2 model = `facebook/dinov2-giant`** (mặc định, phải khớp embedding), `rejectBelow` (0.5), `perBand` (200).
- Gallery: `list_prototype_candidates` → 4 tab lớp (crack/mold/spall/reject), **nhóm theo dải điểm** (≥90/80/70/60/50%) mặc định, toggle sang **Domain** (core cluster step04). Thumbnail lazy. Click chọn prototype; box trong tab reject = reject pick.
- **Nút 1 — "Tạo bank & chạy 06→08":** build flags step05 (`--db --run-id --model-name facebook/dinov2-giant --view-name tight --prototype <id:label,...> --reject <id:reject,...>`), rồi đẩy `PipelineRunner` chuỗi:
  `step05(picks) → step06 → step07 → step08(--model-name facebook/dinov2-giant)`.
  (06/07/08 selector mặc định `latest` nên tự bắt output bước trước — không cần truyền id.)
  Hiển thị chip + log (StepLog). `chain_finished(True)` → refresh + nhảy sang Review/Phân bố + tóm tắt cleaned/review/reject.
- **Nút 2 — "Chỉ tạo bank (step05)".**
- Mẫu lệnh step05 tham chiếu: xem `semi-labeling/README.md` + lịch sử (prototype/reject là chuỗi `result_id:label` ngăn cách dấu phẩy).

### 7.2 Review (`review_ws.py`) — gộp Labeling + Cleaned
- Toggle **Queue** (review_queue, `list_queue`, có % mẫu diverse) / **Cleaned** (`list_cleaned`).
- 2 chế độ xem: **từng box** (box_image + panel: reliability, gợi ý step07, "máy đoán" + cờ disagree + defer reasons) hoặc **lưới**.
- Phím tắt: `1–5` gán nhãn (crack/mold/spall/stain/reject) · `Enter` nhận gợi ý · `Space` tiếp · `Backspace` lùi.
- Queue mode commit → `commit_session` (review_sessions + review_decisions). Cleaned mode sửa → `update_cleaned_label` (in-place) + gom `commit_corrections`. Append-only; mỗi commit = 1 session có timestamp; nạp lại quyết định cũ từ `decided_action`.

### 7.3 Phân bố / QA (`distribution_ws.py`)
- `cleaned_distribution` → bar %/lớp + breakdown decision_type. Click lớp → `list_cleaned(finalLabel, limit=500)` → **lưới thumbnail**.
- Multi-select ô sai → **đổi nhãn hàng loạt** (`update_cleaned_label` từng cái) → gom pending → `commit_corrections`.

### 7.4 Chạy bước (`runsteps_ws.py`)
- Chạy lẻ step04–09 + export: chọn step → form flag (mặc định hợp lý: model dinov2-giant, view tight) → log real-time.
- **step09:** nút "Audit" (không `--apply-promotions`) + nút "Apply promotions" có **QMessageBox xác nhận** (vì ghi đè cleaned_labels/semantic_decisions). `--classifier-run-id` lấy latest hoặc cho chọn.
- Có cả nút chạy chuỗi 05→08 (như Prototype) cho tiện.

### 7.5 Export (`export_ws.py`)
- Chạy `tools.export_dataset` qua QProcess (`--db --run-id --image-root --output-dir --format`), parse dòng JSON cuối → hiện kết quả.

---

## 8. Lộ trình (mỗi phase chạy `cd semi-labeling && python -m semilabel_app` để nghiệm thu)

| Phase | Nội dung | Nghiệm thu |
|---|---|---|
| 0 | Scaffold: app shell, sidebar, 5 workspace rỗng, settings, paths, entry point | App mở, chuyển tab |
| 1 | `db_service` port hết read query + DbWorker + `tests/test_db_service.py` | Test pass trên resemi.sqlite3 (run myrun) |
| 2 | `image_service` + `thumb_grid` (decode async, cache, lazy) | Lưới 500 crop cuộn mượt |
| 3 | `step_runner` + `pipeline_runner` + `step_log` widget | Chạy step06 lẻ thấy log; chuỗi 06→07 dừng đúng khi lỗi |
| 4 | Review workspace (queue+cleaned, box+lưới, phím tắt, commit) | Gán nhãn → commit → session mới trong DB |
| 5 | Phân bố/QA (bar % + lưới + bulk relabel + commit) | Sửa hàng loạt → commit |
| 6 | Prototype workspace + **nút chuỗi 05→08** | Pick → 1 nút → cleaned_labels cập nhật |
| 7 | Chạy bước (lẻ + step09 gated) + Export | step09 apply có xác nhận; export ra dataset |
| 8 | Xoá Electron (`semi-labeling/app`), cập nhật `semi-labeling/README.md`, thêm console_script | `damage-semilabel` chạy; Electron đã gỡ |

---

## 9. Dữ liệu & đường dẫn (máy hiện tại)
- DB: `model_with_inference/semi_labeling/resemi.sqlite3`, `run_id="myrun"`.
- Crop tight: `model_with_inference/semi_labeling/crops/myrun/tight/<result_id>_tight_<hash>.png`.
- Image root: `data/HinhAnh` (1299 ảnh, khớp 100%).
- Embedding: `facebook/dinov2-giant`, view `tight`, dim 1536.
- Bảng dùng: `resemi_runs, review_queue, cleaned_labels, review_sessions, review_decisions,
  semantic_decisions, crop_views, crop_embeddings, core_clusters, core_cluster_members,
  prototype_versions, prototype_items, prototype_scores, classifier_runs,
  classifier_prediction_summary, self_training_runs, self_training_promotions, embedding_runs`.

## 10. Bài học bê nguyên (đừng phát hiện lại)
1. `python` chứ không `python3` (MS Store alias → exit 9009).
2. Prototype query: tách core membership ra Python (un-indexed JOIN = 18 phút), stratified per-band.
3. `reliability_score` bão hoà 1.0 → bin theo dải mới có nghĩa; reject = điểm thấp.
4. step06/07/08 selector mặc định `latest` → chain tự khớp; step08 cần `--model-name facebook/dinov2-giant`.
5. step05 reuse embedding sẵn (prototype là result_id đã embed) → không tải model.
6. Ghi: review_sessions/review_decisions append-only; updateCleanedLabel in-place; mỗi commit = 1 session timestamp.
7. `shared/db/embedding_cache.py::load_embeddings` đã fix chunk IN 900 (chạy qua subprocess nên dùng nguyên).

## 11. Packaging
- Thêm `semi-labeling/semilabel_app/__main__.py` + console_script `damage-semilabel = tools.semilabel_app_launcher:main` vào pyproject/setup.
- Phụ thuộc: PySide6 (đã có cho editor_app), numpy, Pillow. Không thêm Node.
