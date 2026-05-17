# CLAUDE.md — inference_api

This file provides guidance to Claude Code when working in `inference_api/`.

## Vai trò của module

`inference_api` là lớp **orchestration trung tâm** của toàn hệ thống. Nó không chứa ML code — thay vào đó nó:
- Quản lý vòng đời của inference jobs (submit → run → done/cancelled/failed)
- Định tuyến mỗi request vào đúng workflow (sam_dino, unet_only, v.v.)
- Giao tiếp với các model services qua IPC subprocess
- Phát events bất đồng bộ để UI có thể poll

---

## Cấu trúc file

| File | Trách nhiệm |
|------|-------------|
| `contracts.py` | Tất cả frozen dataclasses: `InferenceRequest`, `InferenceResult`, `JobEvent`, `JobSnapshot`, `DetectionResult` |
| `prediction_models.py` | Constants (TASK_GROUP_*, SEGMENTATION_*, DETECTION_*) và `PredictionConfig`, `ResolvedWorkflow` |
| `workflow_resolver.py` | Map `PredictionConfig` → tên workflow string |
| `request_builder.py` | Build `InferenceRequest` từ `PredictionConfig` + editor settings dict |
| `editor_bridge.py` | Chuyển legacy mode string ("sam_dino", "unet") → `InferenceRequest` |
| `api.py` | `InferenceApi` singleton — submit, cancel, terminate, poll events |
| `workflows.py` | Tất cả workflow implementations; dispatcher `run_workflow()` |
| `ipc.py` | JSON-newline message format cho IPC |
| `process_client.py` | `JsonServiceProcess` — spawn subprocess, gửi RPC calls, stream logs |
| `process_worker.py` | `WorkerProtocol` + `run_worker()` — phía worker subprocess |
| `cli_support.py` | Helpers cho CLI tools (parse ROI, parse queries, load boxes JSON) |

---

## Data flow từ đầu đến cuối

```
[UI: editor_bridge.py / request_builder.py]
    build_editor_request(mode, settings, ...) → InferenceRequest
             │
             ▼
[api.py]  InferenceApi.submit(request)
  • tạo job_id (UUID hex)
  • lưu JobSnapshot(status="queued")
  • spawn daemon thread _run_job()
  • return job_id ngay lập tức
             │
             ▼ (background thread)
[api.py]  _run_job(job_id, request)
  • emit JobEvent(type="started")
  • tạo WorkflowContext(job_id, request, emit_event, stop_checker, register_service)
             │
             ▼
[workflows.py]  run_workflow(ctx)
  • lookup handler trong _WORKFLOW_HANDLERS[workflow_name]
  • handler gọi ctx.call_service("dino", get_dino_service, "predict", params)
             │
             ▼
[process_client.py]  JsonServiceProcess.call(method, params, log_fn, stop_checker)
  • ensure_started() → spawn subprocess "python -m object_detection.dino"
  • gửi {"type":"call","id":N,"method":"predict","params":{...}} qua stdin
  • đọc queue: log messages → gọi log_fn(); "result" → return value
             │
             ▼
[workflows.py]  ctx.partial(payload)  →  emit JobEvent(type="partial_result")
              ctx.completed(payload)  →  return InferenceResult
             │
             ▼
[api.py]  set status="done", emit JobEvent(type="completed"), finalize
```

---

## Contracts quan trọng

```python
# Request: frozen, immutable
InferenceRequest(
    workflow="sam_dino",           # tên workflow
    image_path="/path/img.jpg",    # single image
    image_paths=[...],             # batch (hoặc None)
    roi_box=(x1, y1, x2, y2),     # optional crop
    params={                       # service-specific params
        "dino": {...},
        "sam": {...},
    }
)

# Event types theo thứ tự: queued → started → progress* → partial_result* → completed|cancelled|failed
JobEvent(type="progress", job_id=..., workflow=..., message="Warming up DINO...")
JobEvent(type="partial_result", ..., result=InferenceResult(...))  # DINO results trước SAM
JobEvent(type="completed", ..., result=InferenceResult(...))
```

---

## Workflow registry

`_WORKFLOW_HANDLERS` trong `workflows.py` là dict tĩnh. Để thêm workflow mới:
1. Viết handler function `_run_my_workflow(ctx: WorkflowContext) -> InferenceResult`
2. Đăng ký: `"my_workflow": _run_my_workflow`
3. Thêm vào `_CRACK_ONLY_WORKFLOWS` hoặc `_MORE_DAMAGE_WORKFLOWS` trong `workflow_resolver.py`

Mapping hiện tại:

| task_group | segmentation | detection | workflow |
|------------|-------------|-----------|---------|
| crack_only | sam | dino | sam_dino |
| crack_only | sam | none | sam_only |
| crack_only | sam_lora | dino | sam_dino_ft |
| crack_only | sam_lora | none | sam_only_ft |
| crack_only | unet | dino | unet_dino |
| crack_only | unet | none | unet_only |
| more_damage | sam | dino | sam_dino |
| more_damage | sam | none | sam_only |

---

## IPC Process Protocol

Giao tiếp giữa main process và model subprocess qua **JSON newline** trên stdin/stdout:

```
Parent → Subprocess:  {"type":"call","id":1,"method":"predict","params":{...}}\n
Subprocess → Parent:  {"type":"log","id":1,"text":"Loading model..."}\n    ← streaming logs
                      {"type":"result","id":1,"result":{...}}\n             ← final result
                      {"type":"error","id":1,"error":{"type":"...","message":"..."}}\n
Parent → Subprocess:  {"type":"stop","id":1}\n                              ← cancel request
Subprocess → Parent:  {"type":"stopped","id":1}\n
```

**Chú ý quan trọng:**
- Subprocess chỉ xử lý **1 job tại một thời điểm** (`spawn_job` trong `WorkerProtocol` trả về "Busy" nếu đang bận)
- `JsonServiceProcess` tự động restart subprocess khi process chết (`ServiceCrashed`)
- Warmup calls có timeout mặc định 180s; các method khác không có timeout mặc định

---

## WorkflowContext API

```python
ctx.log("Loading DINO model...")         # emit progress event
ctx.partial({"detections": [...]})       # emit partial_result (hiện thi DINO boxes trước SAM)
ctx.completed({"detections": [...]})     # return InferenceResult
ctx.call_service("dino", get_dino_service, "predict", params)
ctx.stop_checker()                       # → True nếu user cancel
```

---

## Cancellation flow

- `InferenceApi.cancel(job_id)`: đánh dấu job để cancel (graceful)
- `InferenceApi.terminate(job_id)`: cancel + gọi `service.close()` để kill subprocess ngay
- Workflows kiểm tra `ctx.stop_checker()` sau mỗi operation lớn; nếu `True` → return `{"stopped": True}`
- Kết quả `{"stopped": True}` được propagate từ service → workflow → job status = "cancelled"

---

## Lưu ý khi sửa code

- Tất cả dataclasses đều `frozen=True` — không mutate, tạo instance mới.
- `InferenceApi` là singleton (`get_inference_api()`), không khởi tạo trực tiếp.
- `_prune_finished_jobs_locked()` giữ tối đa 200 finished jobs trong memory; jobs có unread events không bị xóa.
- Thứ tự `_finished_order` dùng để biết job nào đã xong; prune theo FIFO.
- `_service_getters` dict trong `InferenceApi` dùng cho `terminate()` — nếu thêm service mới cần đăng ký ở đây.
