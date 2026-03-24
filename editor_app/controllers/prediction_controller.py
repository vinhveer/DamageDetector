from __future__ import annotations

import datetime as _dt
from pathlib import Path

from PySide6 import QtCore, QtGui

from editor_app.domain.models import JobRecord
from editor_app.image_io import load_mask
from editor_app.services.run_storage import RunStorageService
from editor_app.stores.history_store import HistoryStore
from editor_app.stores.prediction_store import PredictionStore
from editor_app.stores.workspace_store import WorkspaceStore
from inference_api import get_inference_api
from inference_api.editor_bridge import build_editor_request


class PredictionController(QtCore.QObject):
    errorRaised = QtCore.Signal(str)

    def __init__(
        self,
        workspace_store: WorkspaceStore,
        prediction_store: PredictionStore,
        history_store: HistoryStore,
        run_storage: RunStorageService,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._workspace_store = workspace_store
        self._prediction_store = prediction_store
        self._history_store = history_store
        self._run_storage = run_storage
        self._api = get_inference_api()
        self._runs_by_job_id: dict[str, object] = {}
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(180)
        self._poll_timer.timeout.connect(self.poll_events)

    def submit(
        self,
        *,
        mode: str,
        scope: str,
        settings: dict,
        roi_box: tuple[int, int, int, int] | None = None,
        target_labels: list[str] | None = None,
        outside_value: int | None = None,
        crop_to_bbox: bool | None = None,
        max_depth: int | None = None,
        min_box_px: int | None = None,
    ) -> str | None:
        results_root = self._workspace_store.results_root
        if results_root is None:
            self.errorRaised.emit("Open a workspace folder before running prediction.")
            return None

        current_image = self._workspace_store.current_image_path
        image_paths = list(self._workspace_store.images) if str(scope) == "folder" else None
        if str(scope) == "current" and not current_image:
            self.errorRaised.emit("Open an image before running prediction.")
            return None

        run = self._run_storage.create_run(results_root=Path(results_root), workflow=str(mode), scope=str(scope))
        try:
            request = build_editor_request(
                str(mode),
                settings,
                image_path=current_image,
                image_paths=image_paths,
                roi_box=roi_box,
                output_dir=str(run.output_dir),
                target_labels=target_labels,
                outside_value=outside_value,
                crop_to_bbox=crop_to_bbox,
                max_depth=max_depth,
                min_box_px=min_box_px,
            )
        except Exception as exc:
            self.errorRaised.emit(str(exc))
            return None

        request_data = {
            "workflow": request.workflow,
            "image_path": request.image_path,
            "image_paths": request.image_paths,
            "roi_box": request.roi_box,
            "params": request.params,
            "client_tag": request.client_tag,
            "source": request.source,
        }
        self._run_storage.write_request(run, request_data)
        job_id = self._api.submit(request)
        self._runs_by_job_id[job_id] = run
        self._prediction_store.add(
            JobRecord(
                job_id=job_id,
                run_id=run.run_id,
                workflow=str(mode),
                scope=str(scope),
                status="queued",
                image_path=current_image,
                image_paths=list(image_paths or []),
                run_dir=str(run.run_dir),
                output_dir=str(run.output_dir),
                created_at=_dt.datetime.now().isoformat(timespec="seconds"),
                request_data=request_data,
            )
        )
        self._prediction_store.append_log(job_id, f"Queued run {run.run_id}")
        if not self._poll_timer.isActive():
            self._poll_timer.start()
        return job_id

    def cancel(self, job_id: str) -> None:
        self._api.cancel(job_id)
        self._prediction_store.append_log(job_id, "Cancellation requested.")

    def poll_events(self) -> None:
        active_job_ids = [job.job_id for job in self._prediction_store.active_jobs()]
        if not active_job_ids:
            self._poll_timer.stop()
            return
        for job_id in active_job_ids:
            for event in self._api.drain_events(job_id):
                self._handle_event(job_id, event)
        if not self._prediction_store.active_jobs():
            self._poll_timer.stop()

    def _handle_event(self, job_id: str, event) -> None:
        run = self._runs_by_job_id.get(job_id)
        event_data = {
            "type": getattr(event, "type", ""),
            "job_id": getattr(event, "job_id", job_id),
            "workflow": getattr(event, "workflow", ""),
            "message": getattr(event, "message", None),
            "error": getattr(event, "error", None),
            "result": getattr(event.result, "to_dict", lambda: None)() if getattr(event, "result", None) is not None else None,
        }
        self._prediction_store.append_event(job_id, event_data)
        if run is not None:
            self._run_storage.append_event(run, event_data)

        message = str(getattr(event, "message", "") or "").strip()
        if message:
            self._prediction_store.append_log(job_id, message)

        event_type = str(getattr(event, "type", "") or "")
        if event_type == "started":
            self._prediction_store.update_status(job_id, "running")
            return
        if event_type == "progress":
            self._prediction_store.update_status(job_id, "running")
            return
        if event_type == "partial_result":
            payload = getattr(event.result, "to_dict", lambda: {})() if getattr(event, "result", None) is not None else {}
            self._prediction_store.append_partial(job_id, payload)
            self._apply_payload_if_current(payload, partial=True)
            return
        if event_type == "completed":
            payload = getattr(event.result, "to_dict", lambda: {})() if getattr(event, "result", None) is not None else {}
            self._prediction_store.update_status(job_id, "done")
            self._prediction_store.set_final_payload(job_id, payload)
            if run is not None:
                self._run_storage.write_result(run, status="done", payload=payload)
            self._apply_payload_if_current(payload, partial=False)
            return
        if event_type == "failed":
            error = str(getattr(event, "error", "") or getattr(event, "message", "") or "Prediction failed.")
            self._prediction_store.update_status(job_id, "failed", error=error)
            if run is not None:
                self._run_storage.write_result(run, status="failed", payload={}, error=error)
            return
        if event_type == "cancelled":
            payload = getattr(event.result, "to_dict", lambda: {})() if getattr(event, "result", None) is not None else {}
            self._prediction_store.update_status(job_id, "cancelled")
            if run is not None:
                self._run_storage.write_result(run, status="cancelled", payload=payload, error=getattr(event, "message", None))

    def _apply_payload_if_current(self, payload: dict, *, partial: bool) -> None:
        image_path = str(payload.get("image_path") or "")
        if not image_path or image_path != str(self._workspace_store.current_image_path or ""):
            return
        detections = list(payload.get("detections") or [])
        if detections:
            self._workspace_store.set_highlight_detections(detections)
            if not partial:
                self._workspace_store.set_detections(detections)
        if partial:
            return
        mask_path = str(payload.get("mask_path") or "")
        image = self._workspace_store.current_image
        if mask_path and not image.isNull():
            try:
                loaded = load_mask(mask_path, (image.width(), image.height()))
            except Exception:
                loaded = None
            if loaded is not None:
                self._workspace_store.set_current_mask(mask_path, loaded.mask)
