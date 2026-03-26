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
from inference_api.prediction_models import PredictionConfig
from inference_api.request_builder import build_isolate_request, build_prediction_request
from inference_api.workflow_resolver import resolve_workflow


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
        config: PredictionConfig,
        settings: dict,
        roi_box: tuple[int, int, int, int] | None = None,
    ) -> str | None:
        normalized = config.normalized()
        results_root = self._workspace_store.results_root
        if results_root is None:
            self.errorRaised.emit("Open a workspace folder before running prediction.")
            return None

        current_image = self._workspace_store.current_image_path
        image_paths = list(self._workspace_store.images) if normalized.scope == "folder" else None
        if normalized.scope == "current" and not current_image:
            self.errorRaised.emit("Open an image before running prediction.")
            return None

        try:
            resolved = resolve_workflow(normalized)
            run = self._run_storage.plan_run(results_root=Path(results_root), workflow=resolved.workflow, scope=str(normalized.scope))
            request = build_prediction_request(
                normalized,
                settings,
                image_path=current_image,
                image_paths=image_paths,
                roi_box=roi_box,
                output_dir=str(run.output_dir),
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
            "selection": request.selection,
            "resolved": request.resolved,
            "client_tag": request.client_tag,
            "source": request.source,
        }
        run_meta = {
            "task_group": str(request.selection.get("task_group") or ""),
            "segmentation_model": str(request.selection.get("segmentation_model") or ""),
            "detection_model": str(request.selection.get("detection_model") or ""),
            "resolved_workflow": str(request.resolved.get("workflow") or request.workflow),
        }
        self._run_storage.materialize_run(run, metadata=run_meta)
        self._run_storage.write_request(run, request_data)
        job_id = self._api.submit(request)
        self._runs_by_job_id[job_id] = run
        self._prediction_store.add(
            JobRecord(
                job_id=job_id,
                run_id=run.run_id,
                workflow=request.workflow,
                task_group=str(request.selection.get("task_group") or ""),
                segmentation_model=str(request.selection.get("segmentation_model") or ""),
                detection_model=str(request.selection.get("detection_model") or ""),
                resolved_workflow=str(request.resolved.get("workflow") or request.workflow),
                scope=str(normalized.scope),
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

    def submit_isolate(
        self,
        *,
        settings: dict,
        target_labels: list[str] | None,
        outside_value: int | None,
        crop_to_bbox: bool | None,
        roi_box: tuple[int, int, int, int] | None = None,
    ) -> str | None:
        results_root = self._workspace_store.results_root
        current_image = self._workspace_store.current_image_path
        if results_root is None:
            self.errorRaised.emit("Open a workspace folder before running isolate.")
            return None
        if not current_image:
            self.errorRaised.emit("Open an image before running isolate.")
            return None
        run = self._run_storage.plan_run(results_root=Path(results_root), workflow="isolate", scope="current")
        try:
            request = build_isolate_request(
                settings,
                image_path=current_image,
                output_dir=str(run.output_dir),
                roi_box=roi_box,
                target_labels=target_labels,
                outside_value=outside_value,
                crop_to_bbox=crop_to_bbox,
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
            "selection": request.selection,
            "resolved": request.resolved,
            "client_tag": request.client_tag,
            "source": request.source,
        }
        run_meta = {
            "task_group": str(request.selection.get("task_group") or ""),
            "segmentation_model": str(request.selection.get("segmentation_model") or ""),
            "detection_model": str(request.selection.get("detection_model") or ""),
            "resolved_workflow": str(request.resolved.get("workflow") or request.workflow),
        }
        self._run_storage.materialize_run(run, metadata=run_meta)
        self._run_storage.write_request(run, request_data)
        job_id = self._api.submit(request)
        self._runs_by_job_id[job_id] = run
        self._prediction_store.add(
            JobRecord(
                job_id=job_id,
                run_id=run.run_id,
                workflow=request.workflow,
                task_group=str(request.selection.get("task_group") or ""),
                segmentation_model=str(request.selection.get("segmentation_model") or ""),
                detection_model=str(request.selection.get("detection_model") or ""),
                resolved_workflow=str(request.resolved.get("workflow") or request.workflow),
                scope="current",
                status="queued",
                image_path=current_image,
                image_paths=[],
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

    def terminate(self, job_id: str) -> None:
        self._api.terminate(job_id)
        self._prediction_store.append_log(job_id, "Termination requested.")

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
                meta = {
                    "task_group": str(payload.get("task_group") or ""),
                    "segmentation_model": "",
                    "detection_model": "",
                }
                job = self._prediction_store.get(job_id)
                if job is not None:
                    meta["task_group"] = job.task_group
                    meta["segmentation_model"] = job.segmentation_model
                    meta["detection_model"] = job.detection_model
                    meta["resolved_workflow"] = job.resolved_workflow or job.workflow
                self._run_storage.write_result(run, status="done", payload=payload, metadata=meta)
            self._apply_payload_if_current(payload, partial=False)
            return
        if event_type == "failed":
            error = str(getattr(event, "error", "") or getattr(event, "message", "") or "Prediction failed.")
            self._prediction_store.update_status(job_id, "failed", error=error)
            if run is not None:
                job = self._prediction_store.get(job_id)
                meta = {}
                if job is not None:
                    meta = {
                        "task_group": job.task_group,
                        "segmentation_model": job.segmentation_model,
                        "detection_model": job.detection_model,
                        "resolved_workflow": job.resolved_workflow or job.workflow,
                    }
                self._run_storage.write_result(run, status="failed", payload={}, error=error, metadata=meta)
            return
        if event_type == "cancelled":
            payload = getattr(event.result, "to_dict", lambda: {})() if getattr(event, "result", None) is not None else {}
            self._prediction_store.update_status(job_id, "cancelled")
            if run is not None:
                job = self._prediction_store.get(job_id)
                meta = {}
                if job is not None:
                    meta = {
                        "task_group": job.task_group,
                        "segmentation_model": job.segmentation_model,
                        "detection_model": job.detection_model,
                        "resolved_workflow": job.resolved_workflow or job.workflow,
                    }
                self._run_storage.write_result(
                    run,
                    status="cancelled",
                    payload=payload,
                    error=getattr(event, "message", None),
                    metadata=meta,
                )

    def _apply_payload_if_current(self, payload: dict, *, partial: bool) -> None:
        items = self._payload_items(payload)
        current_path = str(self._workspace_store.current_image_path or "")
        if not current_path:
            return
        if not partial:
            self._workspace_store.set_result_items(items)
        current_item = next((item for item in items if str(item.get("image_path") or "") == current_path), None)
        if current_item is None:
            return
        detections = list(current_item.get("detections") or [])
        if detections:
            self._workspace_store.set_highlight_detections(detections)
            if not partial:
                self._workspace_store.set_detections(detections)
        if partial:
            return
        mask_path = str(current_item.get("mask_path") or "")
        image = self._workspace_store.current_image
        if mask_path and not image.isNull():
            try:
                loaded = load_mask(mask_path, (image.width(), image.height()))
            except Exception:
                loaded = None
            if loaded is not None:
                self._workspace_store.set_current_mask(mask_path, loaded.mask)

    def _payload_items(self, payload: dict) -> list[dict]:
        results = payload.get("results")
        if isinstance(results, list):
            items = [dict(item) for item in results if isinstance(item, dict)]
            if items:
                return items
        return [dict(payload or {})]
