from __future__ import annotations

import threading
import uuid
from collections import defaultdict

from dino import get_dino_service
from inference_api.contracts import InferenceRequest, JobEvent, JobSnapshot
from inference_api.workflows import WorkflowContext, run_workflow
from segmentation.sam.runtime import get_sam_service
from segmentation.sam.finetune import get_sam_finetune_service
from segmentation.unet import get_unet_service


class InferenceApi:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, JobSnapshot] = {}
        self._events: dict[str, list[JobEvent]] = defaultdict(list)
        self._cancelled: set[str] = set()
        self._active_services: dict[str, set[str]] = defaultdict(set)
        self._finished_order: list[str] = []
        self._service_getters = {
            "dino": get_dino_service,
            "sam": get_sam_service,
            "sam_finetune": get_sam_finetune_service,
            "unet": get_unet_service,
        }

    def submit(self, request: InferenceRequest) -> str:
        job_id = uuid.uuid4().hex
        snapshot = JobSnapshot(job_id=job_id, workflow=request.workflow, status="queued", request=request)
        with self._lock:
            self._jobs[job_id] = snapshot
            self._events[job_id].append(JobEvent(type="queued", job_id=job_id, workflow=request.workflow))
        thread = threading.Thread(target=self._run_job, args=(job_id, request), name=f"infer:{job_id}", daemon=True)
        thread.start()
        return job_id

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._cancelled.add(job_id)

    def terminate(self, job_id: str) -> None:
        service_names: list[str] = []
        with self._lock:
            if job_id in self._jobs:
                self._cancelled.add(job_id)
                service_names = list(self._active_services.get(job_id, set()))
        for service_name in service_names:
            getter = self._service_getters.get(service_name)
            if getter is None:
                continue
            try:
                getter().close()
            except Exception:
                pass

    def get_job(self, job_id: str) -> JobSnapshot | None:
        with self._lock:
            return self._jobs.get(job_id)

    def drain_events(self, job_id: str) -> list[JobEvent]:
        with self._lock:
            events = list(self._events.get(job_id, []))
            self._events[job_id].clear()
            return events

    def shutdown(self) -> None:
        for getter in (get_dino_service, get_sam_service, get_sam_finetune_service, get_unet_service):
            try:
                getter().close()
            except Exception:
                pass

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def _register_service(self, job_id: str, service_name: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._active_services[job_id].add(str(service_name))

    def _emit(self, event: JobEvent) -> None:
        with self._lock:
            self._events[event.job_id].append(event)

    def _set_status(self, job_id: str, status: str, *, error: str | None = None) -> None:
        with self._lock:
            snapshot = self._jobs[job_id]
            self._jobs[job_id] = JobSnapshot(
                job_id=snapshot.job_id,
                workflow=snapshot.workflow,
                status=status,
                request=snapshot.request,
                error=error,
            )
            if status in {"done", "failed", "cancelled"} and job_id not in self._finished_order:
                self._finished_order.append(job_id)

    def _finalize_job_state(self, job_id: str) -> None:
        with self._lock:
            self._cancelled.discard(job_id)
            self._active_services.pop(job_id, None)
            self._prune_finished_jobs_locked(max_finished=200)

    def _prune_finished_jobs_locked(self, *, max_finished: int) -> None:
        if len(self._finished_order) <= max_finished:
            return
        kept: list[str] = []
        for job_id in self._finished_order:
            if len(kept) < max_finished:
                kept.append(job_id)
                continue
            if self._events.get(job_id):
                kept.append(job_id)
                continue
            self._jobs.pop(job_id, None)
            self._events.pop(job_id, None)
            self._cancelled.discard(job_id)
            self._active_services.pop(job_id, None)
        self._finished_order = kept

    def _run_job(self, job_id: str, request: InferenceRequest) -> None:
        self._set_status(job_id, "running")
        self._emit(JobEvent(type="started", job_id=job_id, workflow=request.workflow))
        ctx = WorkflowContext(
            job_id=job_id,
            request=request,
            emit_event=self._emit,
            stop_checker=lambda: self._is_cancelled(job_id),
            register_service=lambda service_name: self._register_service(job_id, service_name),
        )
        try:
            if self._is_cancelled(job_id):
                self._set_status(job_id, "cancelled")
                self._emit(JobEvent(type="cancelled", job_id=job_id, workflow=request.workflow))
                self._finalize_job_state(job_id)
                return
            result = run_workflow(ctx)
            if self._is_cancelled(job_id) or result.to_dict().get("stopped"):
                self._set_status(job_id, "cancelled")
                self._emit(JobEvent(type="cancelled", job_id=job_id, workflow=request.workflow, result=result))
                self._finalize_job_state(job_id)
                return
            self._set_status(job_id, "done")
            self._emit(JobEvent(type="completed", job_id=job_id, workflow=request.workflow, result=result))
            self._finalize_job_state(job_id)
        except Exception as exc:
            if self._is_cancelled(job_id):
                self._set_status(job_id, "cancelled")
                self._emit(JobEvent(type="cancelled", job_id=job_id, workflow=request.workflow, message=str(exc)))
                self._finalize_job_state(job_id)
                return
            self._set_status(job_id, "failed", error=str(exc))
            self._emit(JobEvent(type="failed", job_id=job_id, workflow=request.workflow, error=str(exc), message=str(exc)))
            self._finalize_job_state(job_id)


_API: InferenceApi | None = None


def get_inference_api() -> InferenceApi:
    global _API
    if _API is None:
        _API = InferenceApi()
    return _API
