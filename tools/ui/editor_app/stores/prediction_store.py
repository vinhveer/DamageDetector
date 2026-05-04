from __future__ import annotations

from typing import Iterable

from PySide6 import QtCore

from ui.editor_app.domain.models import JobRecord


class PredictionStore(QtCore.QObject):
    jobsChanged = QtCore.Signal()
    activeJobChanged = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: dict[str, JobRecord] = {}
        self._job_order: list[str] = []
        self._active_job_id: str | None = None

    def all_jobs(self) -> list[JobRecord]:
        return [self._jobs[job_id] for job_id in self._job_order if job_id in self._jobs]

    def active_jobs(self) -> list[JobRecord]:
        return [job for job in self.all_jobs() if not job.is_finished()]

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def add(self, job: JobRecord) -> None:
        self._jobs[job.job_id] = job
        if job.job_id not in self._job_order:
            self._job_order.insert(0, job.job_id)
        self.set_active_job(job.job_id)
        self.jobsChanged.emit()

    def update_status(self, job_id: str, status: str, *, error: str | None = None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = str(status)
        if error:
            job.error = str(error)
        self.jobsChanged.emit()

    def append_log(self, job_id: str, text: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.logs.append(str(text))
        self.jobsChanged.emit()

    def append_event(self, job_id: str, event_data: dict) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.events.append(dict(event_data))
        self.jobsChanged.emit()

    def append_partial(self, job_id: str, payload: dict) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.partial_payloads.append(dict(payload))
        self.jobsChanged.emit()

    def set_final_payload(self, job_id: str, payload: dict | None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.final_payload = dict(payload or {})
        self.jobsChanged.emit()

    def set_active_job(self, job_id: str | None) -> None:
        self._active_job_id = str(job_id) if job_id else None
        self.activeJobChanged.emit(self._active_job_id or "")

    def active_job(self) -> JobRecord | None:
        if not self._active_job_id:
            return None
        return self._jobs.get(self._active_job_id)

    def job_ids(self) -> Iterable[str]:
        return list(self._job_order)
