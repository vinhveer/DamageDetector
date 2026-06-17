from __future__ import annotations

from datetime import datetime

from PySide6 import QtCore

from ui.models.job import JobKind, JobSpec, JobStatus, JobUpdate


class JobManager(QtCore.QObject):
    """Lightweight job registry. Job execution is driven by callers, JobManager
    just tracks state and emits signals so panels can render progress."""

    jobAdded = QtCore.Signal(object)              # JobSpec
    jobUpdated = QtCore.Signal(str, object)       # (job_id, JobUpdate)
    jobStatusChanged = QtCore.Signal(object)      # JobSpec
    logEmitted = QtCore.Signal(str, str)          # (job_id, line)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: dict[str, JobSpec] = {}

    def jobs(self) -> list[JobSpec]:
        return list(self._jobs.values())

    def get(self, job_id: str) -> JobSpec | None:
        return self._jobs.get(job_id)

    def submit(self, kind: JobKind, label: str = "", params: dict | None = None) -> JobSpec:
        job = JobSpec(kind=kind, label=label, params=dict(params or {}))
        self._jobs[job.id] = job
        self.jobAdded.emit(job)
        return job

    def mark_running(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.running
        job.started_at = datetime.utcnow()
        self.jobStatusChanged.emit(job)

    def update(self, job_id: str, update: JobUpdate) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        if update.progress is not None:
            job.progress = float(update.progress)
        if update.message is not None:
            job.message = str(update.message)
        if update.log_line:
            self.logEmitted.emit(job_id, update.log_line)
        self.jobUpdated.emit(job_id, update)
        self.jobStatusChanged.emit(job)

    def progress(self, job_id: str, done: int, total: int) -> None:
        if total <= 0:
            return
        self.update(job_id, JobUpdate(progress=done / total, message=f"{done}/{total}"))

    def complete(self, job_id: str, result: object | None = None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.completed
        job.progress = 1.0
        job.result = result
        job.finished_at = datetime.utcnow()
        self.jobStatusChanged.emit(job)

    def fail(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.failed
        job.error = str(error)
        job.message = str(error)
        job.finished_at = datetime.utcnow()
        self.jobStatusChanged.emit(job)

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.cancelled
        job.finished_at = datetime.utcnow()
        self.jobStatusChanged.emit(job)
