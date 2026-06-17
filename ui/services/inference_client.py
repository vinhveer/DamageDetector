from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore

from inference_api.api import get_inference_api
from inference_api.contracts import InferenceRequest, JobEvent

if TYPE_CHECKING:
    from inference_api.contracts import InferenceResult


class InferenceClient(QtCore.QObject):
    """Qt wrapper around InferenceApi.

    Polls drain_events() every 60 ms via QTimer and re-emits as Qt signals
    so panels can update without threading boilerplate.
    """

    logEmitted = QtCore.Signal(str, str)        # (job_id, message)
    progressEmitted = QtCore.Signal(str, float)  # (job_id, 0..1)
    jobCompleted = QtCore.Signal(str, object)    # (job_id, InferenceResult)
    jobFailed = QtCore.Signal(str, str)          # (job_id, error)
    jobCancelled = QtCore.Signal(str)            # (job_id)
    jobStarted = QtCore.Signal(str)              # (job_id)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._api = get_inference_api()
        self._tracked: set[str] = set()

        self._poller = QtCore.QTimer(self)
        self._poller.setInterval(60)
        self._poller.timeout.connect(self._drain_all)
        self._poller.start()

    def submit(self, request: InferenceRequest) -> str:
        job_id = self._api.submit(request)
        self._tracked.add(job_id)
        return job_id

    def cancel(self, job_id: str) -> None:
        self._api.cancel(job_id)

    def terminate(self, job_id: str) -> None:
        self._api.terminate(job_id)

    def _drain_all(self) -> None:
        done: set[str] = set()
        for job_id in list(self._tracked):
            finished = self._drain_job(job_id)
            if finished:
                done.add(job_id)
        self._tracked -= done

    def _drain_job(self, job_id: str) -> bool:
        """Return True when job is terminal (completed/failed/cancelled)."""
        events = self._api.drain_events(job_id)
        terminal = False
        for event in events:
            terminal = terminal or self._handle_event(event)
        return terminal

    def _handle_event(self, event: JobEvent) -> bool:
        """Handle one event. Return True if terminal."""
        etype = event.type
        jid = event.job_id

        if etype == "started":
            self.jobStarted.emit(jid)
        elif etype == "progress":
            if event.message:
                self.logEmitted.emit(jid, event.message)
            if event.progress is not None:
                self.progressEmitted.emit(jid, float(event.progress))
        elif etype == "partial_result":
            if event.message:
                self.logEmitted.emit(jid, event.message)
        elif etype == "completed":
            if event.result is not None:
                self.jobCompleted.emit(jid, event.result)
            else:
                self.jobCompleted.emit(jid, None)
            return True
        elif etype == "failed":
            self.jobFailed.emit(jid, str(event.error or event.message or "Unknown error"))
            return True
        elif etype == "cancelled":
            self.jobCancelled.emit(jid)
            return True

        return False
