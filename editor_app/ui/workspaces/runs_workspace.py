from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from editor_app.domain.models import JobRecord


class RunsWorkspace(QtWidgets.QWidget):
    stopRequested = QtCore.Signal(str)
    openRunRequested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self._summary = QtWidgets.QLabel("No job selected", self)
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        actions = QtWidgets.QHBoxLayout()
        self._stop_btn = QtWidgets.QPushButton("Stop Job", self)
        self._open_btn = QtWidgets.QPushButton("Open Run Folder", self)
        self._stop_btn.clicked.connect(self._emit_stop)
        self._open_btn.clicked.connect(self._emit_open)
        actions.addWidget(self._stop_btn)
        actions.addWidget(self._open_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        self._log = QtWidgets.QPlainTextEdit(self)
        self._log.setReadOnly(True)
        root.addWidget(self._log, 1)

        self._job: JobRecord | None = None

    def set_job(self, job: JobRecord | None) -> None:
        self._job = job
        if job is None:
            self._summary.setText("No job selected")
            self._log.clear()
            self._stop_btn.setEnabled(False)
            self._open_btn.setEnabled(False)
            return
        self._summary.setText(
            f"Run {job.run_id}\n"
            f"Workflow: {job.workflow}\n"
            f"Scope: {job.scope}\n"
            f"Status: {job.status}\n"
            f"Output: {job.output_dir}"
        )
        self._log.setPlainText("\n".join(job.logs))
        self._stop_btn.setEnabled(not job.is_finished())
        self._open_btn.setEnabled(bool(job.run_dir))

    def _emit_stop(self) -> None:
        if self._job is not None:
            self.stopRequested.emit(self._job.job_id)

    def _emit_open(self) -> None:
        if self._job is not None and self._job.run_dir:
            self.openRunRequested.emit(self._job.run_dir)
