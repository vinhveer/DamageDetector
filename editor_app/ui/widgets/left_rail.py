from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from editor_app.domain.models import JobRecord
from editor_app.ui.components.explorer_panel import ExplorerPanel


class LeftRail(QtWidgets.QWidget):
    openFolderRequested = QtCore.Signal()
    openImageRequested = QtCore.Signal()
    openMaskRequested = QtCore.Signal()
    saveMaskRequested = QtCore.Signal()
    stopJobRequested = QtCore.Signal(str)
    openRunRequested = QtCore.Signal(str)
    jobActivated = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setDocumentMode(True)
        self._tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)

        explorer_host = QtWidgets.QWidget(self._tabs)
        explorer_layout = QtWidgets.QVBoxLayout(explorer_host)
        explorer_layout.setContentsMargins(0, 0, 0, 0)
        explorer_layout.setSpacing(6)
        self._explorer = ExplorerPanel(explorer_host)
        explorer_layout.addWidget(self._explorer, 1)

        jobs_host = QtWidgets.QWidget(self._tabs)
        jobs_layout = QtWidgets.QVBoxLayout(jobs_host)
        jobs_layout.setContentsMargins(0, 0, 0, 0)
        jobs_layout.setSpacing(6)
        self._job_list = QtWidgets.QTreeWidget(jobs_host)
        self._job_list.setHeaderLabels(["Run", "Workflow", "Status"])
        self._job_list.itemSelectionChanged.connect(self._emit_job_activated)
        jobs_layout.addWidget(self._job_list, 1)
        job_buttons = QtWidgets.QHBoxLayout()
        self._stop_btn = QtWidgets.QPushButton("Stop", jobs_host)
        self._open_btn = QtWidgets.QPushButton("Open Run", jobs_host)
        self._stop_btn.clicked.connect(self._emit_stop_job)
        self._open_btn.clicked.connect(self._emit_open_run)
        job_buttons.addWidget(self._stop_btn)
        job_buttons.addWidget(self._open_btn)
        jobs_layout.addLayout(job_buttons)

        self._tabs.addTab(explorer_host, "Explorer")
        self._tabs.addTab(jobs_host, "Jobs")
        root.addWidget(self._tabs, 1)

    def explorer(self) -> ExplorerPanel:
        return self._explorer

    def splitter_sizes(self) -> list[int]:
        return []

    def set_splitter_sizes(self, sizes: list[int]) -> None:
        _ = sizes

    def set_jobs(self, jobs: list[JobRecord]) -> None:
        current_id = self.current_job_id()
        self._job_list.clear()
        for job in jobs:
            selection = dict(job.request_data.get("selection") or {})
            seg = str(selection.get("segmentation_model_label") or job.segmentation_model or "").strip()
            det = str(selection.get("detection_model_label") or job.detection_model or "").strip()
            workflow_label = " + ".join(part for part in (seg, det) if part) or job.resolved_workflow or job.workflow
            item = QtWidgets.QTreeWidgetItem([job.run_id, str(workflow_label), job.status])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, job.job_id)
            item.setToolTip(0, job.run_dir)
            self._job_list.addTopLevelItem(item)
            if current_id and job.job_id == current_id:
                self._job_list.setCurrentItem(item)
        if not current_id and self._job_list.topLevelItemCount() > 0:
            self._job_list.setCurrentItem(self._job_list.topLevelItem(0))
        if jobs:
            self._tabs.setTabText(1, f"Jobs ({len(jobs)})")
        else:
            self._tabs.setTabText(1, "Jobs")

    def current_job_id(self) -> str | None:
        item = self._job_list.currentItem()
        if item is None:
            return None
        job_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return str(job_id) if job_id else None

    def _emit_job_activated(self) -> None:
        job_id = self.current_job_id()
        if job_id:
            self._tabs.setCurrentIndex(1)
            self.jobActivated.emit(job_id)

    def _emit_stop_job(self) -> None:
        job_id = self.current_job_id()
        if job_id:
            self.stopJobRequested.emit(job_id)

    def _emit_open_run(self) -> None:
        item = self._job_list.currentItem()
        if item is None:
            return
        path = item.toolTip(0)
        if path:
            self.openRunRequested.emit(str(path))
