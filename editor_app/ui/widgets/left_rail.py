from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from editor_app.domain.models import JobRecord
from editor_app.ui.components.explorer_panel import ExplorerPanel


class LeftRail(QtWidgets.QWidget):
    openFolderRequested = QtCore.Signal()
    addFolderImagesRequested = QtCore.Signal()
    openImageRequested = QtCore.Signal()
    openMaskRequested = QtCore.Signal()
    saveMaskRequested = QtCore.Signal()
    stopJobRequested = QtCore.Signal(str)
    openRunRequested = QtCore.Signal(str)
    jobActivated = QtCore.Signal(str)
    railTabChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setDocumentMode(True)
        self._tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self._tabs.currentChanged.connect(self._emit_rail_tab_changed)
        self._tab_keys: dict[QtWidgets.QWidget, str] = {}

        explorer_host = QtWidgets.QWidget(self._tabs)
        explorer_layout = QtWidgets.QVBoxLayout(explorer_host)
        explorer_layout.setContentsMargins(0, 0, 0, 0)
        explorer_layout.setSpacing(6)
        self._explorer = ExplorerPanel(explorer_host)
        self._explorer.addFolderImagesRequested.connect(self.addFolderImagesRequested.emit)
        explorer_layout.addWidget(self._explorer, 1)

        self._jobs_host = QtWidgets.QWidget(self._tabs)
        jobs_layout = QtWidgets.QVBoxLayout(self._jobs_host)
        jobs_layout.setContentsMargins(0, 0, 0, 0)
        jobs_layout.setSpacing(6)
        self._job_list = QtWidgets.QTreeWidget(self._jobs_host)
        self._job_list.setHeaderLabels(["Run", "Workflow", "Status"])
        self._job_list.itemSelectionChanged.connect(self._emit_job_activated)
        self._job_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._job_list.customContextMenuRequested.connect(self._open_job_context_menu)
        jobs_layout.addWidget(self._job_list, 1)

        self._tabs.addTab(explorer_host, "Explorer")
        self._tab_keys[explorer_host] = "explorer"
        self._tabs.addTab(self._jobs_host, "Jobs")
        self._tab_keys[self._jobs_host] = "jobs"
        root.addWidget(self._tabs, 1)
        self._inspect_panel: QtWidgets.QWidget | None = None
        self._editor_tools_panel: QtWidgets.QWidget | None = None

    def explorer(self) -> ExplorerPanel:
        return self._explorer

    def splitter_sizes(self) -> list[int]:
        return []

    def set_splitter_sizes(self, sizes: list[int]) -> None:
        _ = sizes

    def set_editor_panels(self, inspect_panel: QtWidgets.QWidget, editor_tools_panel: QtWidgets.QWidget) -> None:
        if self._inspect_panel is not None:
            index = self._tabs.indexOf(self._inspect_panel)
            if index >= 0:
                self._tabs.removeTab(index)
            self._tab_keys.pop(self._inspect_panel, None)
        if self._editor_tools_panel is not None:
            index = self._tabs.indexOf(self._editor_tools_panel)
            if index >= 0:
                self._tabs.removeTab(index)
            self._tab_keys.pop(self._editor_tools_panel, None)
        self._inspect_panel = inspect_panel
        self._editor_tools_panel = editor_tools_panel
        inspect_panel.setParent(self._tabs)
        editor_tools_panel.setParent(self._tabs)
        self._tabs.insertTab(1, inspect_panel, "Inspect")
        self._tabs.insertTab(2, editor_tools_panel, "Editor")
        self._tab_keys[inspect_panel] = "inspect"
        self._tab_keys[editor_tools_panel] = "editor"

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
        jobs_index = self._tabs.indexOf(self._jobs_host)
        if jobs:
            if jobs_index >= 0:
                self._tabs.setTabText(jobs_index, f"Jobs ({len(jobs)})")
        else:
            if jobs_index >= 0:
                self._tabs.setTabText(jobs_index, "Jobs")

    def current_job_id(self) -> str | None:
        item = self._job_list.currentItem()
        if item is None:
            return None
        job_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return str(job_id) if job_id else None

    def _emit_job_activated(self) -> None:
        job_id = self.current_job_id()
        if job_id:
            jobs_index = self._tabs.indexOf(self._jobs_host)
            if jobs_index >= 0:
                self._tabs.setCurrentIndex(jobs_index)
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

    def _open_job_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._job_list.itemAt(pos)
        if item is None:
            return
        self._job_list.setCurrentItem(item)
        menu = QtWidgets.QMenu(self)
        open_action = menu.addAction("Open Run Folder")
        stop_action = menu.addAction("Stop Job")
        action = menu.exec(self._job_list.viewport().mapToGlobal(pos))
        if action == open_action:
            self._emit_open_run()
        elif action == stop_action:
            self._emit_stop_job()

    def _emit_rail_tab_changed(self, index: int) -> None:
        widget = self._tabs.widget(index)
        key = self._tab_keys.get(widget, "")
        if key:
            self.railTabChanged.emit(key)
