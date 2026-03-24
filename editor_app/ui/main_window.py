from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from editor_app.config.prediction_settings import (
    DEFAULT_EDITOR_SETTINGS,
    migrate_editor_settings,
)
from editor_app.controllers.history_controller import HistoryController
from editor_app.controllers.compare_controller import CompareController
from editor_app.controllers.editor_controller import EditorController
from editor_app.controllers.isolate_controller import IsolateController
from editor_app.controllers.prediction_controller import PredictionController
from editor_app.controllers.workspace_controller import WorkspaceController
from editor_app.services.compare_service import CompareService
from editor_app.services.export_service import ExportService
from editor_app.services.file_service import FileService
from editor_app.services.run_storage import RunStorageService
from editor_app.services.settings_service import SettingsService
from editor_app.stores.history_store import HistoryStore
from editor_app.stores.compare_store import CompareStore
from editor_app.stores.isolate_store import IsolateStore
from editor_app.stores.prediction_store import PredictionStore
from editor_app.stores.ui_store import UiStore
from editor_app.stores.workspace_store import WorkspaceStore
from editor_app.ui.actions.prediction_actions import PredictionActions
from editor_app.ui.actions.workspace_actions import WorkspaceActions
from editor_app.ui.widgets.left_rail import LeftRail
from editor_app.ui.widgets.top_bar import TopBar
from editor_app.ui.workspaces.compare_workspace import CompareWorkspace
from editor_app.ui.workspaces.editor_workspace import EditorWorkspace
from editor_app.ui.workspaces.history_workspace import HistoryWorkspace
from editor_app.ui.workspaces.isolate_workspace import IsolateWorkspace
from editor_app.ui.workspaces.runs_workspace import RunsWorkspace
from editor_app.ui.workspaces.settings_workspace import SettingsWorkspace


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("editor_app")
        self.resize(1680, 980)

        self._workspace_store = WorkspaceStore(self)
        self._prediction_store = PredictionStore(self)
        self._ui_store = UiStore(self)
        self._history_store = HistoryStore(self)
        self._compare_store = CompareStore(self)
        self._isolate_store = IsolateStore(self)

        self._file_service = FileService()
        self._run_storage = RunStorageService()
        self._compare_service = CompareService(self._run_storage)
        self._export_service = ExportService()
        self._settings_service = SettingsService()
        self._workspace_controller = WorkspaceController(self._workspace_store, self._history_store, self._file_service)
        self._editor_controller = EditorController(self._workspace_store, self._workspace_controller)
        self._history_controller = HistoryController(self._workspace_store, self._history_store, self._run_storage)
        self._compare_controller = CompareController(self._compare_store, self._history_store, self._compare_service)
        self._isolate_controller = IsolateController(
            self._workspace_store,
            self._history_store,
            self._isolate_store,
            self._run_storage,
        )
        self._prediction_controller = PredictionController(
            self._workspace_store,
            self._prediction_store,
            self._history_store,
            self._run_storage,
            self,
        )
        self._prediction_controller.errorRaised.connect(self._show_error)
        self._prediction_store.jobsChanged.connect(self._refresh_jobs)
        self._prediction_store.activeJobChanged.connect(self._refresh_runs_workspace)
        self.setWindowTitle("editor_app")

        persisted = self._settings_service.load()
        merged_settings = dict(DEFAULT_EDITOR_SETTINGS)
        merged_settings.update(migrate_editor_settings(dict(persisted.get("settings") or {})))
        self._saved_settings = dict(merged_settings)
        self._ui_store.set_settings(merged_settings)
        self._ui_store.set_layout(
            main_splitter_sizes=list(persisted.get("main_splitter_sizes") or []),
            left_splitter_sizes=list(persisted.get("left_splitter_sizes") or []),
        )
        self._compare_store.set_gt_dir(str(persisted.get("compare_gt_dir") or ""))
        self._compare_store.set_affix(str(persisted.get("compare_affix") or ""))
        self._compare_store.set_selected_run_dir(str(persisted.get("compare_selected_run_dir") or "") or None)

        central = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._top_bar = TopBar(central)
        self._top_bar.workspaceRequested.connect(self._show_workspace)
        root.addWidget(self._top_bar)

        self._main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        self._main_splitter.setChildrenCollapsible(False)
        self._left_rail = LeftRail(self._main_splitter)
        self._left_rail.stopJobRequested.connect(self._prediction_controller.cancel)
        self._left_rail.jobActivated.connect(self._activate_job)

        self._workspaces = QtWidgets.QStackedWidget(self._main_splitter)
        self._editor_workspace = EditorWorkspace(self._workspaces)
        self._runs_workspace = RunsWorkspace(self._workspaces)
        self._history_workspace = HistoryWorkspace(self._workspaces)
        self._compare_workspace = CompareWorkspace(self._workspaces)
        self._isolate_workspace = IsolateWorkspace(self._workspaces)
        self._settings_workspace = SettingsWorkspace(self._workspaces)
        self._runs_workspace.stopRequested.connect(self._prediction_controller.cancel)
        self._runs_workspace.terminateRequested.connect(self._prediction_controller.terminate)
        self._history_workspace.refresh_button().clicked.connect(self._history_controller.refresh)
        self._history_workspace.runSelected.connect(self._on_history_run_selected)
        self._history_workspace.loadItemRequested.connect(self._load_run_item_into_editor)
        self._compare_workspace.selectedRunChanged.connect(self._compare_controller.set_selected_run_dir)
        self._compare_workspace.groundTruthDirChanged.connect(self._compare_controller.set_ground_truth_dir)
        self._compare_workspace.affixChanged.connect(self._compare_controller.set_affix)
        self._compare_workspace.compareRequested.connect(self._compare_selected_run)
        self._isolate_workspace.refresh_button().clicked.connect(self._isolate_controller.refresh)
        self._isolate_workspace.openIsolateRequested.connect(lambda path: self._load_image(path, switch_workspace=True))
        self._editor_workspace.roiBoxSelected.connect(self._on_roi_box_selected)
        self._editor_workspace.roiSelectionCanceled.connect(self._on_roi_canceled)
        self._editor_workspace.roiAdded.connect(self._on_roi_added)
        self._editor_workspace.roiUpdated.connect(self._on_roi_updated)
        self._editor_workspace.roiDeleted.connect(self._on_roi_deleted)
        self._editor_workspace.roiCleared.connect(self._on_roi_cleared)
        self._editor_workspace.roiSelectionChanged.connect(self._on_roi_selection_changed)
        self._settings_workspace.settingsSaved.connect(self._save_settings_workspace)
        self._workspaces.addWidget(self._editor_workspace)
        self._workspaces.addWidget(self._runs_workspace)
        self._workspaces.addWidget(self._history_workspace)
        self._workspaces.addWidget(self._compare_workspace)
        self._workspaces.addWidget(self._isolate_workspace)
        self._workspaces.addWidget(self._settings_workspace)

        self._main_splitter.addWidget(self._left_rail)
        self._main_splitter.addWidget(self._workspaces)
        self._main_splitter.setSizes(self._ui_store.main_splitter_sizes or [420, 1260])
        if self._ui_store.left_splitter_sizes:
            self._left_rail.set_splitter_sizes(self._ui_store.left_splitter_sizes)
        root.addWidget(self._main_splitter, 1)
        self.setCentralWidget(central)
        self._workspace_actions = WorkspaceActions(
            parent=self,
            workspace_store=self._workspace_store,
            workspace_controller=self._workspace_controller,
            history_controller=self._history_controller,
            isolate_controller=self._isolate_controller,
            left_rail=self._left_rail,
            editor_workspace=self._editor_workspace,
            export_service=self._export_service,
            show_workspace=self._show_workspace,
            show_error=self._show_error,
            show_status=self.statusBar().showMessage,
            persist_state=self._persist_editor_state,
        )
        self._prediction_actions = PredictionActions(
            parent=self,
            workspace_store=self._workspace_store,
            ui_store=self._ui_store,
            prediction_store=self._prediction_store,
            prediction_controller=self._prediction_controller,
            editor_controller=self._editor_controller,
            history_controller=self._history_controller,
            compare_controller=self._compare_controller,
            show_workspace=self._show_workspace,
            show_error=self._show_error,
            show_status=self.statusBar().showMessage,
            persist_state=self._persist_editor_state,
        )
        self._top_bar.predictRequested.connect(self._prediction_actions.run_predict_dialog)
        self._top_bar.predictRoiRequested.connect(lambda: self._prediction_actions.run_predict_dialog_roi(self._editor_workspace.start_prediction_roi_selection))
        self._top_bar.isolateRequested.connect(self._prediction_actions.run_isolate)
        self._left_rail.openFolderRequested.connect(self._workspace_actions.open_folder)
        self._left_rail.openImageRequested.connect(self._workspace_actions.open_image_dialog)
        self._left_rail.openMaskRequested.connect(self._workspace_actions.open_mask)
        self._left_rail.saveMaskRequested.connect(self._workspace_actions.save_mask)
        self._left_rail.openRunRequested.connect(self._workspace_actions.open_local_folder)
        self._left_rail.explorer().imageClicked.connect(lambda path: self._workspace_actions.load_image(path, switch_workspace=False))
        self._left_rail.explorer().imageActivated.connect(lambda path: self._workspace_actions.load_image(path, switch_workspace=True))
        self._runs_workspace.openRunRequested.connect(self._workspace_actions.open_local_folder)
        self._history_workspace.openRunRequested.connect(self._workspace_actions.open_local_folder)
        self._isolate_workspace.openRunRequested.connect(self._workspace_actions.open_local_folder)

        self._workspace_store.imageChanged.connect(self._on_store_image_changed)
        self._workspace_store.maskChanged.connect(self._on_store_mask_changed)
        self._workspace_store.detectionsChanged.connect(self._on_store_detections_changed)
        self._workspace_store.highlightChanged.connect(self._on_store_highlights_changed)
        self._workspace_store.roiChanged.connect(self._on_store_rois_changed)
        self._history_store.runsChanged.connect(self._refresh_history_workspace)
        self._compare_store.runsChanged.connect(self._refresh_compare_workspace)
        self._compare_store.configChanged.connect(self._refresh_compare_workspace)
        self._compare_store.configChanged.connect(self._persist_editor_state)
        self._compare_store.resultsChanged.connect(self._refresh_compare_workspace)
        self._isolate_store.itemsChanged.connect(self._refresh_isolate_workspace)
        self._ui_store.settingsChanged.connect(self._refresh_settings_workspace)
        self._ui_store.layoutChanged.connect(self._persist_editor_state)
        self.statusBar().showMessage("Ready")
        self._build_actions()
        self._show_workspace(str(persisted.get("current_workspace_view") or "editor"))
        self._refresh_settings_workspace()
        last_workspace = str(persisted.get("last_workspace") or "").strip()
        if last_workspace and Path(last_workspace).is_dir():
            images = self._workspace_controller.open_folder(last_workspace, auto_open_first=False)
            self._left_rail.explorer().set_images(images)
            self._history_controller.refresh()
        if not self._workspace_store.current_image_path:
            self.setWindowTitle("editor_app")

    def _on_store_image_changed(self) -> None:
        current_path = self._workspace_store.current_image_path
        self._top_bar.set_current_path(current_path)
        self.setWindowTitle(f"editor_app - {current_path}" if current_path else "editor_app")
        if not self._workspace_store.current_image.isNull():
            self._editor_workspace.set_image(
                self._workspace_store.current_image,
                self._workspace_store.current_image_path,
            )

    def _on_store_mask_changed(self) -> None:
        if not self._workspace_store.current_mask.isNull():
            self._editor_workspace.set_mask(self._workspace_store.current_mask)

    def _on_store_detections_changed(self) -> None:
        self._editor_workspace.set_detections(self._workspace_store.current_detections)
        self._editor_workspace.set_highlight_detections(
            self._workspace_store.highlight_detections or self._workspace_store.current_detections
        )

    def _on_store_highlights_changed(self) -> None:
        self._editor_workspace.set_highlight_detections(
            self._workspace_store.highlight_detections or self._workspace_store.current_detections
        )

    def _on_store_rois_changed(self) -> None:
        self._editor_workspace.set_roi_boxes(self._workspace_store.current_rois, self._workspace_store.current_roi_index)

    def _refresh_jobs(self) -> None:
        self._left_rail.set_jobs(self._prediction_store.all_jobs())
        self._refresh_runs_workspace()
        self._history_controller.refresh()

    def _refresh_runs_workspace(self, *_args) -> None:
        self._runs_workspace.set_job(self._prediction_store.active_job())

    def _refresh_history_workspace(self) -> None:
        self._history_workspace.set_runs(self._history_store.runs)

    def _refresh_compare_workspace(self) -> None:
        self._compare_workspace.set_runs(self._compare_store.runs)
        self._compare_workspace.set_compare_config(
            gt_dir=self._compare_store.gt_dir,
            affix=self._compare_store.affix,
            selected_run_dir=self._compare_store.selected_run_dir,
        )
        self._compare_workspace.set_results(self._compare_store.results)

    def _refresh_isolate_workspace(self) -> None:
        self._isolate_workspace.set_items(self._isolate_store.items)

    def _refresh_settings_workspace(self) -> None:
        self._settings_workspace.set_settings(self._ui_store.settings)

    def _show_workspace(self, name: str) -> None:
        mapping = {
            "editor": self._editor_workspace,
            "runs": self._runs_workspace,
            "history": self._history_workspace,
            "compare": self._compare_workspace,
            "isolate": self._isolate_workspace,
            "settings": self._settings_workspace,
        }
        widget = mapping.get(str(name), self._editor_workspace)
        self._ui_store.set_workspace_view(str(name))
        self._top_bar.set_active_workspace(str(name))
        self._workspaces.setCurrentWidget(widget)

    def _activate_job(self, job_id: str) -> None:
        self._prediction_store.set_active_job(job_id)
        self._show_workspace("runs")

    def _load_image(self, path: str, *, switch_workspace: bool) -> None:
        self._workspace_actions.load_image(path, switch_workspace=switch_workspace)

    def _show_error(self, text: str) -> None:
        QtWidgets.QMessageBox.critical(self, "editor_app", str(text or "Unknown error"))

    def _on_roi_box_selected(self, roi_box_obj) -> None:
        self._prediction_actions.on_roi_box_selected(roi_box_obj)

    def _on_roi_canceled(self) -> None:
        self._prediction_actions.on_roi_canceled()

    def _on_roi_added(self, roi_box_obj) -> None:
        if roi_box_obj is None:
            return
        self._editor_controller.add_roi(tuple(int(x) for x in roi_box_obj))
        self.statusBar().showMessage("ROI added.", 2500)

    def _on_roi_updated(self, index: int, roi_box_obj) -> None:
        if roi_box_obj is None:
            return
        self._editor_controller.update_roi(int(index), tuple(int(x) for x in roi_box_obj))
        self.statusBar().showMessage("ROI updated.", 2500)

    def _on_roi_deleted(self, index: int) -> None:
        self._editor_controller.delete_roi(int(index))
        self.statusBar().showMessage("ROI deleted.", 2500)

    def _on_roi_cleared(self) -> None:
        self._editor_controller.clear_rois()
        self.statusBar().showMessage("ROI list cleared.", 2500)

    def _on_roi_selection_changed(self, index: int) -> None:
        self._editor_controller.select_roi(int(index))

    def _load_run_item_into_editor(self, payload: dict) -> None:
        image_path = self._prediction_actions.load_run_item_into_editor(payload)
        if image_path:
            self._left_rail.explorer().select_path(image_path)

    def _on_history_run_selected(self, run_dir: str) -> None:
        self._prediction_actions.on_history_run_selected(run_dir, self._history_workspace.set_selected_run_details)

    def _compare_selected_run(self, run_dir: str) -> None:
        self._prediction_actions.compare_selected_run(run_dir)

    def _save_settings_workspace(self, payload: dict) -> None:
        settings = dict(self._ui_store.settings)
        settings.update(dict(payload or {}))
        self._ui_store.set_settings(settings)
        self._saved_settings = dict(settings)
        self._persist_editor_state()
        self.statusBar().showMessage("Saved default prediction settings.", 4000)

    def _build_actions(self) -> None:
        act_open_image = QtGui.QAction("Open Image...", self)
        act_open_image.setShortcut(QtGui.QKeySequence.Open)
        act_open_image.triggered.connect(self._workspace_actions.open_image_dialog)
        self.addAction(act_open_image)

        act_open_folder = QtGui.QAction("Open Workspace Folder...", self)
        act_open_folder.setShortcut(QtGui.QKeySequence("Ctrl+Shift+O"))
        act_open_folder.triggered.connect(self._workspace_actions.open_folder)
        self.addAction(act_open_folder)

        act_open_mask = QtGui.QAction("Open Mask...", self)
        act_open_mask.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        act_open_mask.triggered.connect(self._workspace_actions.open_mask)
        self.addAction(act_open_mask)

        act_save_mask = QtGui.QAction("Save Mask As...", self)
        act_save_mask.setShortcut(QtGui.QKeySequence.Save)
        act_save_mask.triggered.connect(self._workspace_actions.save_mask)
        self.addAction(act_save_mask)

        act_export_image = QtGui.QAction("Export Image...", self)
        act_export_image.triggered.connect(self._workspace_actions.export_image)
        self.addAction(act_export_image)

        act_export_overlay = QtGui.QAction("Export Overlay...", self)
        act_export_overlay.triggered.connect(self._workspace_actions.export_overlay)
        self.addAction(act_export_overlay)

        act_export_overlay_boxes = QtGui.QAction("Export Overlay + Boxes...", self)
        act_export_overlay_boxes.triggered.connect(self._workspace_actions.export_overlay_boxes)
        self.addAction(act_export_overlay_boxes)

        act_prev = QtGui.QAction("Previous Image", self)
        act_prev.setShortcut(QtGui.QKeySequence("PgUp"))
        act_prev.triggered.connect(lambda: self._workspace_actions.navigate_image(-1))
        self.addAction(act_prev)

        act_next = QtGui.QAction("Next Image", self)
        act_next.setShortcut(QtGui.QKeySequence("PgDown"))
        act_next.triggered.connect(lambda: self._workspace_actions.navigate_image(1))
        self.addAction(act_next)

        act_editor = QtGui.QAction("Workspace: Editor", self)
        act_editor.setShortcut(QtGui.QKeySequence("Ctrl+1"))
        act_editor.triggered.connect(lambda: self._show_workspace("editor"))
        self.addAction(act_editor)

        act_runs = QtGui.QAction("Workspace: Runs", self)
        act_runs.setShortcut(QtGui.QKeySequence("Ctrl+2"))
        act_runs.triggered.connect(lambda: self._show_workspace("runs"))
        self.addAction(act_runs)

        act_history = QtGui.QAction("Workspace: History", self)
        act_history.setShortcut(QtGui.QKeySequence("Ctrl+3"))
        act_history.triggered.connect(lambda: self._show_workspace("history"))
        self.addAction(act_history)

        act_compare = QtGui.QAction("Workspace: Compare", self)
        act_compare.setShortcut(QtGui.QKeySequence("Ctrl+4"))
        act_compare.triggered.connect(lambda: self._show_workspace("compare"))
        self.addAction(act_compare)

        act_isolate = QtGui.QAction("Workspace: Isolate", self)
        act_isolate.setShortcut(QtGui.QKeySequence("Ctrl+5"))
        act_isolate.triggered.connect(lambda: self._show_workspace("isolate"))
        self.addAction(act_isolate)

        act_settings = QtGui.QAction("Workspace: Settings", self)
        act_settings.setShortcut(QtGui.QKeySequence("Ctrl+6"))
        act_settings.triggered.connect(lambda: self._show_workspace("settings"))
        self.addAction(act_settings)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(act_open_folder)
        file_menu.addAction(act_open_image)
        file_menu.addAction(act_open_mask)
        file_menu.addSeparator()
        file_menu.addAction(act_save_mask)
        file_menu.addAction(act_export_image)
        file_menu.addAction(act_export_overlay)
        file_menu.addAction(act_export_overlay_boxes)

        nav_menu = self.menuBar().addMenu("Navigate")
        nav_menu.addAction(act_prev)
        nav_menu.addAction(act_next)

        workspace_menu = self.menuBar().addMenu("Workspace")
        workspace_menu.addAction(act_editor)
        workspace_menu.addAction(act_runs)
        workspace_menu.addAction(act_history)
        workspace_menu.addAction(act_compare)
        workspace_menu.addAction(act_isolate)
        workspace_menu.addAction(act_settings)

    def _persist_editor_state(self) -> None:
        payload = {
            "settings": dict(self._saved_settings),
            "last_workspace": str(self._workspace_store.workspace_root or ""),
            "current_workspace_view": str(self._ui_store.current_workspace_view or "editor"),
            "main_splitter_sizes": [int(value) for value in self._main_splitter.sizes()],
            "left_splitter_sizes": self._left_rail.splitter_sizes(),
            "compare_gt_dir": self._compare_store.gt_dir,
            "compare_affix": self._compare_store.affix,
            "compare_selected_run_dir": self._compare_store.selected_run_dir or "",
        }
        try:
            self._settings_service.save(payload)
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._persist_editor_state()
        return super().closeEvent(event)
