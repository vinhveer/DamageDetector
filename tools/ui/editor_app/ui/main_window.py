from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ui.editor_app.config.prediction_settings import (
    DEFAULT_EDITOR_SETTINGS,
    migrate_editor_settings,
)
from ui.editor_app.controllers.history_controller import HistoryController
from ui.editor_app.controllers.compare_controller import CompareController
from ui.editor_app.controllers.editor_controller import EditorController
from ui.editor_app.controllers.isolate_controller import IsolateController
from ui.editor_app.controllers.prediction_controller import PredictionController
from ui.editor_app.controllers.workspace_controller import WorkspaceController
from ui.editor_app.services.compare_service import CompareService
from ui.editor_app.services.export_service import ExportService
from ui.editor_app.services.file_service import FileService
from ui.editor_app.services.run_storage import RunStorageService
from ui.editor_app.services.settings_service import SettingsService
from ui.editor_app.stores.history_store import HistoryStore
from ui.editor_app.stores.compare_store import CompareStore
from ui.editor_app.stores.isolate_store import IsolateStore
from ui.editor_app.stores.prediction_store import PredictionStore
from ui.editor_app.stores.ui_store import UiStore
from ui.editor_app.stores.workspace_store import WorkspaceStore
from ui.editor_app.ui.actions.prediction_actions import PredictionActions
from ui.editor_app.ui.actions.workspace_actions import WorkspaceActions
from ui.editor_app.ui.widgets.left_rail import LeftRail
from ui.editor_app.ui.widgets.top_bar import TopBar
from ui.editor_app.ui.workspaces.compare_workspace import CompareWorkspace
from ui.editor_app.ui.workspaces.editor_workspace import EditorWorkspace
from ui.editor_app.ui.workspaces.history_workspace import HistoryWorkspace
from ui.editor_app.ui.workspaces.isolate_workspace import IsolateWorkspace
from ui.editor_app.ui.workspaces.runs_workspace import RunsWorkspace
from ui.editor_app.ui.workspaces.settings_workspace import SettingsWorkspace


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
        self._editor_controller = EditorController(self._workspace_store, self._workspace_controller, self._run_storage)
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
        self._saved_global_settings = dict(DEFAULT_EDITOR_SETTINGS)
        self._saved_global_settings.update(migrate_editor_settings(dict(persisted.get("settings") or {})))
        self._settings_by_workspace = self._migrate_settings_by_workspace(persisted.get("settings_by_workspace"))
        self._active_settings_workspace_key = ""
        self._ui_store.set_settings(self._settings_for_workspace(None))
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
        self._left_rail.railTabChanged.connect(self._on_left_rail_tab_changed)

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
        self._left_rail.set_editor_panels(self._editor_workspace.inspect_panel(), self._editor_workspace.tools_panel())
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
        self._top_bar.saveImageRequested.connect(self._workspace_actions.export_image)
        self._left_rail.openFolderRequested.connect(self._workspace_actions.open_folder)
        self._left_rail.addFolderImagesRequested.connect(self._workspace_actions.add_folder_images)
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
        self._workspace_store.workspaceChanged.connect(self._on_workspace_changed)
        self._history_store.runsChanged.connect(self._refresh_history_workspace)
        self._compare_store.runsChanged.connect(self._refresh_compare_workspace)
        self._compare_store.configChanged.connect(self._refresh_compare_workspace)
        self._compare_store.configChanged.connect(self._persist_editor_state)
        self._compare_store.resultsChanged.connect(self._refresh_compare_workspace)
        self._isolate_store.itemsChanged.connect(self._refresh_isolate_workspace)
        self._ui_store.settingsChanged.connect(self._refresh_settings_workspace)
        self._ui_store.layoutChanged.connect(self._persist_editor_state)
        self._prediction_controller.jobCompleted.connect(self._on_prediction_job_completed)
        self._prediction_controller.jobFinalized.connect(lambda _job_id: self._history_controller.refresh())
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

    def _on_workspace_changed(self) -> None:
        self._apply_workspace_settings(self._workspace_store.workspace_root)

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

    def _on_left_rail_tab_changed(self, name: str) -> None:
        self._editor_workspace.set_left_rail_editor_active(str(name) == "editor")

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
            self._left_rail.explorer().set_images(list(self._workspace_store.images))
            self._left_rail.explorer().select_path(image_path)

    def _on_history_run_selected(self, run_dir: str) -> None:
        bundle, _items = self._prediction_actions.on_history_run_selected(run_dir, self._history_workspace.set_selected_run_details)
        run_meta = dict(bundle.get("run") or {})
        run_id = str(run_meta.get("run_id") or Path(run_dir).name or "").strip()
        if run_id:
            self.statusBar().showMessage(f"History: {run_id}")

    def _compare_selected_run(self, run_dir: str) -> None:
        self._prediction_actions.compare_selected_run(run_dir)

    def _save_settings_workspace(self, payload: dict) -> None:
        settings = dict(self._ui_store.settings)
        settings.update(dict(payload or {}))
        self._ui_store.set_settings(settings)
        workspace_key = self._workspace_key(self._workspace_store.workspace_root)
        if workspace_key:
            self._settings_by_workspace[workspace_key] = dict(settings)
            self._active_settings_workspace_key = workspace_key
        else:
            self._saved_global_settings = dict(settings)
        self._persist_editor_state()
        self.statusBar().showMessage("Saved prediction settings for current folder.", 4000)

    def _build_actions(self) -> None:
        act_open_image = QtGui.QAction("Open Image...", self)
        act_open_image.setShortcut(QtGui.QKeySequence.Open)
        act_open_image.triggered.connect(self._workspace_actions.open_image_dialog)
        self.addAction(act_open_image)

        act_open_folder = QtGui.QAction("Open Workspace Folder...", self)
        act_open_folder.setShortcut(QtGui.QKeySequence("Ctrl+Shift+O"))
        act_open_folder.triggered.connect(self._workspace_actions.open_folder)
        self.addAction(act_open_folder)

        act_add_folder_images = QtGui.QAction("Add Folder Images...", self)
        act_add_folder_images.setShortcut(QtGui.QKeySequence("Ctrl+Shift+I"))
        act_add_folder_images.triggered.connect(self._workspace_actions.add_folder_images)
        self.addAction(act_add_folder_images)

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
        file_menu.addAction(act_add_folder_images)
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
            "settings": dict(self._saved_global_settings),
            "settings_by_workspace": dict(self._settings_by_workspace),
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

    def _workspace_key(self, workspace_root: Path | str | None) -> str:
        if workspace_root is None:
            return ""
        try:
            return str(Path(workspace_root).resolve())
        except Exception:
            return str(workspace_root)

    def _migrate_settings_by_workspace(self, raw: object) -> dict[str, dict]:
        mapped: dict[str, dict] = {}
        if not isinstance(raw, dict):
            return mapped
        for key, value in raw.items():
            workspace_key = self._workspace_key(key)
            if not workspace_key or not isinstance(value, dict):
                continue
            settings = dict(DEFAULT_EDITOR_SETTINGS)
            settings.update(migrate_editor_settings(dict(value)))
            mapped[workspace_key] = settings
        return mapped

    def _settings_for_workspace(self, workspace_root: Path | None) -> dict:
        settings = dict(DEFAULT_EDITOR_SETTINGS)
        workspace_key = self._workspace_key(workspace_root)
        if workspace_key and workspace_key in self._settings_by_workspace:
            settings.update(self._settings_by_workspace[workspace_key])
            return settings
        settings.update(self._saved_global_settings)
        return settings

    def _apply_workspace_settings(self, workspace_root: Path | None, *, force: bool = False) -> None:
        workspace_key = self._workspace_key(workspace_root)
        if not force and workspace_key == self._active_settings_workspace_key:
            return
        self._active_settings_workspace_key = workspace_key
        self._ui_store.set_settings(self._settings_for_workspace(workspace_root))

    def _on_prediction_job_completed(self, job_id: str) -> None:
        job = self._prediction_store.get(job_id)
        if job is None or str(job.scope) != "folder" or not job.run_dir:
            return
        try:
            items = self._run_storage.list_result_items(Path(job.run_dir))
        except Exception as exc:
            self._show_error(str(exc))
            return
        if not items:
            self.statusBar().showMessage("Folder prediction completed, but no result items were produced.", 5000)
            return
        self._workspace_controller.set_result_items(items)
        item_paths = [str(item.get("image_path") or "").strip() for item in items]
        item_paths = [path for path in item_paths if path]
        target_path = None
        current_path = str(self._workspace_store.current_image_path or "").strip()
        if current_path and current_path in item_paths:
            target_path = current_path
        elif item_paths:
            target_path = item_paths[0]
        if target_path:
            try:
                self._workspace_controller.open_image(target_path)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._left_rail.explorer().set_images(list(self._workspace_store.images))
            self._left_rail.explorer().select_path(target_path)
        self._show_workspace("editor")
        self.statusBar().showMessage(f"Loaded {len(items)} result item(s) for folder run {job.run_id}.", 5000)
