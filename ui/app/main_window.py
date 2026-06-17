from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ui.app.detection_mixin import DetectionMixin
from ui.app.image_mixin import ImageMixin
from ui.app.results_mixin import ResultsMixin
from ui.app.segment_mixin import SegmentMixin
from ui.app.tool_mixin import ToolMixin
from ui.canvas.tools import TOOL_REGISTRY
from ui.core.commands import UndoStack
from ui.core.settings import SettingsIO, UiSettings
from ui.models.layer import DetectionGroup, LayerKind, LayerNode, LayerTree
from ui.panels.detect_panel import DetectPanel
from ui.panels.history_panel import HistoryPanel
from ui.panels.inspector_panel import InspectorPanel
from ui.panels.jobs_panel import JobsPanel
from ui.panels.layers_panel import LayersPanel
from ui.panels.segment_panel import SegmentPanel
from ui.panels.tools_palette import ToolsPalette
from ui.services.detect_process import DetectProcess, DetectionRow
from ui.services.inference_client import InferenceClient
from ui.services.job_manager import JobManager
from ui.widgets.canvas import ImageCanvas
from ui.widgets.log_panel import LogPanel


class MainWindow(SegmentMixin, ToolMixin, DetectionMixin, ResultsMixin, ImageMixin, QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._settings: UiSettings = SettingsIO.load()
        self._image_path: Path | None = None
        self._rows: list[DetectionRow] = []
        self._groups: list[DetectionGroup] = []
        self._active_group_id: str | None = None
        self._detect_counter = 0
        self._stabledino_checkpoint = self._settings.stabledino_checkpoint
        self._proc: DetectProcess | None = None
        self._active_processes: list[DetectProcess] = []
        self._proc_job_id: str | None = None

        # SegmentMixin state
        self._seg_job_id: str | None = None
        self._seg_infer_id: str | None = None
        self._seg_infer_to_job: dict[str, str] = {}
        self._seg_tmp_dir: str = ""
        self._seg_group_id: str | None = None

        self._undo = UndoStack(self)
        self._jobs = JobManager(self)
        self._infer = InferenceClient(self)
        self._layers = LayerTree()
        self._tools = {name: cls(self) for name, cls in TOOL_REGISTRY.items()}

        self.setWindowTitle(self._settings.app_name)
        self.resize(self._settings.default_width, self._settings.default_height)
        self._build_ui()
        self._build_actions()
        self._wire_signals()
        self._set_active_tool("pan")
        self._sync_state()

        self._apply_settings_to_panels()

        self.statusBar().showMessage("Open an image to start.")

    def _apply_settings_to_panels(self) -> None:
        self._segment_panel.sam_options.ckpt.setText(self._settings.sam_checkpoint)
        self._segment_panel.sam_options.model_type.setCurrentText(self._settings.sam_model_type)
        self._segment_panel.sam_lora_options.base_ckpt.setText(self._settings.sam_lora_base_checkpoint)
        self._segment_panel.sam_lora_options.lora_ckpt.setText(self._settings.sam_lora_checkpoint)
        self._segment_panel.sam_lora_options.lora_rank.setValue(int(self._settings.sam_lora_rank))
        self._segment_panel.sam_lora_options.refine_ckpt.setText(self._settings.sam_lora_refine_checkpoint)
        self._segment_panel.sam_lora_options.refine_rank.setValue(int(self._settings.sam_lora_refine_rank))
        mode_idx = self._segment_panel.sam_lora_options.predict_mode.findData(self._settings.sam_lora_predict_mode)
        if mode_idx >= 0:
            self._segment_panel.sam_lora_options.predict_mode.setCurrentIndex(mode_idx)
        self._segment_panel.unet_options.ckpt.setText(self._settings.unet_checkpoint)
        self._segment_panel.unet_options.threshold.setValue(float(self._settings.unet_threshold))

        self._detect_panel.yolo_options.checkpoint.setText(self._settings.yolo_checkpoint)
        self._detect_panel.yolo_options.conf.setValue(float(self._settings.yolo_conf))
        self._detect_panel.yolo_options.iou.setValue(float(self._settings.yolo_iou))
        self._detect_panel.yolo_options.imgsz.setValue(int(self._settings.yolo_imgsz))
        self._detect_panel.yolo_options.max_dets.setValue(int(self._settings.yolo_max_dets))
        self._stabledino_checkpoint = self._settings.stabledino_checkpoint

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for proc in list(self._active_processes):
            proc.cancel()
        if self._proc is not None and self._proc not in self._active_processes:
            self._proc.cancel()
        SettingsIO.save(self._settings)
        super().closeEvent(event)

    # ------------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        self._canvas = ImageCanvas(self)
        self._canvas.roisChanged.connect(self._on_rois_changed)
        self.setCentralWidget(self._canvas)

        self._toolbar = QtWidgets.QToolBar("Main", self)
        self._toolbar.setMovable(False)
        self._toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(self._toolbar)

        self._build_tools_dock()
        self._build_inspector_dock()
        self._build_workbench_dock()
        self._build_statusbar()
        self._append_log("Workspace ready.")

    def _build_tools_dock(self) -> None:
        self._tools_palette = ToolsPalette(self)
        tools_dock = QtWidgets.QDockWidget("Tools", self)
        tools_dock.setObjectName("ToolsDock")
        tools_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        tools_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        tools_dock.setWidget(self._tools_palette)
        tools_dock.setMinimumWidth(145)
        tools_dock.setMaximumWidth(220)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, tools_dock)
        self._tools_dock = tools_dock

    def _build_inspector_dock(self) -> None:
        self._inspector = InspectorPanel(self)
        self._detect_panel = DetectPanel(self)
        self._segment_panel = SegmentPanel(self)
        self._layers_panel = LayersPanel(self)
        self._layers_panel.set_tree(self._layers)

        right_area = QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        all_areas = QtCore.Qt.DockWidgetArea.AllDockWidgetAreas

        self._inspector_dock = self._make_dock("Inspector", "InspectorDock", self._inspector, right_area, all_areas)
        self._detect_dock = self._make_dock("Detect", "DetectDock", self._detect_panel, right_area, all_areas)
        self._segment_dock = self._make_dock("Segment", "SegmentDock", self._segment_panel, right_area, all_areas)
        self._layers_dock = self._make_dock("Layers", "LayersDock", self._layers_panel, right_area, all_areas)

        for d in (self._inspector_dock, self._detect_dock, self._segment_dock, self._layers_dock):
            d.setMinimumWidth(280)

        # Tab them together by default — each takes full vertical space; user
        # clicks the tab to focus. Still draggable: drop onto a side to undock.
        self.tabifyDockWidget(self._inspector_dock, self._detect_dock)
        self.tabifyDockWidget(self._detect_dock, self._segment_dock)
        self.tabifyDockWidget(self._segment_dock, self._layers_dock)

        self.setTabPosition(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
            QtWidgets.QTabWidget.TabPosition.North,
        )
        self._inspector_dock.raise_()

    def _make_dock(
        self,
        title: str,
        object_name: str,
        widget: QtWidgets.QWidget,
        initial_area: QtCore.Qt.DockWidgetArea,
        allowed_areas: QtCore.Qt.DockWidgetArea,
    ) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setAllowedAreas(allowed_areas)
        dock.setWidget(widget)
        self.addDockWidget(initial_area, dock)
        return dock

    def _build_workbench_dock(self) -> None:
        bottom_widget = QtWidgets.QTabWidget(self)
        bottom_widget.setDocumentMode(True)
        self._jobs_panel = JobsPanel(bottom_widget)
        self._log_panel = LogPanel(bottom_widget)
        self._history_panel = HistoryPanel(bottom_widget)
        self._history_panel.attach(self._undo)
        bottom_widget.addTab(self._jobs_panel, "Jobs")
        bottom_widget.addTab(self._log_panel, "Log")
        bottom_widget.addTab(self._history_panel, "History")

        bottom_dock = QtWidgets.QDockWidget("Workbench", self)
        bottom_dock.setObjectName("WorkbenchDock")
        bottom_dock.setWidget(bottom_widget)
        bottom_dock.setMinimumHeight(120)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, bottom_dock)
        self._workbench_dock = bottom_dock

    def _build_statusbar(self) -> None:
        self._sb_counts = QtWidgets.QLabel("ROI: 0   Boxes: 0", self)
        self._sb_zoom = QtWidgets.QLabel("100%", self)
        self._sb_cursor = QtWidgets.QLabel("x: -, y: -", self)
        self._sb_tool = QtWidgets.QLabel("Tool: Pan", self)
        self._counts = self._sb_counts  # compat alias for detection_mixin
        self._progress = QtWidgets.QProgressBar(self)
        self._progress.setMaximumWidth(220)
        self._progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._sb_zoom)
        self.statusBar().addPermanentWidget(self._sb_cursor)
        self.statusBar().addPermanentWidget(self._sb_tool)
        self.statusBar().addPermanentWidget(self._sb_counts)
        self.statusBar().addPermanentWidget(self._progress)

    # ------------------------------------------------------------------ actions

    def _build_actions(self) -> None:
        style = self.style()

        self._act_open = QtGui.QAction(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton), "Open Image…", self
        )
        self._act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self._act_open.triggered.connect(self._open_image)

        self._act_fit = QtGui.QAction("Fit to Window", self)
        self._act_fit.setShortcut("F")
        self._act_fit.triggered.connect(self._canvas.fit_image)

        self._act_zoom_in = QtGui.QAction("Zoom In", self)
        self._act_zoom_in.setShortcut(QtGui.QKeySequence.StandardKey.ZoomIn)
        self._act_zoom_in.triggered.connect(lambda: self._canvas._zoom_by(1.25))

        self._act_zoom_out = QtGui.QAction("Zoom Out", self)
        self._act_zoom_out.setShortcut(QtGui.QKeySequence.StandardKey.ZoomOut)
        self._act_zoom_out.triggered.connect(lambda: self._canvas._zoom_by(0.8))

        self._act_actual = QtGui.QAction("Actual Size", self)
        self._act_actual.setShortcut("Ctrl+1")
        self._act_actual.triggered.connect(self._actual_size)

        self._act_undo = QtGui.QAction("Undo", self)
        self._act_undo.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self._act_undo.triggered.connect(self._undo.undo)

        self._act_redo = QtGui.QAction("Redo", self)
        self._act_redo.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        self._act_redo.triggered.connect(self._undo.redo)

        self._act_run = QtGui.QAction(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay), "Run Detection", self
        )
        self._act_run.setShortcut("F5")
        self._act_run.triggered.connect(self._run_detection)

        self._act_cancel = QtGui.QAction(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop), "Cancel", self
        )
        self._act_cancel.setEnabled(False)
        self._act_cancel.triggered.connect(self._cancel_detection)

        self._act_save = QtGui.QAction(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton), "Export Results…", self
        )
        self._act_save.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self._act_save.setEnabled(False)
        self._act_save.triggered.connect(self._save_results)

        self._act_delete = QtGui.QAction("Delete Selected ROI", self)
        self._act_delete.setShortcut(QtGui.QKeySequence.StandardKey.Delete)
        self._act_delete.triggered.connect(self._delete_selected_rois)

        self._act_clear_rois = QtGui.QAction("Clear All ROIs", self)
        self._act_clear_rois.triggered.connect(self._clear_all_rois)

        self._act_clear_masks = QtGui.QAction("Clear Masks", self)
        self._act_clear_masks.triggered.connect(self._canvas.clear_masks)

        self._act_preferences = QtGui.QAction("Preferences…", self)
        self._act_preferences.setShortcut("Ctrl+,")
        self._act_preferences.triggered.connect(self._open_preferences)

        self._populate_toolbar()
        self._populate_menus()

        # Tool keyboard shortcuts (global, work even when canvas not focused)
        self._add_tool_shortcut("H", "pan")
        self._add_tool_shortcut("V", "select")
        self._add_tool_shortcut("R", "rect_roi")

    def _add_tool_shortcut(self, key: str, tool_name: str) -> None:
        sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
        sc.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(lambda tn=tool_name: self._set_active_tool(tn))

    def _populate_toolbar(self) -> None:
        for action in (self._act_open, self._act_fit):
            self._toolbar.addAction(action)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._act_undo)
        self._toolbar.addAction(self._act_redo)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._act_run)
        self._toolbar.addAction(self._act_cancel)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._act_save)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._act_delete)

    def _populate_menus(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("File")
        file_menu.addAction(self._act_open)
        file_menu.addSeparator()
        file_menu.addAction(self._act_save)
        file_menu.addSeparator()
        file_menu.addAction(self._act_preferences)
        act_quit = QtGui.QAction("Quit", self)
        act_quit.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        # Edit
        edit_menu = mb.addMenu("Edit")
        edit_menu.addAction(self._act_undo)
        edit_menu.addAction(self._act_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self._act_delete)
        edit_menu.addAction(self._act_clear_rois)

        # View
        view_menu = mb.addMenu("View")
        view_menu.addAction(self._act_fit)
        view_menu.addAction(self._act_actual)
        view_menu.addAction(self._act_zoom_in)
        view_menu.addAction(self._act_zoom_out)
        view_menu.addSeparator()
        act_toggle_tools = self._tools_dock.toggleViewAction()
        act_toggle_tools.setShortcut("F7")
        act_toggle_inspector = self._inspector_dock.toggleViewAction()
        act_toggle_inspector.setShortcut("F8")
        act_toggle_detect = self._detect_dock.toggleViewAction()
        act_toggle_detect.setShortcut("F2")
        act_toggle_segment = self._segment_dock.toggleViewAction()
        act_toggle_segment.setShortcut("F3")
        act_toggle_layers = self._layers_dock.toggleViewAction()
        act_toggle_layers.setShortcut("F4")
        act_toggle_wb = self._workbench_dock.toggleViewAction()
        act_toggle_wb.setShortcut("F9")
        view_menu.addAction(act_toggle_tools)
        view_menu.addAction(act_toggle_inspector)
        view_menu.addAction(act_toggle_detect)
        view_menu.addAction(act_toggle_segment)
        view_menu.addAction(act_toggle_layers)
        view_menu.addAction(act_toggle_wb)

        # Detect
        detect_menu = mb.addMenu("Detect")
        detect_menu.addAction(self._act_run)
        detect_menu.addAction(self._act_cancel)
        detect_menu.addSeparator()
        act_sd = QtGui.QAction("Set StableDINO checkpoint…", self)
        act_sd.triggered.connect(self._set_stabledino_checkpoint)
        detect_menu.addAction(act_sd)

        # Segment
        seg_menu = mb.addMenu("Segment")
        act_run_seg = QtGui.QAction("Run Segmentation", self)
        act_run_seg.setShortcut("Shift+F5")
        act_run_seg.triggered.connect(self._run_segment)
        seg_menu.addAction(act_run_seg)
        seg_menu.addAction(self._act_clear_masks)

        # Layer
        layer_menu = mb.addMenu("Layer")
        layer_menu.addAction(self._act_clear_masks)
        layer_menu.addSeparator()
        act_toggle_det = QtGui.QAction("Toggle Detections", self)
        act_toggle_det.triggered.connect(self._toggle_detections_layer)
        layer_menu.addAction(act_toggle_det)

        # Window
        win_menu = mb.addMenu("Window")
        win_menu.addAction(act_toggle_tools)
        win_menu.addAction(act_toggle_inspector)
        win_menu.addAction(act_toggle_detect)
        win_menu.addAction(act_toggle_segment)
        win_menu.addAction(act_toggle_layers)
        win_menu.addAction(act_toggle_wb)

    def _open_preferences(self) -> None:
        from ui.dialogs.settings_dialog import PreferencesDialog
        dlg = PreferencesDialog(self._settings, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            SettingsIO.save(self._settings)
            self._segment_panel.sam_ckpt.setText(self._settings.sam_checkpoint)
            self.statusBar().showMessage("Preferences saved.")

    def _actual_size(self) -> None:
        self._canvas.resetTransform()
        self._canvas.zoomChanged.emit(1.0)

    def _clear_all_rois(self) -> None:
        self._canvas.clear_rois()

    def _toggle_detections_layer(self) -> None:
        group = self._active_group()
        if group is None:
            return
        layer = self._layers.by_id(group.layer_id)
        if layer is not None:
            layer.visible = not layer.visible
            self._on_layer_visibility(layer.id, layer.visible)
            self._refresh_layers_panel()

    # ------------------------------------------------------------------ detection groups

    def _active_group(self) -> DetectionGroup | None:
        return self._group_by_id(self._active_group_id)

    def _group_by_id(self, layer_id: str | None) -> DetectionGroup | None:
        if layer_id is None:
            return None
        for group in self._groups:
            if group.layer_id == layer_id:
                return group
        return None

    def _add_detection_group(self, detector: str, rows: list[DetectionRow]) -> DetectionGroup:
        self._detect_counter += 1
        node = LayerNode(
            kind=LayerKind.detections,
            name=f"Detect {detector} #{self._detect_counter}",
            z_order=20 + self._detect_counter,
        )
        self._layers.add_layer(node)
        group = DetectionGroup(layer_id=node.id, name=node.name, detector=str(detector), rows=list(rows))
        self._groups.append(group)
        self._set_active_group(node.id)
        return group

    def _set_active_group(self, layer_id: str | None) -> None:
        self._active_group_id = layer_id
        group = self._group_by_id(layer_id)
        self._rows = list(group.rows) if group is not None else []
        self._layers_panel.set_active_layer(layer_id)

    def _render_groups(self) -> None:
        # Attach the layer's current visibility to each group for the canvas.
        for group in self._groups:
            layer = self._layers.by_id(group.layer_id)
            setattr(group, "visible", bool(layer.visible) if layer is not None else True)
        self._canvas.render_detection_groups(self._groups, self._active_group_id)

    def _refresh_layers_panel(self) -> None:
        self._layers_panel.set_tree(self._layers)
        self._layers_panel.set_active_layer(self._active_group_id)

    def _on_layer_selected(self, layer_id: str) -> None:
        if self._group_by_id(layer_id) is None:
            return
        self._set_active_group(layer_id)
        self._render_groups()
        self._refresh_results()

    def _reset_detection_groups(self) -> None:
        for layer in self._layers.detection_layers():
            self._layers.remove_layer(layer.id)
        self._groups.clear()
        self._active_group_id = None
        self._rows = []
        self._detect_counter = 0
        self._refresh_layers_panel()

    # ------------------------------------------------------------------ signals

    def _wire_signals(self) -> None:
        self._wire_tool_signals()
        self._wire_detection_signals()
        self._wire_segment_signals()
        self._layers_panel.visibilityChanged.connect(self._on_layer_visibility)
        self._layers_panel.opacityChanged.connect(self._on_layer_opacity)
        self._layers_panel.layerSelected.connect(self._on_layer_selected)
        self._canvas.cursorMoved.connect(self._on_cursor_moved)
        self._canvas.zoomChanged.connect(self._on_zoom_changed)
        self._undo.cursorChanged.connect(self._on_undo_cursor)

    def _sync_state(self) -> None:
        roi_count = len(self._canvas.roi_items())
        shown_count = len(self._filtered_rows())
        self._sb_counts.setText(f"ROI: {roi_count}   Boxes: {len(self._rows)}   Shown: {shown_count}")
        self._act_save.setEnabled(bool(self._rows) and self._proc is None)
        self._act_undo.setEnabled(self._undo.can_undo())
        self._act_redo.setEnabled(self._undo.can_redo())

    def _on_cursor_moved(self, x: float, y: float) -> None:
        self._sb_cursor.setText(f"x: {int(x)}, y: {int(y)}")

    def _on_zoom_changed(self, factor: float) -> None:
        self._sb_zoom.setText(f"{int(factor * 100)}%")

    def _on_undo_cursor(self, _cursor: int) -> None:
        self._act_undo.setEnabled(self._undo.can_undo())
        self._act_redo.setEnabled(self._undo.can_redo())

    def _append_log(self, message: str) -> None:
        text = str(message)
        self._log_panel.appendPlainText(text)
        self.statusBar().showMessage(text)

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self._act_open,
            self._act_fit,
            self._act_run,
            self._act_save,
            self._detect_panel.detector_combo,
            self._detect_panel.run_conf,
            self._detect_panel.run_button,
        ):
            widget.setEnabled(not busy)
        self._detect_panel.cancel_button.setEnabled(busy)
        self._act_cancel.setEnabled(busy)
        if not busy:
            self._act_save.setEnabled(bool(self._rows))
