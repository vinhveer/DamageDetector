from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from editor_app.canvas import ImageCanvas
from editor_app.ui.components.image_tools_panel import ImageToolsPanel


class EditorInspectPanel(QtWidgets.QWidget):
    addRoiRequested = QtCore.Signal()
    detectionSelectionChanged = QtCore.Signal(object)
    roiSelectionChanged = QtCore.Signal(int)
    editSelectedRoiRequested = QtCore.Signal()
    roiDeleted = QtCore.Signal(int)
    roiCleared = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_roi_index = -1
        self._highlighted: list[dict] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        actions_row = QtWidgets.QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(8)
        self._roi_add_button = QtWidgets.QPushButton("Add ROI", self)
        self._roi_add_button.clicked.connect(self.addRoiRequested.emit)
        actions_row.addWidget(self._roi_add_button)
        actions_row.addStretch(1)
        root.addLayout(actions_row)

        root.addWidget(QtWidgets.QLabel("ROI List", self))
        self._roi_list = QtWidgets.QTreeWidget(self)
        self._roi_list.setHeaderLabels(["ROI", "Bounds"])
        self._roi_list.setRootIsDecorated(False)
        self._roi_list.setUniformRowHeights(True)
        self._roi_list.itemSelectionChanged.connect(self._emit_roi_selection_changed)
        self._roi_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._roi_list.customContextMenuRequested.connect(self._open_roi_context_menu)
        root.addWidget(self._roi_list)

        root.addWidget(QtWidgets.QLabel("Detections", self))
        self._detection_list = QtWidgets.QListWidget(self)
        self._detection_list.itemSelectionChanged.connect(self._emit_selection_highlight)
        root.addWidget(self._detection_list, 1)

    def set_detections(self, detections: list[dict]) -> None:
        self._highlighted = [dict(det) for det in detections]
        self._detection_list.clear()
        for det in detections:
            label = str(det.get("label") or "object")
            score = det.get("score")
            model_name = str(det.get("model_name") or "")
            text = f"{label} | {score:.3f}" if isinstance(score, (int, float)) else label
            if model_name:
                text = f"{text} | {model_name}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(det))
            self._detection_list.addItem(item)

    def set_roi_boxes(self, roi_boxes: list[tuple[int, int, int, int]], current_index: int) -> None:
        self._current_roi_index = int(current_index)
        blocker = QtCore.QSignalBlocker(self._roi_list)
        self._roi_list.clear()
        for idx, roi in enumerate(roi_boxes):
            item = QtWidgets.QTreeWidgetItem([f"ROI {idx + 1}", f"{roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}"])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, idx)
            self._roi_list.addTopLevelItem(item)
            if idx == self._current_roi_index:
                self._roi_list.setCurrentItem(item)
        del blocker

    def _emit_selection_highlight(self) -> None:
        selected = []
        for item in self._detection_list.selectedItems():
            det = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(det, dict):
                selected.append(det)
        self.detectionSelectionChanged.emit(selected or list(self._highlighted))

    def _emit_roi_selection_changed(self) -> None:
        item = self._roi_list.currentItem()
        index = -1
        if item is not None:
            value = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if value is not None:
                index = int(value)
        self._current_roi_index = index
        self.roiSelectionChanged.emit(index)

    def _open_roi_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        item = self._roi_list.itemAt(pos)
        if item is not None:
            self._roi_list.setCurrentItem(item)
            edit_action = menu.addAction("Edit ROI")
            delete_action = menu.addAction("Delete ROI")
            menu.addSeparator()
            clear_action = menu.addAction("Clear All ROI")
            action = menu.exec(self._roi_list.viewport().mapToGlobal(pos))
            if action == edit_action:
                self.editSelectedRoiRequested.emit()
            elif action == delete_action and self._current_roi_index >= 0:
                self.roiDeleted.emit(self._current_roi_index)
            elif action == clear_action:
                self.roiCleared.emit()
            return
        if self._roi_list.topLevelItemCount() > 0:
            clear_action = menu.addAction("Clear All ROI")
            action = menu.exec(self._roi_list.viewport().mapToGlobal(pos))
            if action == clear_action:
                self.roiCleared.emit()


class EditorToolsPanel(QtWidgets.QWidget):
    def __init__(self, overlay_canvas: ImageCanvas, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        root.addWidget(QtWidgets.QLabel("Mask Edit Tools", self))
        root.addWidget(ImageToolsPanel(overlay_canvas, self), 1)


class EditorWorkspace(QtWidgets.QWidget):
    roiBoxSelected = QtCore.Signal(object)
    roiSelectionCanceled = QtCore.Signal()
    roiAdded = QtCore.Signal(object)
    roiUpdated = QtCore.Signal(int, object)
    roiDeleted = QtCore.Signal(int)
    roiCleared = QtCore.Signal()
    roiSelectionChanged = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._overlay_canvas = ImageCanvas(self, render_mode="overlay", editable=True)
        self._image_canvas = ImageCanvas(self, render_mode="image", editable=False)
        self._mask_canvas = ImageCanvas(self, render_mode="mask", editable=False)
        self._overlay_canvas.set_editable(False)
        self._overlay_canvas.maskChanged.connect(self._sync_mask_views)

        self._highlighted: list[dict] = []
        self._roi_boxes: list[tuple[int, int, int, int]] = []
        self._current_roi_index: int = -1
        self._roi_capture_mode: str | None = None
        self._edit_mode = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._view_tabs = QtWidgets.QTabWidget(self)
        self._view_tabs.addTab(self._overlay_canvas, "Overlay")
        self._view_tabs.addTab(self._image_canvas, "Image")
        self._view_tabs.addTab(self._mask_canvas, "Mask")
        root.addWidget(self._view_tabs, 1)

        self._inspect_panel = EditorInspectPanel()
        self._inspect_panel.addRoiRequested.connect(self.start_roi_selection)
        self._inspect_panel.detectionSelectionChanged.connect(self._on_inspect_detection_selection_changed)
        self._inspect_panel.roiSelectionChanged.connect(self._on_sidebar_roi_selection_changed)
        self._inspect_panel.editSelectedRoiRequested.connect(self._start_edit_selected_roi)
        self._inspect_panel.roiDeleted.connect(self.roiDeleted.emit)
        self._inspect_panel.roiCleared.connect(self.roiCleared.emit)

        self._tools_panel = EditorToolsPanel(self._overlay_canvas)

        self._overlay_canvas.roiCanceled.connect(self._on_roi_canceled)
        self._overlay_canvas.roiSelected.connect(self._on_roi_selected)

    def inspect_panel(self) -> EditorInspectPanel:
        return self._inspect_panel

    def tools_panel(self) -> EditorToolsPanel:
        return self._tools_panel

    def set_left_rail_editor_active(self, active: bool) -> None:
        enabled = bool(active)
        if enabled == self._edit_mode:
            return
        self._edit_mode = enabled
        self._overlay_canvas.set_editable(enabled)
        if enabled:
            self._view_tabs.setCurrentWidget(self._overlay_canvas)

    def set_image(self, image: QtGui.QImage, path: str | None = None) -> None:
        self._overlay_canvas.set_image(image)
        self._image_canvas.set_image(image)
        self._sync_mask_views()
        self._sync_roi_boxes_to_canvas()

    def set_mask(self, mask: QtGui.QImage) -> None:
        self._overlay_canvas.set_mask(mask)
        self._sync_mask_views()

    def set_detections(self, detections: list[dict]) -> None:
        self._inspect_panel.set_detections(detections)

    def set_roi_boxes(self, roi_boxes: list[tuple[int, int, int, int]], current_index: int) -> None:
        self._roi_boxes = [tuple(int(x) for x in roi) for roi in roi_boxes]
        self._current_roi_index = int(current_index)
        self._inspect_panel.set_roi_boxes(self._roi_boxes, self._current_roi_index)
        self._sync_roi_boxes_to_canvas()

    def set_highlight_detections(self, detections: list[dict]) -> None:
        self._highlighted = [dict(det) for det in detections]
        boxes = []
        for det in self._highlighted:
            box = det.get("box")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            rect = QtCore.QRectF(float(box[0]), float(box[1]), float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
            boxes.append((rect, str(det.get("label") or "")))
        self._overlay_canvas.set_highlight_boxes(boxes)
        self._image_canvas.set_highlight_boxes(boxes)
        self._mask_canvas.set_highlight_boxes(boxes)

    def clear_highlights(self) -> None:
        self.set_highlight_detections([])

    def current_roi_box(self) -> tuple[int, int, int, int] | None:
        if self._current_roi_index < 0 or self._current_roi_index >= len(self._roi_boxes):
            return None
        return tuple(int(x) for x in self._roi_boxes[self._current_roi_index])

    def start_roi_selection(self) -> None:
        self._roi_capture_mode = "add"
        self.set_left_rail_editor_active(False)
        self._view_tabs.setCurrentWidget(self._overlay_canvas)
        self._overlay_canvas.start_roi_selection()

    def start_prediction_roi_selection(self) -> None:
        self._roi_capture_mode = "predict"
        self.set_left_rail_editor_active(False)
        self._view_tabs.setCurrentWidget(self._overlay_canvas)
        self._overlay_canvas.start_roi_selection()

    def overlay_canvas(self) -> ImageCanvas:
        return self._overlay_canvas

    def image_canvas(self) -> ImageCanvas:
        return self._image_canvas

    def mask_canvas(self) -> ImageCanvas:
        return self._mask_canvas

    def _start_edit_selected_roi(self) -> None:
        if self.current_roi_box() is None:
            return
        self._roi_capture_mode = "edit"
        self.set_left_rail_editor_active(False)
        self._view_tabs.setCurrentWidget(self._overlay_canvas)
        self._overlay_canvas.start_roi_selection()

    def _sync_mask_views(self) -> None:
        mask = self._overlay_canvas.mask()
        self._mask_canvas.set_mask(mask)

    def _on_inspect_detection_selection_changed(self, detections_obj) -> None:
        detections = [dict(det) for det in (detections_obj or []) if isinstance(det, dict)]
        self.set_highlight_detections(detections or self._highlighted)

    def _on_sidebar_roi_selection_changed(self, index: int) -> None:
        self._current_roi_index = int(index)
        self._sync_roi_boxes_to_canvas()
        self.roiSelectionChanged.emit(index)

    def _on_roi_selected(self, roi_box) -> None:
        mode = str(self._roi_capture_mode or "")
        self._roi_capture_mode = None
        if roi_box is None:
            if mode == "predict":
                self.roiBoxSelected.emit(None)
            return
        if mode == "predict":
            self.roiBoxSelected.emit(roi_box)
        elif mode == "edit":
            if self._current_roi_index >= 0:
                self.roiUpdated.emit(self._current_roi_index, roi_box)
        else:
            self.roiAdded.emit(roi_box)

    def _on_roi_canceled(self) -> None:
        mode = str(self._roi_capture_mode or "")
        self._roi_capture_mode = None
        if mode == "predict":
            self.roiSelectionCanceled.emit()

    def _sync_roi_boxes_to_canvas(self) -> None:
        boxes = []
        for idx, roi in enumerate(self._roi_boxes):
            rect = QtCore.QRectF(float(roi[0]), float(roi[1]), float(roi[2] - roi[0]), float(roi[3] - roi[1]))
            boxes.append((rect, f"ROI {idx + 1}", idx == self._current_roi_index))
        self._overlay_canvas.set_roi_boxes(boxes)
        self._image_canvas.set_roi_boxes(boxes)
        self._mask_canvas.set_roi_boxes(boxes)
