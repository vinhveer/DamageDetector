from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6 import QtCore, QtGui, QtWidgets

from pineline.label_app.io_utils import BoxData, load_boxes, save_boxes

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Class label -> color, mirrors roi_detect_app palette.
CLASS_COLORS: dict[str, QtGui.QColor] = {
    "crack": QtGui.QColor(55, 150, 255),
    "mold": QtGui.QColor(52, 211, 153),
    "stain": QtGui.QColor(245, 158, 11),
    "spall": QtGui.QColor(248, 113, 113),
}
DEFAULT_LABELS = ["crack", "mold", "stain", "spall"]


def color_for_label(label: str) -> QtGui.QColor:
    return CLASS_COLORS.get(label, QtGui.QColor(255, 198, 41))


HANDLE_SIZE = 9.0  # screen pixels (handles ignore view transform)
MIN_BOX = 4.0      # minimum box size in scene/image pixels


class _Handle(QtWidgets.QGraphicsRectItem):
    """A small corner grip. Kept at constant screen size via ItemIgnoresTransformations."""

    def __init__(self, corner: int, parent: "EditableBoxItem") -> None:
        super().__init__(parent)
        self._corner = int(corner)  # 0=TL 1=TR 2=BR 3=BL
        self._owner = parent
        s = HANDLE_SIZE
        self.setRect(-s / 2, -s / 2, s, s)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(10)
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        self.setPen(QtGui.QPen(QtGui.QColor(30, 30, 30), 1))
        cursors = {
            0: QtCore.Qt.CursorShape.SizeFDiagCursor,
            1: QtCore.Qt.CursorShape.SizeBDiagCursor,
            2: QtCore.Qt.CursorShape.SizeFDiagCursor,
            3: QtCore.Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursors[self._corner])

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._owner.begin_resize(self._corner)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._owner.resize_to(self._corner, event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._owner.end_resize()
        event.accept()


class EditableBoxItem(QtWidgets.QGraphicsRectItem):
    """A movable / 4-corner-resizable labelled box in image (scene) coordinates."""

    def __init__(self, data: BoxData) -> None:
        x1, y1, x2, y2 = data.normalized().box
        super().__init__(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
        self.label = data.label
        self.score = float(data.score)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(2)
        self._handles = [_Handle(i, self) for i in range(4)]
        self._sync_handles()
        self._apply_style(False)

    # --- geometry helpers -------------------------------------------------
    def scene_rect(self) -> QtCore.QRectF:
        return self.mapRectToScene(self.rect()).normalized()

    def to_box_data(self) -> BoxData:
        r = self.scene_rect()
        return BoxData(box=[r.left(), r.top(), r.right(), r.bottom()],
                       label=self.label, score=self.score)

    def set_label(self, label: str) -> None:
        self.label = str(label)
        self._apply_style(self.isSelected())
        self.update()

    def _corner_points(self) -> list[QtCore.QPointF]:
        r = self.rect()
        return [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]

    def _sync_handles(self) -> None:
        pts = self._corner_points()
        for h, p in zip(self._handles, pts):
            h.setPos(p)

    def set_handles_visible(self, visible: bool) -> None:
        for h in self._handles:
            h.setVisible(visible)

    # --- resize driven by handles ----------------------------------------
    def begin_resize(self, corner: int) -> None:
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def resize_to(self, corner: int, scene_pos: QtCore.QPointF) -> None:
        local = self.mapFromScene(scene_pos)
        r = self.rect()
        left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()
        if corner == 0:      # top-left
            left, top = local.x(), local.y()
        elif corner == 1:    # top-right
            right, top = local.x(), local.y()
        elif corner == 2:    # bottom-right
            right, bottom = local.x(), local.y()
        elif corner == 3:    # bottom-left
            left, bottom = local.x(), local.y()
        new = QtCore.QRectF(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom)).normalized()
        if new.width() < MIN_BOX:
            new.setWidth(MIN_BOX)
        if new.height() < MIN_BOX:
            new.setHeight(MIN_BOX)
        self.prepareGeometryChange()
        self.setRect(new)
        self._sync_handles()
        self._notify_changed()

    def end_resize(self) -> None:
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self._notify_changed()

    def _notify_changed(self) -> None:
        scene = self.scene()
        view = scene.views()[0] if scene and scene.views() else None
        if isinstance(view, ImageCanvas):
            view.boxesChanged.emit()

    # --- styling / painting ----------------------------------------------
    def _apply_style(self, selected: bool) -> None:
        base = color_for_label(self.label)
        color = base.lighter(125) if selected else base
        pen = QtGui.QPen(color, 3 if selected else 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(QtGui.QColor(base.red(), base.green(), base.blue(),
                                                55 if selected else 30)))

    def itemChange(self, change, value):  # noqa: ANN001
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            sel = bool(value)
            self._apply_style(sel)
            self.set_handles_visible(sel)
        elif change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._notify_changed()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None) -> None:  # noqa: ANN001
        super().paint(painter, option, widget)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        font = painter.font()
        font.setBold(True)
        font.setPointSizeF(max(9.0, self.rect().height() * 0.07))
        painter.setFont(font)
        tag = self.label or "?"
        if self.score < 0.999:
            tag = f"{tag} {self.score:.2f}"
        base = color_for_label(self.label)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(tag) + 10.0
        th = metrics.height() + 4.0
        top_left = self.rect().topLeft()
        painter.fillRect(QtCore.QRectF(top_left.x(), top_left.y(), tw, th),
                         QtGui.QColor(base.red(), base.green(), base.blue(), 220))
        painter.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20)))
        painter.drawText(QtCore.QPointF(top_left.x() + 5.0, top_left.y() + metrics.ascent() + 2.0), tag)
        painter.restore()


class ImageCanvas(QtWidgets.QGraphicsView):
    boxesChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._draw_mode = False
        self._drag_start: QtCore.QPointF | None = None
        self._rubber: QtWidgets.QGraphicsRectItem | None = None
        self._new_label = DEFAULT_LABELS[0]

    # --- mode / loading ---------------------------------------------------
    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = bool(enabled)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag if enabled
                         else QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor)

    def set_new_label(self, label: str) -> None:
        self._new_label = str(label) or DEFAULT_LABELS[0]

    def load(self, image_path: Path, boxes: list[BoxData]) -> None:
        pixmap = QtGui.QPixmap(str(image_path))
        if pixmap.isNull():
            raise RuntimeError(f"Cannot load image: {image_path}")
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        for data in boxes:
            self._scene.addItem(EditableBoxItem(data))
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._hide_all_handles()
        self.boxesChanged.emit()

    def clear(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self.boxesChanged.emit()

    def image_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF() if self._pixmap_item is None else self._pixmap_item.boundingRect()

    # --- box access -------------------------------------------------------
    def box_items(self) -> list[EditableBoxItem]:
        return [it for it in self._scene.items() if isinstance(it, EditableBoxItem)]

    def boxes(self) -> list[BoxData]:
        bounds = self.image_rect()
        out: list[BoxData] = []
        for it in self.box_items():
            r = it.scene_rect()
            if not bounds.isNull():
                r = r.intersected(bounds)
            if r.width() >= MIN_BOX and r.height() >= MIN_BOX:
                out.append(BoxData(box=[r.left(), r.top(), r.right(), r.bottom()],
                                   label=it.label, score=it.score))
        return out

    def _hide_all_handles(self) -> None:
        for it in self.box_items():
            it.set_handles_visible(False)

    def add_box_at(self, scene_rect: QtCore.QRectF) -> None:
        rect = scene_rect.normalized().intersected(self.image_rect())
        if rect.width() < MIN_BOX or rect.height() < MIN_BOX:
            return
        item = EditableBoxItem(BoxData(box=[rect.left(), rect.top(), rect.right(), rect.bottom()],
                                       label=self._new_label, score=1.0))
        self._scene.addItem(item)
        self._scene.clearSelection()
        item.setSelected(True)
        self.boxesChanged.emit()

    def delete_selected(self) -> int:
        removed = 0
        for it in self.box_items():
            if it.isSelected():
                self._scene.removeItem(it)
                removed += 1
        if removed:
            self.boxesChanged.emit()
        return removed

    def selected_items(self) -> list[EditableBoxItem]:
        return [it for it in self.box_items() if it.isSelected()]

    def select_index(self, index: int) -> None:
        items = self.box_items()
        self._scene.clearSelection()
        if 0 <= index < len(items):
            items[index].setSelected(True)
            self.centerOn(items[index])

    def set_label_for_selected(self, label: str) -> None:
        changed = False
        for it in self.selected_items():
            it.set_label(label)
            changed = True
        if changed:
            self.boxesChanged.emit()

    def fit_image(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    # --- zoom (mouse wheel + trackpad pinch), copied behaviour ------------
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        modifiers = event.modifiers()
        zoom_mod = bool(modifiers & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.MetaModifier))
        is_trackpad_scroll = not event.pixelDelta().isNull()
        if is_trackpad_scroll and not zoom_mod:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta == 0:
            super().wheelEvent(event)
            return
        self._zoom_by(1.0015 ** delta, event.position())
        event.accept()

    def _zoom_by(self, factor: float, view_pos: QtCore.QPointF | None = None) -> None:
        factor = max(0.2, min(5.0, float(factor)))
        current = self.transform().m11()
        target = current * factor
        if target < 0.02:
            factor = 0.02 / current
        elif target > 60.0:
            factor = 60.0 / current
        if view_pos is not None:
            old_scene = self.mapToScene(view_pos.toPoint())
        self.scale(factor, factor)
        if view_pos is not None:
            new_view = self.mapFromScene(old_scene)
            delta = new_view - view_pos.toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + int(delta.y()))

    def viewportEvent(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.NativeGesture and isinstance(event, QtGui.QNativeGestureEvent):
            if event.gestureType() == QtCore.Qt.NativeGestureType.ZoomNativeGesture:
                self._zoom_by(1.0 + float(event.value()), event.position())
                return True
            if event.gestureType() == QtCore.Qt.NativeGestureType.SmartZoomNativeGesture:
                self.fit_image()
                return True
        return super().viewportEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            if self.delete_selected():
                event.accept()
                return
        super().keyPressEvent(event)

    # --- draw new box -----------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._draw_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_start = self.mapToScene(event.position().toPoint())
            pen = QtGui.QPen(color_for_label(self._new_label), 2, QtCore.Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._rubber = self._scene.addRect(QtCore.QRectF(self._drag_start, self._drag_start), pen)
            self._rubber.setZValue(5)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._draw_mode and self._drag_start is not None and self._rubber is not None:
            end = self.mapToScene(event.position().toPoint())
            rect = QtCore.QRectF(self._drag_start, end).normalized().intersected(self.image_rect())
            self._rubber.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._draw_mode and event.button() == QtCore.Qt.MouseButton.LeftButton and self._rubber is not None:
            rect = self._rubber.rect().normalized()
            self._scene.removeItem(self._rubber)
            self._rubber = None
            self._drag_start = None
            self.add_box_at(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)


@dataclass
class ImageEntry:
    image: Path
    json: Path

    @property
    def jsonm(self) -> Path:
        return self.json.with_suffix(self.json.suffix + "m")

    @property
    def saved(self) -> bool:
        return self.jsonm.exists()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Box Label Editor")
        self.resize(1500, 900)
        self._entries: list[ImageEntry] = []
        self._current: int = -1
        self._dirty = False
        self._build_ui()
        self._build_actions()
        self._refresh_box_list()

    # --- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        self._canvas = ImageCanvas(self)
        self.setCentralWidget(self._canvas)
        self._canvas.boxesChanged.connect(self._on_boxes_changed)
        self._canvas._scene.selectionChanged.connect(self._on_scene_selection)

        self._toolbar = QtWidgets.QToolBar("Main", self)
        self._toolbar.setMovable(False)
        self._toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(self._toolbar)

        # Left dock: image list
        files_dock = QtWidgets.QDockWidget("Images", self)
        files_panel = QtWidgets.QWidget(files_dock)
        files_layout = QtWidgets.QVBoxLayout(files_panel)
        files_layout.setContentsMargins(6, 6, 6, 6)
        self._file_list = QtWidgets.QListWidget(files_panel)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        files_layout.addWidget(QtWidgets.QLabel("Image / JSON pairs", files_panel))
        files_layout.addWidget(self._file_list, 1)
        files_dock.setWidget(files_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, files_dock)

        # Right dock: boxes
        box_dock = QtWidgets.QDockWidget("Boxes", self)
        box_panel = QtWidgets.QWidget(box_dock)
        box_layout = QtWidgets.QVBoxLayout(box_panel)
        box_layout.setContentsMargins(6, 6, 6, 6)

        form = QtWidgets.QFormLayout()
        self._label_combo = QtWidgets.QComboBox(box_panel)
        self._label_combo.setEditable(True)
        self._label_combo.addItems(DEFAULT_LABELS)
        self._label_combo.currentTextChanged.connect(self._canvas.set_new_label)
        form.addRow("New box label", self._label_combo)
        box_layout.addLayout(form)

        self._box_list = QtWidgets.QListWidget(box_panel)
        self._box_list.currentRowChanged.connect(self._on_box_row_selected)
        box_layout.addWidget(QtWidgets.QLabel("Boxes", box_panel))
        box_layout.addWidget(self._box_list, 1)

        btns = QtWidgets.QHBoxLayout()
        self._btn_set_label = QtWidgets.QPushButton("Set label", box_panel)
        self._btn_set_label.clicked.connect(self._set_label_selected)
        self._btn_del = QtWidgets.QPushButton("Delete", box_panel)
        self._btn_del.clicked.connect(lambda: self._canvas.delete_selected())
        btns.addWidget(self._btn_set_label)
        btns.addWidget(self._btn_del)
        box_layout.addLayout(btns)
        box_dock.setWidget(box_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, box_dock)

        self._counts = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self._counts)
        self.statusBar().showMessage("Open a folder with image + .json pairs.")

    def _build_actions(self) -> None:
        style = self.style()
        self._act_open = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon), "Open Folder", self)
        self._act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self._act_open.triggered.connect(self._open_folder)

        self._act_prev = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack), "Prev", self)
        self._act_prev.setShortcut("[")
        self._act_prev.triggered.connect(lambda: self._step(-1))

        self._act_next = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward), "Next", self)
        self._act_next.setShortcut("]")
        self._act_next.triggered.connect(lambda: self._step(1))

        self._act_draw = QtGui.QAction("Draw box", self)
        self._act_draw.setCheckable(True)
        self._act_draw.setShortcut("D")
        self._act_draw.toggled.connect(self._canvas.set_draw_mode)

        self._act_fit = QtGui.QAction("Fit", self)
        self._act_fit.setShortcut("F")
        self._act_fit.triggered.connect(self._canvas.fit_image)

        self._act_save = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton), "Save", self)
        self._act_save.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self._act_save.triggered.connect(self._save)

        for act in (self._act_open, self._act_prev, self._act_next):
            self._toolbar.addAction(act)
        self._toolbar.addSeparator()
        for act in (self._act_draw, self._act_fit):
            self._toolbar.addAction(act)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._act_save)

    # --- folder / navigation ---------------------------------------------
    def _open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder", str(Path.cwd()))
        if not folder:
            return
        self._load_folder(Path(folder))

    def _load_folder(self, folder: Path) -> None:
        entries: list[ImageEntry] = []
        for json_path in sorted(folder.glob("*.json")):
            for ext in IMAGE_EXTS:
                img = json_path.with_suffix(ext)
                if img.exists():
                    entries.append(ImageEntry(image=img, json=json_path))
                    break
        self._entries = entries
        self._populate_file_list()
        if not entries:
            self._canvas.clear()
            self._current = -1
            self.statusBar().showMessage(f"No image+.json pairs found in {folder}")
            return
        self._file_list.setCurrentRow(0)

    def _populate_file_list(self) -> None:
        self._file_list.blockSignals(True)
        self._file_list.clear()
        for e in self._entries:
            mark = "\u2713 " if e.saved else "   "
            self._file_list.addItem(f"{mark}{e.image.name}")
        self._file_list.blockSignals(False)

    def _on_file_selected(self, row: int) -> None:
        if row == self._current or not (0 <= row < len(self._entries)):
            return
        if not self._confirm_discard():
            self._file_list.blockSignals(True)
            self._file_list.setCurrentRow(self._current)
            self._file_list.blockSignals(False)
            return
        self._load_entry(row)

    def _step(self, delta: int) -> None:
        if not self._entries:
            return
        target = max(0, min(len(self._entries) - 1, self._current + delta))
        if target != self._current:
            self._file_list.setCurrentRow(target)

    def _load_entry(self, row: int) -> None:
        entry = self._entries[row]
        boxes = load_boxes(entry.jsonm) if entry.saved else load_boxes(entry.json)
        try:
            self._canvas.load(entry.image, boxes)
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        self._current = row
        self._dirty = False
        src = "edited" if entry.saved else "original"
        self.statusBar().showMessage(f"Loaded {entry.image.name} ({len(boxes)} boxes, {src})")
        self._refresh_box_list()

    # --- box list sync ----------------------------------------------------
    def _on_boxes_changed(self) -> None:
        self._dirty = True
        self._refresh_box_list()

    def _refresh_box_list(self) -> None:
        items = self._canvas.box_items()
        self._box_list.blockSignals(True)
        self._box_list.clear()
        for i, it in enumerate(items):
            r = it.scene_rect()
            self._box_list.addItem(f"#{i + 1}  {it.label or '?'}  {it.score:.2f}  ({int(r.width())}x{int(r.height())})")
        self._box_list.blockSignals(False)
        self._counts.setText(f"Boxes: {len(items)}{'  *' if self._dirty else ''}")

    def _on_box_row_selected(self, row: int) -> None:
        if row >= 0:
            self._canvas.select_index(row)

    def _on_scene_selection(self) -> None:
        items = self._canvas.box_items()
        selected = self._canvas.selected_items()
        if not selected:
            return
        idx = items.index(selected[0])
        if self._box_list.currentRow() != idx:
            self._box_list.blockSignals(True)
            self._box_list.setCurrentRow(idx)
            self._box_list.blockSignals(False)

    def _set_label_selected(self) -> None:
        self._canvas.set_label_for_selected(self._label_combo.currentText())

    # --- save -------------------------------------------------------------
    def _save(self) -> None:
        if not (0 <= self._current < len(self._entries)):
            return
        entry = self._entries[self._current]
        boxes = self._canvas.boxes()
        if entry.jsonm.exists():
            ok = QtWidgets.QMessageBox.question(
                self, "Overwrite", f"{entry.jsonm.name} already exists. Overwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if ok != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        save_boxes(entry.jsonm, boxes)
        self._dirty = False
        self._populate_file_list()
        self._file_list.blockSignals(True)
        self._file_list.setCurrentRow(self._current)
        self._file_list.blockSignals(False)
        self._refresh_box_list()
        self.statusBar().showMessage(f"Saved {entry.jsonm.name} ({len(boxes)} boxes)")

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        ans = QtWidgets.QMessageBox.question(
            self, "Unsaved changes", "Discard unsaved changes on this image?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        return ans == QtWidgets.QMessageBox.StandardButton.Yes

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._confirm_discard():
            super().closeEvent(event)
        else:
            event.ignore()


def main() -> int:
    QtCore.QLoggingCategory.setFilterRules("qt.accessibility.table.warning=false")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
