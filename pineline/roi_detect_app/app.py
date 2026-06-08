from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6 import QtCore, QtGui, QtWidgets


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DetectionRow:
    roi_index: int
    detector_name: str
    group_name: str
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


class RoiRectItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, rect: QtCore.QRectF, roi_index: int) -> None:
        super().__init__(rect)
        self.roi_index = int(roi_index)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(2)
        self._apply_style(False)

    def _apply_style(self, selected: bool) -> None:
        color = QtGui.QColor(255, 120, 80) if selected else QtGui.QColor(255, 198, 41)
        pen = QtGui.QPen(color, 3 if selected else 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 45 if selected else 30)))

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value):  # noqa: ANN001
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._apply_style(bool(value))
        return super().itemChange(change, value)

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionGraphicsItem, widget=None) -> None:  # noqa: ANN001
        super().paint(painter, option, widget)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        font = painter.font()
        font.setBold(True)
        font.setPointSizeF(max(10.0, self.rect().height() * 0.08))
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20)))
        tag = f"#{self.roi_index}"
        pos = self.rect().topLeft() + QtCore.QPointF(6.0, max(16.0, font.pointSizeF() + 4.0))
        painter.fillRect(QtCore.QRectF(self.rect().left(), self.rect().top(), 14.0 + 12.0 * len(tag), font.pointSizeF() + 10.0), QtGui.QColor(255, 198, 41, 200))
        painter.drawText(pos, tag)
        painter.restore()


class ImageCanvas(QtWidgets.QGraphicsView):
    roisChanged = QtCore.Signal()

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
        self._roi_counter = 0
        self._detections: list[QtWidgets.QGraphicsRectItem] = []

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = bool(enabled)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag if enabled else QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor)

    def set_image(self, path: Path) -> None:
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            raise RuntimeError(f"Cannot load image: {path}")
        self._scene.clear()
        self._detections.clear()
        self._roi_counter = 0
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.roisChanged.emit()

    def image_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF() if self._pixmap_item is None else self._pixmap_item.boundingRect()

    def roi_items(self) -> list[RoiRectItem]:
        return [item for item in self._scene.items() if isinstance(item, RoiRectItem)]

    def roi_rects(self) -> list[QtCore.QRectF]:
        rects = []
        bounds = self.image_rect()
        for item in sorted(self.roi_items(), key=lambda r: r.roi_index):
            rect = item.mapRectToScene(item.rect()).normalized().intersected(bounds)
            if rect.width() >= 4 and rect.height() >= 4:
                rects.append(rect)
        return rects

    def clear_rois(self) -> None:
        for item in self.roi_items():
            self._scene.removeItem(item)
        self.clear_detections()
        self.roisChanged.emit()

    def delete_roi(self, roi_index: int) -> None:
        for item in self.roi_items():
            if item.roi_index == int(roi_index):
                self._scene.removeItem(item)
        self.roisChanged.emit()

    def delete_selected_rois(self) -> int:
        removed = 0
        for item in self.roi_items():
            if item.isSelected():
                self._scene.removeItem(item)
                removed += 1
        if removed:
            self.roisChanged.emit()
        return removed

    def select_roi(self, roi_index: int) -> None:
        for item in self.roi_items():
            selected = item.roi_index == int(roi_index)
            item.setSelected(selected)
            if selected:
                self.centerOn(item)

    def fit_image(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def clear_detections(self) -> None:
        for item in self._detections:
            self._scene.removeItem(item)
        self._detections.clear()

    def show_detections(self, rows: list[DetectionRow]) -> None:
        self.clear_detections()
        colors = {
            "crack": QtGui.QColor(55, 150, 255),
            "mold": QtGui.QColor(52, 211, 153),
            "stain": QtGui.QColor(245, 158, 11),
            "spall": QtGui.QColor(248, 113, 113),
        }
        for row in rows:
            rect = QtCore.QRectF(row.x1, row.y1, row.x2 - row.x1, row.y2 - row.y1).normalized()
            item = self._scene.addRect(rect)
            color = colors.get(row.group_name, QtGui.QColor(255, 255, 255))
            pen = QtGui.QPen(color, 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 20)))
            item.setToolTip(f"{row.group_name} {row.score:.3f}")
            item.setZValue(4)
            self._detections.append(item)

    def render_overlay(self) -> QtGui.QImage:
        image = QtGui.QImage(int(self.image_rect().width()), int(self.image_rect().height()), QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        self._scene.render(painter, QtCore.QRectF(image.rect()), self.image_rect())
        painter.end()
        return image

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        modifiers = event.modifiers()
        zoom_mod = bool(modifiers & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.MetaModifier))
        # Trackpads emit pixelDelta (two-finger scroll) -> treat as pan/scroll.
        # A real mouse wheel only emits angleDelta -> treat as zoom.
        pixel_delta = event.pixelDelta()
        is_trackpad_scroll = not pixel_delta.isNull()

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
                # macOS trackpad pinch: value is incremental scale delta.
                self._zoom_by(1.0 + float(event.value()), event.position())
                return True
            if event.gestureType() == QtCore.Qt.NativeGestureType.SmartZoomNativeGesture:
                self.fit_image()
                return True
        return super().viewportEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            if self.delete_selected_rois():
                event.accept()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._draw_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_start = self.mapToScene(event.position().toPoint())
            self._rubber = self._scene.addRect(QtCore.QRectF(self._drag_start, self._drag_start), QtGui.QPen(QtGui.QColor(255, 198, 41), 2, QtCore.Qt.PenStyle.DashLine))
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
            rect = self._rubber.rect().normalized().intersected(self.image_rect())
            self._scene.removeItem(self._rubber)
            self._rubber = None
            self._drag_start = None
            if rect.width() >= 8 and rect.height() >= 8:
                self._roi_counter += 1
                self._scene.addItem(RoiRectItem(rect, self._roi_counter))
                self.roisChanged.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class DetectProcess(QtCore.QObject):
    """Runs detect_job.py as an independent OS process in its own session.

    Cancel kills the whole process group, so the GroundingDINO worker
    subprocesses spawned by the job are terminated too. The process is fully
    detached from the GUI event loop except for stdout streaming.
    """

    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, *, image_path: Path, rois: list[QtCore.QRectF], detector_name: str, conf: float, tmp_dir: Path, stabledino_checkpoint: str = "") -> None:
        super().__init__()
        self._image_path = Path(image_path)
        self._rois = [QtCore.QRectF(r) for r in rois]
        self._detector_name = str(detector_name)
        self._conf = float(conf)
        self._tmp_dir = Path(tmp_dir)
        self._stabledino_checkpoint = str(stabledino_checkpoint or "")
        self._proc: QtCore.QProcess | None = None
        self._out_json = self._tmp_dir / "detections_out.json"
        self._rois_json = self._tmp_dir / "rois_in.json"
        self._buf = ""
        self._cancelled = False

    def start(self) -> None:
        import os
        import json as _json

        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        rois_payload = [[float(r.left()), float(r.top()), float(r.right()), float(r.bottom())] for r in self._rois]
        self._rois_json.write_text(_json.dumps(rois_payload), encoding="utf-8")
        if self._out_json.exists():
            self._out_json.unlink()

        proc = QtCore.QProcess(self)
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        proc.setWorkingDirectory(str(ROOT))
        # The job calls os.setsid() itself on startup so it becomes a session
        # leader; killpg(pid) then reaches its GDINO worker children.
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)

        args = [
            "-m", "pineline.roi_detect_app.detect_job",
            "--image", str(self._image_path),
            "--detector", self._detector_name,
            "--conf", f"{self._conf}",
            "--rois-json", str(self._rois_json),
            "--tmp-dir", str(self._tmp_dir),
            "--out-json", str(self._out_json),
        ]
        if self._stabledino_checkpoint:
            args += ["--stabledino-checkpoint", self._stabledino_checkpoint]
        self._proc = proc
        self.log.emit(f"Launching detect process: pid will own its own session (rois={len(self._rois)})")
        proc.start(sys.executable, args)

    def cancel(self) -> None:
        import os
        import signal

        self._cancelled = True
        proc = self._proc
        if proc is None or proc.state() == QtCore.QProcess.ProcessState.NotRunning:
            return
        pid = int(proc.processId())
        self.log.emit(f"Cancel: killing process group of pid {pid}...")
        killed = False
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                killed = True
            except (ProcessLookupError, PermissionError):
                killed = False
        if not killed:
            proc.kill()

    @QtCore.Slot()
    def _on_output(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            if line.startswith("PROGRESS "):
                try:
                    done, total = line[len("PROGRESS "):].split("/")
                    self.progress.emit(int(done), int(total))
                except ValueError:
                    pass
                continue
            self.log.emit(line)

    @QtCore.Slot()
    def _on_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if self._cancelled:
            return
        self.failed.emit(f"Process error: {error}")

    @QtCore.Slot()
    def _on_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        import json as _json

        del exit_status
        if self._cancelled:
            self.failed.emit("Cancelled by user.")
            return
        if int(exit_code) != 0:
            self.failed.emit(f"Detect process exited with code {exit_code}")
            return
        try:
            payload = _json.loads(self._out_json.read_text(encoding="utf-8"))
        except Exception as exc:
            self.failed.emit(f"Cannot read results: {exc}")
            return
        rows = [
            DetectionRow(
                roi_index=int(item["roi_index"]),
                detector_name=str(item["detector_name"]),
                group_name=str(item["group_name"]),
                label=str(item["label"]),
                score=float(item["score"]),
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                x2=float(item["x2"]),
                y2=float(item["y2"]),
            )
            for item in payload
        ]
        self.finished.emit(rows)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROI Detect Review")
        self.resize(1500, 900)
        self._image_path: Path | None = None
        self._rows: list[DetectionRow] = []
        self._stabledino_checkpoint = ""
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="roi_detect_")
        self._proc: DetectProcess | None = None
        self._build_ui()
        self._build_actions()
        self._update_counts()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._proc is not None:
            self._proc.cancel()
        self._tmp_dir.cleanup()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self._canvas = ImageCanvas(self)
        self.setCentralWidget(self._canvas)
        self._canvas.roisChanged.connect(self._on_rois_changed)

        self._toolbar = QtWidgets.QToolBar("Main", self)
        self._toolbar.setMovable(False)
        self._toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(self._toolbar)

        # ROI dock
        roi_dock = QtWidgets.QDockWidget("ROIs", self)
        roi_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        roi_panel = QtWidgets.QWidget(roi_dock)
        roi_layout = QtWidgets.QVBoxLayout(roi_panel)
        roi_layout.setContentsMargins(6, 6, 6, 6)
        self._roi_list = QtWidgets.QListWidget(roi_panel)
        self._roi_list.itemSelectionChanged.connect(self._on_roi_list_selected)
        roi_layout.addWidget(QtWidgets.QLabel("Drawn ROIs", roi_panel))
        roi_layout.addWidget(self._roi_list, 1)
        roi_btns = QtWidgets.QHBoxLayout()
        self._btn_del_roi = QtWidgets.QPushButton("Delete selected", roi_panel)
        self._btn_del_roi.clicked.connect(self._delete_selected_roi)
        self._btn_clear = QtWidgets.QPushButton("Clear all", roi_panel)
        self._btn_clear.clicked.connect(self._clear)
        roi_btns.addWidget(self._btn_del_roi)
        roi_btns.addWidget(self._btn_clear)
        roi_layout.addLayout(roi_btns)
        roi_dock.setWidget(roi_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, roi_dock)

        # Settings + results dock
        result_dock = QtWidgets.QDockWidget("Detection", self)
        result_panel = QtWidgets.QWidget(result_dock)
        result_layout = QtWidgets.QVBoxLayout(result_panel)
        result_layout.setContentsMargins(6, 6, 6, 6)

        form = QtWidgets.QFormLayout()
        self._detector_combo = QtWidgets.QComboBox(result_panel)
        self._detector_combo.addItems(["gdino", "stabledino"])
        self._conf = QtWidgets.QDoubleSpinBox(result_panel)
        self._conf.setRange(0.001, 1.0)
        self._conf.setDecimals(3)
        self._conf.setSingleStep(0.01)
        self._conf.setValue(0.05)
        self._min_score = QtWidgets.QDoubleSpinBox(result_panel)
        self._min_score.setRange(0.0, 1.0)
        self._min_score.setDecimals(3)
        self._min_score.setSingleStep(0.01)
        self._min_score.setValue(0.0)
        self._min_score.valueChanged.connect(self._refresh_results)
        self._class_filter = QtWidgets.QComboBox(result_panel)
        self._class_filter.addItem("all")
        self._class_filter.addItems(["crack", "mold", "stain", "spall"])
        self._class_filter.currentTextChanged.connect(self._refresh_results)
        form.addRow("Detector", self._detector_combo)
        form.addRow("Run conf", self._conf)
        form.addRow("Show score >=", self._min_score)
        form.addRow("Show class", self._class_filter)
        result_layout.addLayout(form)

        self._table = QtWidgets.QTableWidget(0, 4, result_panel)
        self._table.setHorizontalHeaderLabels(["ROI", "Class", "Score", "Box"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.itemSelectionChanged.connect(self._on_table_selected)
        result_layout.addWidget(QtWidgets.QLabel("Detections", result_panel))
        result_layout.addWidget(self._table, 1)
        result_dock.setWidget(result_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, result_dock)

        # Log dock
        log_dock = QtWidgets.QDockWidget("Log", self)
        self._log = QtWidgets.QPlainTextEdit(log_dock)
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setPlaceholderText("Detection log...")
        log_dock.setWidget(self._log)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

        # Status bar
        self._counts = QtWidgets.QLabel("")
        self._progress = QtWidgets.QProgressBar()
        self._progress.setMaximumWidth(220)
        self._progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._counts)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("Open image, draw ROI, run detect, then save if OK.")

    def _build_actions(self) -> None:
        style = self.style()

        self._act_open = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton), "Open", self)
        self._act_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self._act_open.triggered.connect(self._open_image)

        self._act_draw = QtGui.QAction("Draw ROI", self)
        self._act_draw.setCheckable(True)
        self._act_draw.setShortcut("D")
        self._act_draw.toggled.connect(self._canvas.set_draw_mode)

        self._act_fit = QtGui.QAction("Fit", self)
        self._act_fit.setShortcut("F")
        self._act_fit.triggered.connect(self._canvas.fit_image)

        self._act_run = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay), "Run", self)
        self._act_run.setShortcut("R")
        self._act_run.triggered.connect(self._run_detect)

        self._act_cancel = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop), "Cancel", self)
        self._act_cancel.setEnabled(False)
        self._act_cancel.triggered.connect(self._cancel_detect)

        self._act_save = QtGui.QAction(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton), "Save", self)
        self._act_save.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self._act_save.setEnabled(False)
        self._act_save.triggered.connect(self._save)

        for act in (self._act_open, self._act_draw, self._act_fit):
            self._toolbar.addAction(act)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._act_run)
        self._toolbar.addAction(self._act_cancel)
        self._toolbar.addSeparator()
        self._toolbar.addAction(self._act_save)

        menu = self.menuBar().addMenu("Detector")
        act_sd = QtGui.QAction("Set StableDINO checkpoint...", self)
        act_sd.triggered.connect(self._set_stabledino_checkpoint)
        menu.addAction(act_sd)

    def _set_stabledino_checkpoint(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "StableDINO checkpoint (.pth)", str(self._stabledino_checkpoint or Path.cwd()), "Checkpoint (*.pth)")
        if path:
            self._stabledino_checkpoint = path
            self.statusBar().showMessage(f"StableDINO checkpoint: {path}")

    def _open_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image", str(Path.cwd()), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if not path:
            return
        self._image_path = Path(path)
        self._rows = []
        self._canvas.set_image(self._image_path)
        self._refresh_results()
        self._update_counts()
        self.statusBar().showMessage(f"Loaded {self._image_path.name}")

    def _clear(self) -> None:
        self._rows = []
        self._canvas.clear_rois()
        self._log.clear()
        self._refresh_results()
        self._update_counts()
        self.statusBar().showMessage("ROIs cleared")

    def _delete_selected_roi(self) -> None:
        if self._canvas.delete_selected_rois() == 0:
            item = self._roi_list.currentItem()
            if item is not None:
                self._canvas.delete_roi(int(item.data(QtCore.Qt.ItemDataRole.UserRole)))

    def _on_rois_changed(self) -> None:
        self._roi_list.blockSignals(True)
        self._roi_list.clear()
        for item in sorted(self._canvas.roi_items(), key=lambda r: r.roi_index):
            rect = item.mapRectToScene(item.rect()).normalized()
            li = QtWidgets.QListWidgetItem(f"ROI #{item.roi_index}  ({int(rect.width())}x{int(rect.height())})")
            li.setData(QtCore.Qt.ItemDataRole.UserRole, item.roi_index)
            self._roi_list.addItem(li)
        self._roi_list.blockSignals(False)
        self._update_counts()

    def _on_roi_list_selected(self) -> None:
        item = self._roi_list.currentItem()
        if item is not None:
            self._canvas.select_roi(int(item.data(QtCore.Qt.ItemDataRole.UserRole)))

    def _update_counts(self) -> None:
        roi_n = len(self._canvas.roi_items())
        self._counts.setText(f"ROI: {roi_n}   Boxes: {len(self._rows)}   Shown: {self._table.rowCount()}")
        self._act_save.setEnabled(bool(self._rows))

    def _filtered_rows(self) -> list[DetectionRow]:
        min_score = float(self._min_score.value())
        cls = self._class_filter.currentText()
        rows = [r for r in self._rows if r.score >= min_score]
        if cls != "all":
            rows = [r for r in rows if r.group_name == cls]
        return rows

    def _refresh_results(self) -> None:
        rows = self._filtered_rows()
        self._canvas.show_detections(rows)
        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(row.roi_index)))
            self._table.setItem(i, 1, QtWidgets.QTableWidgetItem(row.group_name))
            self._table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{row.score:.3f}"))
            self._table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{int(row.x1)},{int(row.y1)},{int(row.x2)},{int(row.y2)}"))
        self._update_counts()

    def _on_table_selected(self) -> None:
        rows = self._filtered_rows()
        idx = self._table.currentRow()
        if 0 <= idx < len(rows):
            self._canvas.select_roi(rows[idx].roi_index)

    def _run_detect(self) -> None:
        if self._image_path is None:
            QtWidgets.QMessageBox.information(self, "No image", "Open an image first.")
            return
        rois = self._canvas.roi_rects()
        if not rois:
            QtWidgets.QMessageBox.information(self, "No ROI", "Draw at least one ROI first.")
            return

        self._rows = []
        self._canvas.clear_detections()
        self._log.clear()
        self._set_busy(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(rois))
        self._progress.setValue(0)
        sd_ckpt = ""
        if self._detector_combo.currentText() == "stabledino":
            sd_ckpt = self._stabledino_checkpoint
            if not sd_ckpt:
                self._set_busy(False)
                self._progress.setVisible(False)
                QtWidgets.QMessageBox.information(self, "StableDINO checkpoint", "Set StableDINO checkpoint first (Detector menu).")
                return
        self._proc = DetectProcess(
            image_path=self._image_path,
            rois=rois,
            detector_name=self._detector_combo.currentText(),
            conf=float(self._conf.value()),
            tmp_dir=Path(self._tmp_dir.name),
            stabledino_checkpoint=sd_ckpt,
        )
        self._proc.log.connect(self._append_log)
        self._proc.progress.connect(self._on_progress)
        self._proc.finished.connect(self._on_detect_finished)
        self._proc.failed.connect(self._on_detect_failed)
        self._proc.start()

    def _cancel_detect(self) -> None:
        if self._proc is not None:
            self._proc.cancel()
            self._append_log("Cancel requested...")

    @QtCore.Slot(int, int)
    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(done)

    @QtCore.Slot(str)
    def _append_log(self, message: str) -> None:
        text = str(message)
        self._log.appendPlainText(text)
        self.statusBar().showMessage(text)

    @QtCore.Slot(list)
    def _on_detect_finished(self, rows: list[DetectionRow]) -> None:
        self._rows = list(rows)
        self._refresh_results()
        self._set_busy(False)
        self._progress.setVisible(False)
        self._proc = None
        self._append_log(f"Detect done: {len(self._canvas.roi_rects())} ROI, {len(self._rows)} boxes. Review overlay, then Save if OK.")

    @QtCore.Slot(str)
    def _on_detect_failed(self, message: str) -> None:
        self._set_busy(False)
        self._progress.setVisible(False)
        self._proc = None
        self._append_log(f"ERROR: {message}")
        if not str(message).startswith("Cancelled"):
            QtWidgets.QMessageBox.warning(self, "Detect failed", str(message))

    def _set_busy(self, busy: bool) -> None:
        for widget in (self._act_open, self._act_draw, self._act_fit, self._act_run, self._act_save,
                       self._detector_combo, self._conf, self._btn_del_roi, self._btn_clear):
            widget.setEnabled(not busy)
        self._act_cancel.setEnabled(busy)
        if not busy:
            self._act_save.setEnabled(bool(self._rows))
        if busy:
            self.statusBar().showMessage("Detect running...")

    def _save(self) -> None:
        if self._image_path is None:
            return
        out_dir_str = QtWidgets.QFileDialog.getExistingDirectory(self, "Save results to", str(self._image_path.parent))
        if not out_dir_str:
            return
        out_dir = Path(out_dir_str) / f"roi_detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        crops_dir = out_dir / "rois"
        out_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        overlay = self._canvas.render_overlay()
        overlay.save(str(out_dir / f"{self._image_path.stem}_overlay.png"))

        image = QtGui.QImage(str(self._image_path)).convertToFormat(QtGui.QImage.Format.Format_RGB888)
        for roi_index, rect in enumerate(self._canvas.roi_rects(), start=1):
            x1 = max(0, int(rect.left()))
            y1 = max(0, int(rect.top()))
            x2 = min(image.width(), int(rect.right()))
            y2 = min(image.height(), int(rect.bottom()))
            image.copy(x1, y1, max(1, x2 - x1), max(1, y2 - y1)).save(str(crops_dir / f"roi_{roi_index:03d}.png"))

        with (out_dir / "detections.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["roi_index", "detector_name", "group_name", "label", "score", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            for row in self._filtered_rows():
                writer.writerow(row.__dict__)
        self.statusBar().showMessage(f"Saved: {out_dir}")


def main() -> int:
    QtCore.QLoggingCategory.setFilterRules("qt.accessibility.table.warning=false")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
