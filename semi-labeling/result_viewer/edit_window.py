"""
Edit window – opened when the user clicks "Edit" on an image card.

Features
--------
* **Zoom / Pan** via mouse-wheel (pinch on trackpad) and middle-button drag.
* **Crop mode** – draw a rectangle on the image; the drawn box becomes the new
  bounding box for the item.  Old box is replaced on save.
* **Relabel mode** – pick a different label from a combo-box; saves immediately.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from image_loader import resolve_image_path
from models import LABELS, AssignmentRow

if TYPE_CHECKING:
    from PySide6.QtGui import QMouseEvent

# ── Colours for bounding boxes ───────────────────────────
_BOX_PEN = QPen(QColor(0, 200, 255, 200), 2.5, Qt.SolidLine)
_DRAW_PEN = QPen(QColor(255, 80, 80, 220), 2.5, Qt.DashLine)
_DRAW_FILL = QBrush(QColor(255, 80, 80, 40))


# ─────────────────────────────────────────────────────────
#  Zoomable / pannable QGraphicsView
# ─────────────────────────────────────────────────────────


class ZoomPanView(QGraphicsView):
    """QGraphicsView that supports:
    * Scroll-wheel / trackpad zoom (Ctrl+Wheel or pinch)
    * Middle-button drag to pan
    * Left-click rectangle drawing when *draw_mode* is active.
    """

    rect_drawn = Signal(QRectF)   # emitted when a crop rect is finished

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._zoom_factor = 1.0
        self._draw_mode = False
        self._drawing = False
        self._draw_start: QPointF | None = None
        self._draw_rect_item: QGraphicsRectItem | None = None

    # ── draw mode toggle ────────────────────────────────

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)

    # ── zoom ────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        angle = event.angleDelta().y()
        if angle == 0:
            super().wheelEvent(event)
            return
        factor = 1.25 if angle > 0 else 1 / 1.25
        self._zoom_factor *= factor
        # clamp
        if self._zoom_factor < 0.05:
            self._zoom_factor = 0.05
            return
        if self._zoom_factor > 50:
            self._zoom_factor = 50
            return
        self.scale(factor, factor)

    # ── rectangle drawing ───────────────────────────────

    def mousePressEvent(self, event) -> None:
        if self._draw_mode and event.button() == Qt.LeftButton:
            self._drawing = True
            self._draw_start = self.mapToScene(event.position().toPoint())
            if self._draw_rect_item is not None:
                self.scene().removeItem(self._draw_rect_item)
                self._draw_rect_item = None
            self._draw_rect_item = QGraphicsRectItem()
            self._draw_rect_item.setPen(_DRAW_PEN)
            self._draw_rect_item.setBrush(_DRAW_FILL)
            self.scene().addItem(self._draw_rect_item)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drawing and self._draw_start is not None and self._draw_rect_item is not None:
            current = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._draw_start, current).normalized()
            self._draw_rect_item.setRect(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            if self._draw_rect_item is not None:
                rect = self._draw_rect_item.rect()
                if rect.width() > 3 and rect.height() > 3:
                    self.rect_drawn.emit(rect)
            return
        super().mouseReleaseEvent(event)

    def clear_drawn_rect(self) -> None:
        if self._draw_rect_item is not None:
            self.scene().removeItem(self._draw_rect_item)
            self._draw_rect_item = None


# ─────────────────────────────────────────────────────────
#  Edit Window (QDialog)
# ─────────────────────────────────────────────────────────


class EditWindow(QDialog):
    """Modal-less editor window for one image + its bounding boxes.

    Signals
    -------
    saved(list[AssignmentRow])
        Emitted when the user saves; carries the *modified* rows for this image.
    """

    saved = Signal(list)   # list[AssignmentRow]

    def __init__(
        self,
        rows: list[AssignmentRow],
        image_root: Path | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Annotations")
        self.resize(1100, 800)
        self.setMinimumSize(600, 400)

        self._rows = list(rows)            # mutable working copy
        self._image_root = image_root
        self._pending_crop_rects: list[QRectF] = []
        self._crop_mode = False
        self._current_crop_label: str = LABELS[0]

        # ── Load the underlying full image ───────────────
        first_row = rows[0]
        self._image_path = resolve_image_path(first_row, image_root)
        self._full_pixmap = QPixmap(str(self._image_path))
        if self._full_pixmap.isNull():
            QMessageBox.warning(self, "Image error", f"Cannot load:\n{self._image_path}")

        # ── Scene + View ─────────────────────────────────
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem(self._full_pixmap)
        self._scene.addItem(self._pixmap_item)
        self._box_items: list[QGraphicsRectItem] = []
        self._draw_existing_boxes()

        self._view = ZoomPanView(self._scene, self)
        self._view.rect_drawn.connect(self._on_rect_drawn)

        # ── Toolbar ──────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.crop_btn = QPushButton("Crop")
        self.crop_btn.setFixedSize(120, 36)
        self.crop_btn.setCheckable(True)
        self.crop_btn.setToolTip("Draw new bounding boxes on the image")
        self.crop_btn.clicked.connect(self._toggle_crop_mode)
        toolbar.addWidget(self.crop_btn)

        self.crop_label_combo = QComboBox()
        self.crop_label_combo.setFixedHeight(36)
        for lbl in LABELS:
            self.crop_label_combo.addItem(lbl)
        self.crop_label_combo.currentTextChanged.connect(self._on_crop_label_changed)
        self.crop_label_combo.setToolTip("Label for new crop boxes")
        toolbar.addWidget(QLabel("Label:"))
        toolbar.addWidget(self.crop_label_combo)

        self.relabel_btn = QPushButton("Relabel")
        self.relabel_btn.setFixedSize(120, 36)
        self.relabel_btn.setToolTip("Change the label of all boxes in this image")
        self.relabel_btn.clicked.connect(self._on_relabel)
        toolbar.addWidget(self.relabel_btn)

        self.undo_crop_btn = QPushButton("Undo crop")
        self.undo_crop_btn.setFixedSize(120, 36)
        self.undo_crop_btn.setEnabled(False)
        self.undo_crop_btn.clicked.connect(self._undo_last_crop)
        toolbar.addWidget(self.undo_crop_btn)

        toolbar.addStretch(1)

        self.save_btn = QPushButton("Save")
        self.save_btn.setFixedSize(120, 36)
        self.save_btn.clicked.connect(self._on_save)
        toolbar.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedSize(120, 36)
        self.cancel_btn.clicked.connect(self.reject)
        toolbar.addWidget(self.cancel_btn)

        # ── Status label ─────────────────────────────────
        self.status_label = QLabel(f"{len(self._rows)} box(es)  —  {self._image_path.name}")
        self.status_label.setFont(QFont("Arial", 10))

        # ── Layout ───────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)
        root.addLayout(toolbar)
        root.addWidget(self._view, 1)
        root.addWidget(self.status_label)

        # Fit view initially
        self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    # ── drawing existing boxes ──────────────────────────

    def _draw_existing_boxes(self) -> None:
        for item in self._box_items:
            self._scene.removeItem(item)
        self._box_items.clear()
        for row in self._rows:
            x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
            if x2 <= x1 or y2 <= y1:
                continue
            rect_item = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
            rect_item.setPen(_BOX_PEN)
            rect_item.setToolTip(f"id={row.result_id}  {row.predicted_label}")
            self._scene.addItem(rect_item)
            self._box_items.append(rect_item)

    # ── Crop mode ───────────────────────────────────────

    def _toggle_crop_mode(self) -> None:
        self._crop_mode = self.crop_btn.isChecked()
        self._view.set_draw_mode(self._crop_mode)
        if self._crop_mode:
            self._pending_crop_rects.clear()
            self.status_label.setText("CROP MODE: Draw rectangle(s) on the image, then Save")
        else:
            self.status_label.setText(f"{len(self._rows)} box(es)")

    def _on_crop_label_changed(self, text: str) -> None:
        self._current_crop_label = text

    def _on_rect_drawn(self, rect: QRectF) -> None:
        """Called when the user finishes drawing a rectangle."""
        # Clamp to image bounds
        iw = self._full_pixmap.width()
        ih = self._full_pixmap.height()
        x1 = max(0.0, rect.left())
        y1 = max(0.0, rect.top())
        x2 = min(float(iw), rect.right())
        y2 = min(float(ih), rect.bottom())
        clamped = QRectF(x1, y1, x2 - x1, y2 - y1)

        self._pending_crop_rects.append(clamped)
        self.undo_crop_btn.setEnabled(True)
        self.status_label.setText(
            f"CROP: {len(self._pending_crop_rects)} new box(es) drawn  "
            f"(label={self._current_crop_label})"
        )

    def _undo_last_crop(self) -> None:
        if self._pending_crop_rects:
            self._pending_crop_rects.pop()
            self._view.clear_drawn_rect()
            self.status_label.setText(
                f"CROP: {len(self._pending_crop_rects)} new box(es) drawn"
            )
        self.undo_crop_btn.setEnabled(bool(self._pending_crop_rects))

    # ── Relabel ─────────────────────────────────────────

    def _on_relabel(self) -> None:
        dlg = _RelabelDialog(self._rows, parent=self)
        if dlg.exec() == QDialog.Accepted:
            new_label = dlg.selected_label()
            if new_label:
                self._rows = [
                    replace(row, suggested_label=new_label) for row in self._rows
                ]
                self._draw_existing_boxes()
                self.status_label.setText(
                    f"Relabelled {len(self._rows)} box(es) → {new_label}"
                )

    # ── Save ────────────────────────────────────────────

    def _on_save(self) -> None:
        if self._pending_crop_rects:
            # Replace old rows with new ones from drawn rectangles
            template = self._rows[0] if self._rows else None
            if template is None:
                return
            new_rows: list[AssignmentRow] = []
            for rect in self._pending_crop_rects:
                new_rows.append(
                    replace(
                        template,
                        x1=rect.left(),
                        y1=rect.top(),
                        x2=rect.right(),
                        y2=rect.bottom(),
                        suggested_label=self._current_crop_label,
                    )
                )
            self._rows = new_rows
        self.saved.emit(self._rows)
        self.accept()


# ─────────────────────────────────────────────────────────
#  Relabel Dialog
# ─────────────────────────────────────────────────────────


class _RelabelDialog(QDialog):
    def __init__(self, rows: list[AssignmentRow], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Relabel")
        self.resize(300, 120)
        layout = QVBoxLayout(self)
        current_labels = set(r.predicted_label for r in rows)
        layout.addWidget(QLabel(f"Current: {', '.join(sorted(current_labels))}"))
        layout.addWidget(QLabel("New label:"))
        self.combo = QComboBox()
        for lbl in LABELS:
            self.combo.addItem(lbl)
        layout.addWidget(self.combo)
        btns = QHBoxLayout()
        ok = QPushButton("OK")
        ok.setFixedSize(100, 36)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.setFixedSize(100, 36)
        cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

    def selected_label(self) -> str:
        return self.combo.currentText()
