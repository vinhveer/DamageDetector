from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ...services.image_service import item_image_path


# Per-label colours for the context boxes (the box being reviewed stays red).
LABEL_COLORS: dict[str, str] = {
    "crack": "#4a90d9",
    "mold": "#27ae60",
    "spall": "#f39c12",
    "stain": "#9b59b6",
    "reject": "#7f8c8d",
}
OTHER_COLOR = "#7f8c8d"
CURRENT_COLOR = "#da4453"


class BoxImage(QtWidgets.QWidget):
    clicked = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QtGui.QPixmap()
        self._box: tuple[float, float, float, float] | None = None
        self._draw_box = False
        self._other_boxes: list[dict[str, Any]] = []
        self._show_others = True
        self._message = "Chưa có ảnh"
        self._caption = ""
        self._decision_indicator = ""
        self.setMinimumSize(420, 320)
        self.setAutoFillBackground(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    @staticmethod
    def item_path(item: Any, prefer_full_image: bool = False) -> str:
        path = ""
        if prefer_full_image:
            uri = str(getattr(item, "image_uri", "") or "")
            path = QtCore.QUrl(uri).toLocalFile() if uri.startswith("file:") else uri
        return path or item_image_path(item)

    def set_item(self, item: Any, prefer_full_image: bool = False) -> None:
        path = self.item_path(item, prefer_full_image=prefer_full_image)
        self._box = getattr(item, "box", None)
        self._draw_box = bool(prefer_full_image and path)
        self._other_boxes = []  # cleared until the caller supplies the image's other boxes
        self._caption = ""
        self._pixmap = QtGui.QPixmap(path) if path and Path(path).is_file() else QtGui.QPixmap()
        self._message = "Chưa có ảnh" if self._pixmap.isNull() else ""
        self.update()

    def set_loading_item(self, item: Any, prefer_full_image: bool = False) -> None:
        path = self.item_path(item, prefer_full_image=prefer_full_image)
        self._box = getattr(item, "box", None)
        self._draw_box = bool(prefer_full_image and path)
        self._other_boxes = []
        self._caption = ""
        self._pixmap = QtGui.QPixmap()
        self._message = "Đang tải ảnh..."
        self.update()

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self._message = "" if not pixmap.isNull() else "Chưa có ảnh"
        self.update()

    def set_caption(self, text: str) -> None:
        self._caption = str(text or "")
        self.update()

    def set_decision_indicator(self, text: str) -> None:
        self._decision_indicator = str(text or "")
        self.update()

    def set_overlay_loading(self, *, boxes: list[dict[str, Any]], caption: str = "") -> None:
        self._box = None
        self._draw_box = True
        self._other_boxes = list(boxes or [])
        self._caption = str(caption or "")
        self._pixmap = QtGui.QPixmap()
        self._message = "Đang tải ảnh..."
        self.update()

    def set_overlay_boxes(self, *, boxes: list[dict[str, Any]], caption: str = "") -> None:
        self._box = None
        self._draw_box = True
        self._other_boxes = list(boxes or [])
        self._caption = str(caption or "")
        self.update()

    def set_other_boxes(self, boxes: list[dict[str, Any]], current_result_id: int | None = None) -> None:
        """Context boxes for the full image (each {result_id, label, box}); the current one is skipped."""
        self._other_boxes = [b for b in boxes if int(b.get("result_id", -1)) != int(current_result_id or -1)]
        self.update()

    def set_show_others(self, show: bool) -> None:
        self._show_others = bool(show)
        self.update()

    def clear(self) -> None:
        self._pixmap = QtGui.QPixmap()
        self._box = None
        self._draw_box = False
        self._other_boxes = []
        self._caption = ""
        self._decision_indicator = ""
        self._message = "Chưa có ảnh"
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#fcfcfc"))
        if self._pixmap.isNull():
            painter.setPen(QtGui.QColor("#9aa0a4"))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self._message or "Chưa có ảnh")
            return
        scaled = self._pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        if self._caption:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_rect = metrics.boundingRect(self._caption).adjusted(-8, -4, 8, 4)
            text_rect.moveTopLeft(QtCore.QPoint(x + 10, y + 10))
            painter.fillRect(text_rect, QtGui.QColor(255, 255, 255, 220))
            painter.setPen(QtGui.QColor("#202020"))
            painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._caption)
        if self._decision_indicator:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_rect = metrics.boundingRect(self._decision_indicator).adjusted(-10, -5, 10, 5)
            text_rect.moveBottomRight(QtCore.QPoint(x + scaled.width() - 10, y + scaled.height() - 10))
            painter.fillRect(text_rect, QtGui.QColor(32, 38, 46, 220))
            painter.setPen(QtGui.QColor("#ffffff"))
            painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._decision_indicator)
        if not (self._draw_box and self._pixmap.width() > 0 and self._pixmap.height() > 0):
            return
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        sx = scaled.width() / self._pixmap.width()
        sy = scaled.height() / self._pixmap.height()

        def to_rect(box: tuple[float, float, float, float]) -> QtCore.QRectF:
            x1, y1, x2, y2 = box
            return QtCore.QRectF(x + x1 * sx, y + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)

        # Context boxes first (thin, semi-transparent, coloured by label) so the
        # current box draws on top.
        if self._show_others:
            for entry in self._other_boxes:
                box = entry.get("box")
                if not box:
                    continue
                color = QtGui.QColor(LABEL_COLORS.get(str(entry.get("label", "")), OTHER_COLOR))
                color.setAlpha(170)
                painter.setPen(QtGui.QPen(color, 1.4))
                painter.drawRect(to_rect(box))

        if self._box:
            painter.setPen(QtGui.QPen(QtGui.QColor(CURRENT_COLOR), 3))
            painter.drawRect(to_rect(self._box))
