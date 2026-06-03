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
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QtGui.QPixmap()
        self._box: tuple[float, float, float, float] | None = None
        self._draw_box = False
        self._other_boxes: list[dict[str, Any]] = []
        self._show_others = True
        self.setMinimumSize(420, 320)
        self.setAutoFillBackground(True)

    def set_item(self, item: Any, prefer_full_image: bool = False) -> None:
        path = ""
        if prefer_full_image:
            uri = str(getattr(item, "image_uri", "") or "")
            path = QtCore.QUrl(uri).toLocalFile() if uri.startswith("file:") else uri
        if not path:
            path = item_image_path(item)
        self._box = getattr(item, "box", None)
        self._draw_box = bool(prefer_full_image and path)
        self._other_boxes = []  # cleared until the caller supplies the image's other boxes
        self._pixmap = QtGui.QPixmap(path) if path and Path(path).is_file() else QtGui.QPixmap()
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
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#fcfcfc"))
        if self._pixmap.isNull():
            painter.setPen(QtGui.QColor("#9aa0a4"))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Chưa có ảnh")
            return
        scaled = self._pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
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
