from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ...services.image_service import ImageService, item_image_path


class ThumbListModel(QtCore.QAbstractListModel):
    ItemRole = QtCore.Qt.ItemDataRole.UserRole + 1
    PickedRole = QtCore.Qt.ItemDataRole.UserRole + 2

    def __init__(self, image_service: ImageService, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[Any] = []
        self._picked_ids: set[int] = set()
        self._image_service = image_service
        self._placeholder = QtGui.QPixmap(150, 110)
        self._placeholder.fill(QtGui.QColor("#e3e5e7"))
        self._image_service.imageLoaded.connect(self._on_image_loaded)

    def set_items(self, items: list[Any]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()

    def set_picked_ids(self, picked_ids: set[int]) -> None:
        self._picked_ids = {int(x) for x in picked_ids}
        if self._items:
            top = self.index(0, 0)
            bottom = self.index(len(self._items) - 1, 0)
            self.dataChanged.emit(top, bottom, [self.PickedRole])

    def item_at(self, row: int) -> Any | None:
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        item = self._items[index.row()]
        result_id = int(getattr(item, "result_id", 0) or 0)
        if role == self.ItemRole:
            return item
        if role == self.PickedRole:
            return result_id in self._picked_ids
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            label = getattr(item, "final_label", getattr(item, "label", getattr(item, "suggested_label", "")))
            score = getattr(item, "reliability_score", None)
            score_text = f" {float(score):.2f}" if score is not None else ""
            return f"{result_id}\n{label}{score_text}"
        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            pixmap = self._image_service.cached(result_id)
            if pixmap is not None:
                return pixmap
            self._image_service.load_thumbnail(result_id, item_image_path(item), 170)
            return self._placeholder
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return str(getattr(item, "image_rel_path", "") or result_id)
        return None

    def _on_image_loaded(self, key: object, _pixmap: QtGui.QPixmap) -> None:
        try:
            result_id = int(key)
        except Exception:
            return
        for row, item in enumerate(self._items):
            if int(getattr(item, "result_id", -1)) == result_id:
                idx = self.index(row, 0)
                self.dataChanged.emit(idx, idx, [QtCore.Qt.ItemDataRole.DecorationRole])
                break


class ThumbDelegate(QtWidgets.QStyledItemDelegate):
    """Draws an accent border + check badge over picked thumbnails."""

    ACCENT = QtGui.QColor("#3daee9")

    def paint(self, painter, option, index):  # noqa: ANN001
        super().paint(painter, option, index)
        if not index.data(ThumbListModel.PickedRole):
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = QtCore.QRectF(option.rect).adjusted(3, 3, -3, -3)
        pen = QtGui.QPen(self.ACCENT, 3)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(61, 174, 233, 38))
        painter.drawRoundedRect(rect, 6, 6)
        # check badge, top-right
        badge_d = 22.0
        cx = rect.right() - badge_d / 2 - 2
        cy = rect.top() + badge_d / 2 + 2
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.ACCENT)
        painter.drawEllipse(QtCore.QPointF(cx, cy), badge_d / 2, badge_d / 2)
        check = QtGui.QPen(QtGui.QColor("#ffffff"), 2.4)
        check.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        check.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(check)
        path = QtGui.QPainterPath()
        path.moveTo(cx - 5.0, cy + 0.5)
        path.lineTo(cx - 1.5, cy + 4.0)
        path.lineTo(cx + 5.5, cy - 4.0)
        painter.drawPath(path)
        painter.restore()


class ThumbGrid(QtWidgets.QWidget):
    itemActivated = QtCore.Signal(object)
    selectionChanged = QtCore.Signal(list)

    def __init__(self, image_service: ImageService | None = None, parent: QtWidgets.QWidget | None = None) -> None:
        if isinstance(image_service, QtWidgets.QWidget) and parent is None:
            parent = image_service
            image_service = None
        super().__init__(parent)
        self.image_service = image_service or ImageService(parent=self)
        self.model = ThumbListModel(self.image_service, self)
        self.view = QtWidgets.QListView(self)
        self.view.setModel(self.model)
        self.view.setItemDelegate(ThumbDelegate(self.view))
        self.view.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.view.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.view.setMovement(QtWidgets.QListView.Movement.Static)
        self.view.setUniformItemSizes(True)
        self.view.setIconSize(QtCore.QSize(170, 130))
        self.view.setGridSize(QtCore.QSize(210, 178))
        self.view.setSpacing(4)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.activated.connect(self._emit_item)
        self.view.clicked.connect(self._emit_item)
        self.view.selectionModel().selectionChanged.connect(self._emit_selection)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

    def set_items(self, items: list[Any]) -> None:
        self.model.set_items(items)

    def set_picked_ids(self, picked_ids: set[int]) -> None:
        self.model.set_picked_ids(picked_ids)

    def selected_items(self) -> list[Any]:
        return [
            self.model.item_at(index.row())
            for index in self.view.selectionModel().selectedIndexes()
            if self.model.item_at(index.row()) is not None
        ]

    def _emit_item(self, index: QtCore.QModelIndex) -> None:
        item = self.model.item_at(index.row())
        if item is not None:
            self.itemActivated.emit(item)

    def _emit_selection(self) -> None:
        self.selectionChanged.emit(self.selected_items())
