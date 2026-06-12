"""Reusable list / grid views backed by a lightweight payload model.

These were previously defined inline in ``main_window``.  They are pure view
components: they hold opaque ``payload`` objects, render a title + lazy
thumbnail, and emit the current payload / visible rows so pages can drive
on-demand thumbnail loading.
"""
from __future__ import annotations

from typing import Any, Callable

from PySide6 import QtCore, QtGui, QtWidgets

TitleFn = Callable[[Any], str]
ThumbKeyFn = Callable[[Any], object]


class PayloadListModel(QtCore.QAbstractListModel):
    PayloadRole = QtCore.Qt.ItemDataRole.UserRole + 1
    ThumbnailKeyRole = QtCore.Qt.ItemDataRole.UserRole + 2

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._payloads: list[Any] = []
        self._title_fn: TitleFn | None = None
        self._thumb_key_fn: ThumbKeyFn | None = None
        self._thumbs: dict[object, QtGui.QPixmap] = {}

    def set_payloads(self, payloads: list[Any], title_fn: TitleFn, thumb_key_fn: ThumbKeyFn | None = None) -> None:
        self.beginResetModel()
        self._payloads = list(payloads)
        self._title_fn = title_fn
        self._thumb_key_fn = thumb_key_fn
        self.endResetModel()

    def thumb_key_at(self, row: int) -> object | None:
        payload = self.payload_at(row)
        if payload is None or self._thumb_key_fn is None:
            return None
        return self._thumb_key_fn(payload)

    def set_thumbnail(self, key: object, pixmap: QtGui.QPixmap) -> None:
        if key is None or pixmap.isNull():
            return
        self._thumbs[key] = pixmap
        for row, _payload in enumerate(self._payloads):
            if self.thumb_key_at(row) == key:
                idx = self.index(row, 0)
                self.dataChanged.emit(idx, idx, [QtCore.Qt.ItemDataRole.DecorationRole])
                break

    def payload_at(self, row: int) -> Any | None:
        if 0 <= row < len(self._payloads):
            return self._payloads[row]
        return None

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._payloads)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        payload = self.payload_at(index.row())
        if payload is None:
            return None
        if role == self.PayloadRole:
            return payload
        if role == self.ThumbnailKeyRole:
            return self.thumb_key_at(index.row())
        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            key = self.thumb_key_at(index.row())
            return self._thumbs.get(key) if key is not None else None
        if role == QtCore.Qt.ItemDataRole.SizeHintRole:
            return QtCore.QSize(280, 104)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if self._title_fn is None:
                return str(payload)
            return self._title_fn(payload)
        if role == QtCore.Qt.ItemDataRole.FontRole and self._title_fn is not None:
            title = str(self._title_fn(payload))
            if "PICK ->" in title or "EXCLUDED" in title:
                font = QtGui.QFont()
                font.setBold(True)
                return font
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and self._title_fn is not None:
            title = str(self._title_fn(payload))
            if "EXCLUDED" in title:
                return QtGui.QBrush(QtGui.QColor("#b00020"))
            if "PICK ->" in title:
                return QtGui.QBrush(QtGui.QColor("#0057b8"))
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return str(getattr(payload, "image_rel_path", "") or getattr(payload, "result_id", ""))
        return None


class PayloadList(QtWidgets.QListView):
    currentPayloadChanged = QtCore.Signal(object)
    visibleRowsChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = PayloadListModel(self)
        self.setModel(self._model)
        self.setAlternatingRowColors(True)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setIconSize(QtCore.QSize(96, 96))
        self.setSpacing(2)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.selectionModel().currentChanged.connect(self._emit_current)
        self.verticalScrollBar().valueChanged.connect(self._emit_visible_rows_later)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._emit_visible_rows_later()

    def set_payloads(self, payloads: list[Any], title_fn: TitleFn, thumb_key_fn: ThumbKeyFn | None = None) -> None:
        self._model.set_payloads(payloads, title_fn, thumb_key_fn)
        if self.count() > 0:
            self.setCurrentRow(0)
        self._emit_visible_rows_later()

    def set_thumbnail(self, key: object, pixmap: QtGui.QPixmap) -> None:
        self._model.set_thumbnail(key, pixmap)

    def visible_rows(self, preload: int = 18) -> list[int]:
        total = self.count()
        if total <= 0:
            return []
        first_index = self.indexAt(QtCore.QPoint(2, 2))
        last_index = self.indexAt(QtCore.QPoint(2, max(2, self.viewport().height() - 2)))
        first = first_index.row() if first_index.isValid() else max(0, self.currentRow())
        last = last_index.row() if last_index.isValid() else min(total - 1, first + preload)
        first = max(0, first - 4)
        last = min(total - 1, last + preload)
        return list(range(first, last + 1))

    def current_payload(self) -> Any | None:
        return self._model.payload_at(self.currentRow())

    def payload_at(self, row: int) -> Any | None:
        return self._model.payload_at(row)

    def currentRow(self) -> int:  # noqa: N802 - keep QListWidget-compatible API
        idx = self.currentIndex()
        return idx.row() if idx.isValid() else -1

    def setCurrentRow(self, row: int) -> None:  # noqa: N802 - keep QListWidget-compatible API
        idx = self._model.index(int(row), 0)
        if idx.isValid():
            self.setCurrentIndex(idx)
            self.scrollTo(idx, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
            self._emit_visible_rows_later()

    def count(self) -> int:
        return self._model.rowCount()

    def _emit_visible_rows_later(self) -> None:
        QtCore.QTimer.singleShot(0, lambda: self.visibleRowsChanged.emit(self.visible_rows()))

    @QtCore.Slot(object, object)
    def _emit_current(self, current: QtCore.QModelIndex, _previous: QtCore.QModelIndex) -> None:
        payload = self._model.payload_at(current.row()) if current.isValid() else None
        if payload is not None:
            self.currentPayloadChanged.emit(payload)


class PayloadGrid(PayloadList):
    """Thumbnail-first grid for prototype / cluster triage."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.setMovement(QtWidgets.QListView.Movement.Static)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.setWrapping(True)
        self.setWordWrap(True)
        self.setUniformItemSizes(True)
        self.setIconSize(QtCore.QSize(148, 148))
        self.setGridSize(QtCore.QSize(190, 210))
        self.setSpacing(8)
        self.setSelectionRectVisible(False)

    def visible_rows(self, preload: int = 60) -> list[int]:
        total = self.count()
        if total <= 0:
            return []
        indexes: set[int] = set()
        step_x = max(40, self.gridSize().width() // 2)
        step_y = max(40, self.gridSize().height() // 2)
        for y in range(2, max(3, self.viewport().height()), step_y):
            for x in range(2, max(3, self.viewport().width()), step_x):
                idx = self.indexAt(QtCore.QPoint(x, y))
                if idx.isValid():
                    indexes.add(idx.row())
        if not indexes:
            current = self.currentRow()
            start = max(0, current if current >= 0 else 0)
            return list(range(start, min(total, start + preload)))
        first = max(0, min(indexes) - 12)
        last = min(total - 1, max(indexes) + preload)
        return list(range(first, last + 1))
