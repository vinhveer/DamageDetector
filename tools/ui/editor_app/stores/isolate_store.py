from __future__ import annotations

from PySide6 import QtCore


class IsolateStore(QtCore.QObject):
    itemsChanged = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.items: list[dict] = []

    def set_items(self, items: list[dict]) -> None:
        self.items = [dict(item) for item in items]
        self.itemsChanged.emit()
