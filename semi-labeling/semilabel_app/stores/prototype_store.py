from __future__ import annotations

from PySide6 import QtCore


class PrototypeStore(QtCore.QObject):
    candidatesChanged = QtCore.Signal()
    picksChanged = QtCore.Signal()
    statusChanged = QtCore.Signal(str)
    errorRaised = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.candidates: list[object] = []
        self.picks: dict[int, dict] = {}
        self.latest_prototype: dict | None = None

    def set_candidates(self, items: list[object], latest: dict | None = None) -> None:
        self.candidates = list(items)
        self.latest_prototype = latest
        self.candidatesChanged.emit()

    def toggle_pick(self, result_id: int, label: str, is_reject: bool = False) -> None:
        key = int(result_id)
        current = self.picks.get(key)
        next_value = {"label": "reject" if is_reject else str(label), "is_reject": bool(is_reject)}
        if current == next_value:
            self.picks.pop(key, None)
        else:
            self.picks[key] = next_value
        self.picksChanged.emit()
