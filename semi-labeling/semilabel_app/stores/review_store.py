from __future__ import annotations

from PySide6 import QtCore


class ReviewStore(QtCore.QObject):
    queueChanged = QtCore.Signal()
    cleanedChanged = QtCore.Signal()
    selectionChanged = QtCore.Signal()
    pendingChanged = QtCore.Signal()
    errorRaised = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.queue_items: list[object] = []
        self.cleaned_items: list[object] = []
        self.current_index = 0
        self.pending_decisions: dict[int, dict] = {}
        self.pending_corrections: dict[int, dict] = {}
        self.mode = "queue"

    def set_queue(self, items: list[object]) -> None:
        self.queue_items = list(items)
        self.current_index = 0
        self.queueChanged.emit()
        self.selectionChanged.emit()

    def set_cleaned(self, items: list[object]) -> None:
        self.cleaned_items = list(items)
        self.current_index = 0
        self.cleanedChanged.emit()
        self.selectionChanged.emit()

    def set_index(self, index: int, mode: str | None = None) -> None:
        if mode:
            self.mode = mode
        items = self.queue_items if self.mode == "queue" else self.cleaned_items
        if not items:
            self.current_index = 0
        else:
            self.current_index = max(0, min(int(index), len(items) - 1))
        self.selectionChanged.emit()

    def current_item(self) -> object | None:
        items = self.queue_items if self.mode == "queue" else self.cleaned_items
        if 0 <= self.current_index < len(items):
            return items[self.current_index]
        return None

    def set_decision(self, result_id: int, decision: dict) -> None:
        self.pending_decisions[int(result_id)] = dict(decision)
        self.pendingChanged.emit()

    def set_correction(self, result_id: int, correction: dict) -> None:
        self.pending_corrections[int(result_id)] = dict(correction)
        self.pendingChanged.emit()

    def clear_pending_decisions(self) -> None:
        self.pending_decisions.clear()
        self.pendingChanged.emit()

    def clear_pending_corrections(self) -> None:
        self.pending_corrections.clear()
        self.pendingChanged.emit()
