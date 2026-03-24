from __future__ import annotations

from PySide6 import QtCore

from editor_app.domain.models import RunSummary


class HistoryStore(QtCore.QObject):
    runsChanged = QtCore.Signal()
    selectedRunChanged = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.runs: list[RunSummary] = []
        self.selected_run_id: str | None = None

    def set_runs(self, runs: list[RunSummary]) -> None:
        self.runs = list(runs)
        self.runsChanged.emit()

    def set_selected_run(self, run_id: str | None) -> None:
        self.selected_run_id = str(run_id) if run_id else None
        self.selectedRunChanged.emit(self.selected_run_id or "")
