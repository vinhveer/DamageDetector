from __future__ import annotations

from PySide6 import QtCore

from ui.editor_app.domain.models import RunSummary


class HistoryStore(QtCore.QObject):
    runsChanged = QtCore.Signal()
    selectedRunChanged = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.runs: list[RunSummary] = []
        self.selected_run_dir: str | None = None

    def set_runs(self, runs: list[RunSummary]) -> None:
        self.runs = list(runs)
        self.runsChanged.emit()

    def set_selected_run(self, run_dir: str | None) -> None:
        normalized = str(run_dir) if run_dir else None
        if normalized == self.selected_run_dir:
            return
        self.selected_run_dir = normalized
        self.selectedRunChanged.emit(self.selected_run_dir or "")
