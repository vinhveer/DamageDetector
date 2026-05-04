from __future__ import annotations

from PySide6 import QtCore


class CompareStore(QtCore.QObject):
    runsChanged = QtCore.Signal()
    configChanged = QtCore.Signal()
    resultsChanged = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.runs: list[tuple[str, str]] = []
        self.selected_run_dir: str | None = None
        self.gt_dir: str = ""
        self.affix: str = ""
        self.results: list[dict] = []

    def set_runs(self, runs: list[tuple[str, str]]) -> None:
        self.runs = list(runs)
        if self.selected_run_dir:
            known = {run_dir for _label, run_dir in self.runs}
            if self.selected_run_dir not in known:
                self.selected_run_dir = None
        if not self.selected_run_dir and self.runs:
            self.selected_run_dir = str(self.runs[0][1] or "")
        self.runsChanged.emit()

    def set_selected_run_dir(self, run_dir: str | None) -> None:
        self.selected_run_dir = str(run_dir) if run_dir else None
        self.configChanged.emit()

    def set_gt_dir(self, gt_dir: str) -> None:
        self.gt_dir = str(gt_dir or "")
        self.configChanged.emit()

    def set_affix(self, affix: str) -> None:
        self.affix = str(affix or "")
        self.configChanged.emit()

    def set_results(self, results: list[dict]) -> None:
        self.results = [dict(result) for result in results]
        self.resultsChanged.emit()
