from __future__ import annotations

from pathlib import Path

from editor_app.services.compare_service import CompareService
from editor_app.stores.compare_store import CompareStore
from editor_app.stores.history_store import HistoryStore


class CompareController:
    def __init__(self, compare_store: CompareStore, history_store: HistoryStore, compare_service: CompareService) -> None:
        self._compare_store = compare_store
        self._history_store = history_store
        self._compare_service = compare_service
        self._history_store.runsChanged.connect(self.refresh_runs)

    def refresh_runs(self) -> None:
        runs = [(f"{run.run_id} | {run.workflow} | {run.status}", run.run_dir) for run in self._history_store.runs]
        self._compare_store.set_runs(runs)

    def set_selected_run_dir(self, run_dir: str | None) -> None:
        self._compare_store.set_selected_run_dir(run_dir)

    def set_ground_truth_dir(self, gt_dir: str) -> None:
        self._compare_store.set_gt_dir(gt_dir)

    def set_affix(self, affix: str) -> None:
        self._compare_store.set_affix(affix)

    def run_compare(self, *, run_dir: str | None = None) -> list[dict]:
        run_dir_value = str(run_dir or self._compare_store.selected_run_dir or "").strip()
        if not run_dir_value:
            raise ValueError("Select a run first.")
        gt_dir_value = str(self._compare_store.gt_dir or "").strip()
        if not gt_dir_value:
            raise ValueError("Select a ground-truth folder first.")
        run_path = Path(run_dir_value)
        gt_path = Path(gt_dir_value)
        if not run_path.is_dir():
            raise ValueError("Select a valid run first.")
        if not gt_path.is_dir():
            raise ValueError("Select a valid ground-truth folder.")
        results = self._compare_service.compare_run(
            run_dir=run_path,
            gt_dir=gt_path,
            affix=self._compare_store.affix,
        )
        self._compare_store.set_results(results)
        return results
