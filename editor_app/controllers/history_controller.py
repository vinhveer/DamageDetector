from __future__ import annotations

from pathlib import Path

from editor_app.services.run_storage import RunStorageService
from editor_app.stores.history_store import HistoryStore
from editor_app.stores.workspace_store import WorkspaceStore


class HistoryController:
    def __init__(self, workspace_store: WorkspaceStore, history_store: HistoryStore, run_storage: RunStorageService) -> None:
        self._workspace_store = workspace_store
        self._history_store = history_store
        self._run_storage = run_storage
        self._run_bundle_cache: dict[str, dict] = {}
        self._run_items_cache: dict[str, list[dict]] = {}

    def refresh(self) -> None:
        results_root = self._workspace_store.results_root
        if results_root is None:
            self._history_store.set_runs([])
            self._run_bundle_cache.clear()
            self._run_items_cache.clear()
            return
        runs = self._run_storage.list_runs(Path(results_root))
        self._history_store.set_runs(runs)
        valid_run_dirs = {run.run_dir for run in runs}
        self._run_bundle_cache.clear()
        self._run_items_cache.clear()
        if self._history_store.selected_run_dir not in valid_run_dirs:
            self._history_store.set_selected_run(None)

    def load_run_details(self, run_dir: str) -> tuple[dict, list[dict]]:
        normalized = str(run_dir or "").strip()
        if not normalized:
            return {}, []
        bundle = self._run_bundle_cache.get(normalized)
        if bundle is None:
            bundle = self._run_storage.load_run_bundle(Path(normalized))
            self._run_bundle_cache[normalized] = bundle
        items = self._run_items_cache.get(normalized)
        if items is None:
            items = self._run_storage.list_result_items(Path(normalized))
            self._run_items_cache[normalized] = items
        self._history_store.set_selected_run(normalized)
        return dict(bundle or {}), list(items or [])
