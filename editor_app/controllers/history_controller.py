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

    def refresh(self) -> None:
        results_root = self._workspace_store.results_root
        if results_root is None:
            self._history_store.set_runs([])
            return
        self._history_store.set_runs(self._run_storage.list_runs(Path(results_root)))
