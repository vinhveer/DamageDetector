from __future__ import annotations

from pathlib import Path

from editor_app.services.run_storage import RunStorageService
from editor_app.stores.history_store import HistoryStore
from editor_app.stores.isolate_store import IsolateStore
from editor_app.stores.workspace_store import WorkspaceStore


class IsolateController:
    def __init__(
        self,
        workspace_store: WorkspaceStore,
        history_store: HistoryStore,
        isolate_store: IsolateStore,
        run_storage: RunStorageService,
    ) -> None:
        self._workspace_store = workspace_store
        self._history_store = history_store
        self._isolate_store = isolate_store
        self._run_storage = run_storage
        self._history_store.runsChanged.connect(self.refresh)

    def refresh(self) -> None:
        results_root = self._workspace_store.results_root
        if results_root is None:
            self._isolate_store.set_items([])
            return
        self._isolate_store.set_items(self._run_storage.list_isolate_items(Path(results_root)))
