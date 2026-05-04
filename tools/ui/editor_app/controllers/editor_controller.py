from __future__ import annotations

from pathlib import Path

from ui.editor_app.controllers.workspace_controller import WorkspaceController
from ui.editor_app.services.run_storage import RunStorageService
from ui.editor_app.stores.workspace_store import WorkspaceStore


class EditorController:
    def __init__(
        self,
        workspace_store: WorkspaceStore,
        workspace_controller: WorkspaceController,
        run_storage: RunStorageService,
    ) -> None:
        self._workspace_store = workspace_store
        self._workspace_controller = workspace_controller
        self._run_storage = run_storage

    def add_roi(self, roi_box: tuple[int, int, int, int]) -> None:
        self._workspace_store.add_roi(tuple(int(x) for x in roi_box))

    def update_roi(self, index: int, roi_box: tuple[int, int, int, int]) -> None:
        self._workspace_store.update_roi(int(index), tuple(int(x) for x in roi_box))

    def delete_roi(self, index: int) -> None:
        self._workspace_store.remove_roi(int(index))

    def clear_rois(self) -> None:
        self._workspace_store.clear_rois()

    def select_roi(self, index: int) -> None:
        self._workspace_store.select_roi_index(int(index))

    def load_history_item(self, payload: dict) -> str:
        image_path = str(payload.get("image_path") or "")
        if not image_path:
            raise ValueError("Selected run item has no image_path.")
        run_dir = str(payload.get("_run_dir") or "").strip()
        if run_dir and Path(run_dir).is_dir():
            items = self._run_storage.list_result_items(Path(run_dir))
            self._workspace_controller.set_result_items(items, adopt_images=True)
        self._workspace_controller.open_image(image_path)
        return image_path
