from __future__ import annotations

from pathlib import Path

from editor_app.controllers.workspace_controller import WorkspaceController
from editor_app.stores.workspace_store import WorkspaceStore


class EditorController:
    def __init__(self, workspace_store: WorkspaceStore, workspace_controller: WorkspaceController) -> None:
        self._workspace_store = workspace_store
        self._workspace_controller = workspace_controller

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
        self._workspace_controller.open_image(image_path)
        mask_path = str(payload.get("mask_path") or "")
        if mask_path and Path(mask_path).is_file():
            self._workspace_controller.open_mask(mask_path)
        detections = list(payload.get("detections") or [])
        self._workspace_store.set_detections(detections)
        self._workspace_store.set_highlight_detections(detections)
        return image_path
