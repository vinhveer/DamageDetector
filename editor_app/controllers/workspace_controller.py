from __future__ import annotations

from pathlib import Path

from editor_app.image_io import ImageIoError, load_image, load_mask, new_blank_mask, save_mask_png_0255
from editor_app.services.file_service import FileService
from editor_app.stores.history_store import HistoryStore
from editor_app.stores.workspace_store import WorkspaceStore


class WorkspaceController:
    def __init__(self, workspace_store: WorkspaceStore, history_store: HistoryStore, file_service: FileService) -> None:
        self._workspace_store = workspace_store
        self._history_store = history_store
        self._file_service = file_service

    def open_folder(self, folder: str, *, auto_open_first: bool = True) -> list[str]:
        root = Path(folder)
        images = self._file_service.list_images(root)
        results_root = root / "_editor_app_runs"
        results_root.mkdir(parents=True, exist_ok=True)
        self._workspace_store.set_workspace(root, results_root, images)
        if auto_open_first and images:
            self.open_image(images[0])
        return images

    def import_folder(self, source_folder: str) -> list[str]:
        workspace_root = self._workspace_store.workspace_root
        results_root = self._workspace_store.results_root
        if workspace_root is None or results_root is None:
            raise ImageIoError("Open a workspace folder before importing images.")
        copied = self._file_service.import_images(source_folder, workspace_root)
        images = self._file_service.list_images(workspace_root)
        self._workspace_store.set_workspace(workspace_root, results_root, images)
        return copied

    def set_result_items(self, items: list[dict], *, adopt_images: bool = False) -> None:
        if adopt_images:
            image_paths = [str(item.get("image_path") or "").strip() for item in items]
            image_paths = [path for path in image_paths if path]
            if image_paths:
                workspace_root = self._workspace_store.workspace_root or Path(image_paths[0]).resolve().parent
                results_root = self._workspace_store.results_root or (workspace_root / "_editor_app_runs")
                self._workspace_store.set_workspace(workspace_root, Path(results_root), image_paths)
        self._workspace_store.set_result_items(items)

    def open_image(self, path: str) -> None:
        image = load_image(path)
        self._workspace_store.set_current_image(path, image)
        blank = new_blank_mask((image.width(), image.height())).mask
        self._workspace_store.clear_mask(blank)
        self._workspace_store.set_detections([])
        self._workspace_store.set_highlight_detections([])
        result_item = self._workspace_store.result_item_for(path)
        if result_item is None:
            return
        detections = list(result_item.get("detections") or [])
        if detections:
            self._workspace_store.set_detections(detections)
            self._workspace_store.set_highlight_detections(detections)
        mask_path = str(result_item.get("mask_path") or "").strip()
        if mask_path:
            try:
                loaded = load_mask(mask_path, (image.width(), image.height()))
            except Exception:
                loaded = None
            if loaded is not None:
                self._workspace_store.set_current_mask(mask_path, loaded.mask)

    def open_mask(self, path: str) -> None:
        image = self._workspace_store.current_image
        if image.isNull():
            raise ImageIoError("Open an image before loading a mask.")
        loaded = load_mask(path, (image.width(), image.height()))
        self._workspace_store.set_current_mask(path, loaded.mask)

    def save_mask(self, path: str) -> None:
        mask = self._workspace_store.current_mask
        if mask.isNull():
            raise ImageIoError("Mask is empty.")
        save_mask_png_0255(path, mask)

    def navigate(self, delta: int) -> str | None:
        images = self._workspace_store.images
        if not images:
            return None
        if self._workspace_store.current_index < 0:
            next_index = 0
        else:
            next_index = max(0, min(len(images) - 1, self._workspace_store.current_index + int(delta)))
        next_path = images[next_index]
        self.open_image(next_path)
        return next_path
