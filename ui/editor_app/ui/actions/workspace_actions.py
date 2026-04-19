from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ui.editor_app.controllers.history_controller import HistoryController
from ui.editor_app.controllers.isolate_controller import IsolateController
from ui.editor_app.controllers.workspace_controller import WorkspaceController
from ui.editor_app.paths import repo_root
from ui.editor_app.services.export_service import ExportService
from ui.editor_app.stores.workspace_store import WorkspaceStore
from ui.editor_app.ui.widgets.left_rail import LeftRail
from ui.editor_app.ui.workspaces.editor_workspace import EditorWorkspace


class WorkspaceActions(QtCore.QObject):
    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget,
        workspace_store: WorkspaceStore,
        workspace_controller: WorkspaceController,
        history_controller: HistoryController,
        isolate_controller: IsolateController,
        left_rail: LeftRail,
        editor_workspace: EditorWorkspace,
        export_service: ExportService,
        show_workspace: Callable[[str], None],
        show_error: Callable[[str], None],
        show_status: Callable[[str, int], None],
        persist_state: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._workspace_store = workspace_store
        self._workspace_controller = workspace_controller
        self._history_controller = history_controller
        self._isolate_controller = isolate_controller
        self._left_rail = left_rail
        self._editor_workspace = editor_workspace
        self._export_service = export_service
        self._show_workspace = show_workspace
        self._show_error = show_error
        self._show_status = show_status
        self._persist_state = persist_state

    def open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self._parent, "Open Workspace Folder")
        if not folder:
            return
        images = self._workspace_controller.open_folder(folder)
        self._left_rail.explorer().set_images(images)
        self._history_controller.refresh()
        self._isolate_controller.refresh()
        self._persist_state()
        self._show_status(f"Loaded workspace: {folder}", 5000)

    def add_folder_images(self) -> None:
        if self._workspace_store.workspace_root is None:
            self._show_error("Open a workspace folder before importing images.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self._parent, "Add Folder Images To Workspace")
        if not folder:
            return
        try:
            copied = self._workspace_controller.import_folder(folder)
        except Exception as exc:
            self._show_error(str(exc))
            return
        images = list(self._workspace_store.images)
        self._left_rail.explorer().set_images(images)
        self._history_controller.refresh()
        self._isolate_controller.refresh()
        self._persist_state()
        if copied:
            self._show_status(f"Imported {len(copied)} image(s) into workspace.", 5000)
            return
        self._show_status("No supported images found in the selected folder.", 5000)

    def open_image_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._parent,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)",
        )
        if not path:
            return
        self.load_image(path, switch_workspace=True)

    def load_image(self, path: str, *, switch_workspace: bool) -> None:
        try:
            self._workspace_controller.open_image(path)
        except Exception as exc:
            self._show_error(str(exc))
            return
        if path not in self._workspace_store.images:
            images = list(self._workspace_store.images)
            images.append(path)
            workspace_root = Path(path).resolve().parent
            results_root = self._workspace_store.results_root or (workspace_root / "_editor_app_runs")
            Path(results_root).mkdir(parents=True, exist_ok=True)
            self._workspace_store.set_workspace(workspace_root, Path(results_root), images)
            self._left_rail.explorer().set_images(images)
        else:
            self._left_rail.explorer().select_path(path)
        self._history_controller.refresh()
        self._isolate_controller.refresh()
        self._persist_state()
        if switch_workspace:
            self._show_workspace("editor")

    def open_mask(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._parent,
            "Open Mask",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)",
        )
        if not path:
            return
        try:
            self._workspace_controller.open_mask(path)
        except Exception as exc:
            self._show_error(str(exc))

    def save_mask(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._parent,
            "Save Mask",
            "",
            "PNG Files (*.png);;All files (*)",
        )
        if not path:
            return
        try:
            self._workspace_controller.save_mask(path)
            self._show_status(f"Saved mask: {path}", 5000)
        except Exception as exc:
            self._show_error(str(exc))

    def open_local_folder(self, path: str) -> None:
        if not path:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(path)))

    def navigate_image(self, delta: int) -> None:
        next_path = self._workspace_controller.navigate(delta)
        if not next_path:
            return
        self._left_rail.explorer().select_path(next_path)
        self._show_workspace("editor")

    def export_image(self) -> None:
        image = self._editor_workspace.image_canvas().image()
        if image.isNull():
            image = self._editor_workspace.overlay_canvas().image()
        stem = Path(self._workspace_store.current_image_path).stem if self._workspace_store.current_image_path else "image"
        self._save_qimage_dialog(title="Export Image", suggested_name=f"{stem}_image.png", image=image)

    def export_overlay(self) -> None:
        stem = Path(self._workspace_store.current_image_path).stem if self._workspace_store.current_image_path else "image"
        image = self._export_service.build_overlay_image(
            base_image=self._editor_workspace.overlay_canvas().image(),
            overlay_visual=self._editor_workspace.overlay_canvas().overlay_visual(),
            mask_image=self._editor_workspace.overlay_canvas().mask(),
            overlay_opacity=self._editor_workspace.overlay_canvas().canvas_state().overlay_opacity,
        )
        self._save_qimage_dialog(title="Export Overlay", suggested_name=f"{stem}_overlay.png", image=image)

    def export_overlay_boxes(self) -> None:
        stem = Path(self._workspace_store.current_image_path).stem if self._workspace_store.current_image_path else "image"
        image = self._export_service.build_overlay_boxes_image(
            base_image=self._editor_workspace.overlay_canvas().image(),
            overlay_visual=self._editor_workspace.overlay_canvas().overlay_visual(),
            mask_image=self._editor_workspace.overlay_canvas().mask(),
            overlay_opacity=self._editor_workspace.overlay_canvas().canvas_state().overlay_opacity,
            detections=self._workspace_store.highlight_detections or self._workspace_store.current_detections,
        )
        self._save_qimage_dialog(
            title="Export Overlay + Boxes",
            suggested_name=f"{stem}_overlay_boxes.png",
            image=image,
        )

    def _save_qimage_dialog(self, *, title: str, suggested_name: str, image: QtGui.QImage) -> None:
        if image.isNull():
            self._show_error(f"No image available for {title}.")
            return
        default_dir = Path(self._workspace_store.current_image_path).parent if self._workspace_store.current_image_path else repo_root()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._parent,
            title,
            str(default_dir / suggested_name),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return
        if not image.save(path):
            self._show_error(f"Failed to save image: {path}")
            return
        self._show_status(f"Saved: {path}", 5000)
