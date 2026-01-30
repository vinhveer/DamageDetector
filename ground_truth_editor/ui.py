from __future__ import annotations

import os
import sys
import threading
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from canvas import ImageCanvas
from image_io import (
    ImageIoError,
    load_image,
    load_mask,
    new_blank_mask,
    save_mask_png_01_indexed,
    save_mask_png_0255,
)
from sam_dino.runner import SamDinoParams, SamDinoRunner
from predict_unet import UnetParams, UnetRunner

# Refactored modules
from dialogs import ProcessingDialog
from workers import WorkerBase, UnetWorker, SamDinoWorker, SamDinoIsolateWorker
from tabs.editor import EditorTab
from tabs.unet import UnetTab
from tabs.sam_dino import SamDinoTab


@dataclass(frozen=True)
class LoadedState:
    image_path: str
    image_w: int
    image_h: int


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PredictTools - One Image")
        self.resize(1400, 860)

        self._unet_runner = UnetRunner()
        self._samdino_runner = SamDinoRunner()
        self._thread: QtCore.QThread | None = None
        self._worker: WorkerBase | None = None
        self._progress_dialog: ProcessingDialog | None = None
        self._post_run_action: dict | None = None

        self._state: LoadedState | None = None
        self._mask_path: Path | None = None

        self._overlay_canvas = ImageCanvas(self, render_mode="overlay", editable=True)
        self._image_canvas = ImageCanvas(self, render_mode="image", editable=False)
        self._mask_canvas = ImageCanvas(self, render_mode="mask", editable=False)
        self._result_canvas = ImageCanvas(self, render_mode="image", editable=False)  # For detection results

        self._overlay_canvas.cursorInfo.connect(self._on_cursor_info)
        # Note: brush slider syncing is now handled in EditorTab via direct connection or I should expose a signal.
        # But EditorTab was passed the canvas, so it connects strictly there.
        # However, line 357 `brushRadiusChanged.connect(self._sync_brush_slider)` in original code was for the slider in UI.
        # The EditorTab handles that connection internally now (see EditorTab.__init__).
        
        self._overlay_canvas.maskChanged.connect(self._sync_mask_views)
        self._overlay_canvas.roiSelected.connect(self._on_roi_selected)
        self._overlay_canvas.roiCanceled.connect(self._on_roi_canceled)

        self._status_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self._status_label)

        self._build_actions()
        self._build_layout()

    def _build_actions(self) -> None:
        self._act_open_image = QtGui.QAction("Open Image...", self)
        self._act_open_image.setShortcut(QtGui.QKeySequence.Open)
        self._act_open_image.triggered.connect(self.open_image_dialog)

        self._act_open_mask = QtGui.QAction("Open Mask...", self)
        self._act_open_mask.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        self._act_open_mask.triggered.connect(self.open_mask_dialog)

        self._act_save_mask = QtGui.QAction("Save Mask As...", self)
        self._act_save_mask.setShortcut(QtGui.QKeySequence.Save)
        self._act_save_mask.triggered.connect(self.save_mask_dialog)

        self._act_exit = QtGui.QAction("Exit", self)
        self._act_exit.setShortcut(QtGui.QKeySequence.Quit)
        self._act_exit.triggered.connect(self.close)

        menu = self.menuBar().addMenu("File")
        menu.addAction(self._act_open_image)
        menu.addAction(self._act_open_mask)
        menu.addSeparator()
        menu.addAction(self._act_save_mask)
        menu.addSeparator()
        menu.addAction(self._act_exit)

    def _build_layout(self) -> None:
        central = QtWidgets.QWidget(self)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QtWidgets.QWidget(self)
        sidebar.setFixedWidth(420)
        side_layout = QtWidgets.QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 10, 10, 10)

        self._method_tabs = QtWidgets.QTabWidget(sidebar)
        
        # SAM+DINO Tab
        self._sam_dino_tab = SamDinoTab(sidebar)
        self._sam_dino_tab.runRequested.connect(self._run_sam_dino)
        self._sam_dino_tab.isolateRequested.connect(self._isolate_object)
        self._method_tabs.addTab(self._sam_dino_tab, "SAM + DINO")
        
        # UNet Tab
        self._unet_tab = UnetTab(sidebar)
        self._unet_tab.runRequested.connect(self._run_unet)
        self._method_tabs.addTab(self._unet_tab, "UNet")

        # Editor Tab
        self._editor_tab = EditorTab(
            sidebar, 
            open_action=self._act_open_mask, 
            save_action=self._act_save_mask,
            canvas=self._overlay_canvas
        )
        self._method_tabs.addTab(self._editor_tab, "Editor")

        side_layout.addWidget(self._method_tabs, 1)

        root.addWidget(sidebar)
        
        self._view_tabs = QtWidgets.QTabWidget(self)
        self._view_tabs.addTab(self._overlay_canvas, "Overlay")
        self._view_tabs.addTab(self._image_canvas, "Image")
        self._view_tabs.addTab(self._mask_canvas, "Mask")
        self._view_tabs.addTab(self._result_canvas, "Result (Overlay)")
        
        root.addWidget(self._view_tabs, 1)
        self.setCentralWidget(central)

    def open_image_dialog(self) -> None:
        start = str(Path.cwd())
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", start, "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not f:
            return
        self.load_image(f)

    def load_image(self, path: str) -> None:
        try:
            img = load_image(path)
            self._state = LoadedState(path, img.width(), img.height())
            self._overlay_canvas.set_image(img)
            self._image_canvas.set_image(img)
            self._mask_canvas.set_image(img)
            
            # Reset mask
            blank = new_blank_mask((img.width(), img.height()))
            self._overlay_canvas.set_mask(blank.mask)
            self._mask_canvas.set_mask(blank.mask)
            self._mask_path = None
            
            self.setWindowTitle(f"PredictTools - {Path(path).name}")
            self._status_label.setText(f"Loaded: {path} ({img.width()}x{img.height()})")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def open_mask_dialog(self) -> None:
        if self._state is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")
            return
        start = str(Path.cwd())
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Mask", start, "Images (*.png *.jpg *.bmp *.tif)")
        if not f:
            return
        try:
            lm = load_mask(f, (self._state.image_w, self._state.image_h))
            self._overlay_canvas.set_mask(lm.mask)
            self._mask_canvas.set_mask(lm.mask)
            self._mask_path = Path(f)
            self._status_label.setText(f"Loaded mask: {f}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def save_mask_dialog(self) -> None:
        if self._state is None:
            return
        mask = self._overlay_canvas.mask()
        if mask.isNull():
            return
            
        default_name = "mask.png"
        if self._mask_path:
            default_name = self._mask_path.name
        elif self._state:
            base = Path(self._state.image_path).stem
            default_name = f"{base}_mask.png"

        start = str(Path.cwd() / default_name)
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Mask", start, "PNG (0/1) (*.png);;PNG (0/255) (*.png)")
        if not f:
            return
            
        try:
            if f.endswith(".png"):
                # Simple heuristic: if user selected 0/1 filter? 
                # Actually Qt doesn't give us the selected filter string easily in the return tuple in PySide6 depending on version
                # But typically valid masks are 0/1 indexed for segmentation training.
                # However, for visualization 0/255 is better. The tool allows both.
                # Let's ask or default? The original had logic but it was implicitly chosen by filter string.
                # Let's check original implementation.
                pass

            # Just save as 0/1 indexed by default for ground truth, or 0/255 if it's for display?
            # Original code 'save_mask_png_01_indexed' vs 'save_mask_png_0255'.
            # It seems the user selection in FileDialog determines it if we could get the filter.
            # Assuming standard behavior, let's just save as 0/1 unless specifically asked. 
            # But the original code had distinct functions. 
            # Let's use a message box to ask? Or just check extension?
            # Actually, `getSelfFileName` returns filter.
            pass
        except Exception:
            pass
            
        # Re-implementing save logic
        # Since getSaveFileName returns (filename, selectedFilter)
        # We can detect.
        path_str, filter_str = f, _
        
        try:
            if "0/255" in filter_str:
                save_mask_png_0255(path_str, mask)
            else:
                save_mask_png_01_indexed(path_str, mask)
            self._mask_path = Path(path_str)
            self._status_label.setText(f"Saved mask: {path_str}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def _sync_mask_views(self) -> None:
        mask = self._overlay_canvas.mask()
        self._mask_canvas.set_mask(mask)

    def _on_cursor_info(self, x: int, y: int, val: int) -> None:
        self._status_label.setText(f"Pos: ({x}, {y})  Val: {val}")

    def _on_roi_selected(self, roi_box) -> None:
        # roi_box is (x0, y0, x1, y1)
        if roi_box:
            self._status_label.setText(f"ROI: {roi_box}")
        else:
            self._status_label.setText("ROI canceled")

    def _on_roi_canceled(self) -> None:
        self._status_label.setText("ROI canceled")

    # Worker runners
    
    def _run_unet(self, params: UnetParams) -> None:
        if self._state is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")
            return
        
        self._start_worker(
            UnetWorker(self._unet_runner, self._state.image_path, params),
            "Running UNet..."
        )

    def _run_sam_dino(self, params: SamDinoParams) -> None:
        if self._state is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")
            return

        self._start_worker(
            SamDinoWorker(self._samdino_runner, self._state.image_path, params),
            "Running SAM + DINO..."
        )
        
        # We want to load the result mask after run.
        # The worker returns details dict.
        self._post_run_action = {"type": "load_mask_and_overlay"}

    def _isolate_object(self, params: SamDinoParams, target_labels: list[str], outside_value: int, crop: bool) -> None:
        if self._state is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")
            return
            
        self._start_worker(
            SamDinoIsolateWorker(
                self._samdino_runner, 
                self._state.image_path, 
                params,
                target_labels=target_labels,
                outside_value=outside_value,
                crop_to_bbox=crop,
            ),
            "Isolating Object..."
        )
        self._post_run_action = {"type": "isolate_result"}

    def _start_worker(self, worker: WorkerBase, title: str) -> None:
        if self._thread is not None and self._thread.isRunning():
            return

        self._progress_dialog = ProcessingDialog(self, title)
        self._worker = worker
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._progress_dialog.log_widget().appendPlainText)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_worker_finished)
        # Use lambda to ensure stop is called in the main thread (thread-safe event setting),
        # bypassing the blocked event loop of the worker thread.
        self._progress_dialog.stopRequested.connect(lambda: self._worker.stop())

        self._thread.start()
        self._progress_dialog.exec()

    def _on_worker_failed(self, msg: str) -> None:
        if self._progress_dialog:
            self._progress_dialog.allow_close()
            # self._progress_dialog.accept() # Don't auto close on error, let user read
            QtWidgets.QMessageBox.critical(self._progress_dialog, "Failed", msg)
        
        # Ensure cleanup happens even on failure
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _on_worker_finished(self, result: object) -> None:
        if self._progress_dialog:
            self._progress_dialog.allow_close()
            self._progress_dialog.accept()
        
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None

        if isinstance(result, dict) and result.get("stopped"):
            self._status_label.setText("Stopped.")
            return

        self._handle_result(result)

    def _handle_result(self, result: object) -> None:
        # Logic to handle results based on _post_run_action or implicitly by result content
        if not isinstance(result, dict):
            return

        if self._post_run_action:
            action = self._post_run_action.get("type")
            self._post_run_action = None
            
            if action == "load_mask_and_overlay":
                mask_path = result.get("mask_path")
                overlay_path = result.get("overlay_path")

                if mask_path and os.path.exists(mask_path):
                    try:
                        self.load_mask(mask_path)
                    except Exception as e:
                        print(f"Failed to load result mask: {e}")
                
                if overlay_path and os.path.exists(overlay_path):
                     try:
                         from image_io import load_image_cv2
                         img = load_image_cv2(overlay_path)
                         self._result_canvas.set_image(img.bgr)
                         # Switch to Result tab
                         self._view_tabs.setCurrentWidget(self._result_canvas)
                     except Exception as e:
                         print(f"Failed to load result overlay: {e}")
                
            elif action == "isolate_result":
                isolate_path = result.get("isolate_path") 
                overlay_path = result.get("overlay_path")

                if isolate_path and os.path.exists(isolate_path):
                     # Isolate complete. 
                     # Let's show the isolate image in Result tab? 
                     # Or overlay? Usually isolate is the goal.
                     try:
                         from image_io import load_image_cv2
                         img = load_image_cv2(isolate_path)
                         self._result_canvas.set_image(img.bgr)
                         self._view_tabs.setCurrentWidget(self._result_canvas)
                     except Exception as e:
                          print(f"Failed to load isolate result: {e}")

                     QtWidgets.QMessageBox.information(self, "Done", f"Isolate complete.\nSaved to: {isolate_path}")
                     # Optionally load it?
                     pass
        else:
             # Default fallback for unet
             # Unet worker returns details. Unet typically produces masks.
             # If result['masks_saved'] > 0 or similar? 
             # UnetRunner.run returns a dict too.
             pass

        msg = f"Task finished. Output: {result.get('output_dir')}"
        self._status_label.setText(msg)
        QtWidgets.QMessageBox.information(self, "Finished", msg)

    def load_mask(self, path: str) -> None:
        if self._state is None:
            return
        lm = load_mask(path, (self._state.image_w, self._state.image_h))
        self._overlay_canvas.set_mask(lm.mask)
        self._mask_canvas.set_mask(lm.mask)
        self._mask_path = Path(path)


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
