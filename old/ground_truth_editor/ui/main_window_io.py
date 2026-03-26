from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from image_io import (
    ImageIoError,
    load_image,
    load_mask,
    new_blank_mask,
    save_mask_png_01_indexed,
    save_mask_png_0255,
)

from .io_worker import ImageIoWorker
from .state import LoadedState


class MainWindowIOMixin:
    def _run_with_loading(self, title: str, message: str, func):
        dlg = QtWidgets.QProgressDialog(message, "", 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dlg.setValue(0)
        dlg.show()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            return func()
        finally:
            dlg.close()
            dlg.deleteLater()
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QApplication.processEvents()

    def _show_async_loading(self, title: str, message: str) -> None:
        dlg = QtWidgets.QProgressDialog(message, "", 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dlg.setValue(0)
        dlg.show()
        self._io_progress_dialog = dlg
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents()

    def _hide_async_loading(self) -> None:
        dlg = getattr(self, "_io_progress_dialog", None)
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()
            self._io_progress_dialog = None
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QApplication.processEvents()

    def _start_async_io_job(
        self,
        *,
        kind: str,
        path: str,
        switch_tab: bool = True,
        expected_size: tuple[int, int] | None = None,
    ) -> None:
        current_thread = getattr(self, "_io_thread", None)
        if current_thread is not None and current_thread.isRunning():
            self.statusBar().showMessage("Image or mask is still loading...", 3000)
            return

        self._active_io_job = {
            "kind": str(kind),
            "path": str(path),
            "switch_tab": bool(switch_tab),
            "expected_size": expected_size,
        }
        self._show_async_loading(
            "Open Image" if str(kind) == "image" else "Open Mask",
            "Loading image..." if str(kind) == "image" else "Loading mask...",
        )

        class IoSignalBroker(QtCore.QObject):
            finished = QtCore.Signal(object)
            failed = QtCore.Signal(str)
            cleanup = QtCore.Signal()

        thread = QtCore.QThread(self)
        worker = ImageIoWorker(str(kind), str(path), expected_size)
        broker = IoSignalBroker(self)
        worker.moveToThread(thread)
        broker.finished.connect(self._on_async_io_finished, type=QtCore.Qt.ConnectionType.QueuedConnection)
        broker.failed.connect(self._on_async_io_failed, type=QtCore.Qt.ConnectionType.QueuedConnection)
        broker.cleanup.connect(self._cleanup_async_io_job, type=QtCore.Qt.ConnectionType.QueuedConnection)
        thread.started.connect(worker.run, type=QtCore.Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(broker.finished, type=QtCore.Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(broker.failed, type=QtCore.Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(thread.quit, type=QtCore.Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(thread.quit, type=QtCore.Qt.ConnectionType.QueuedConnection)
        thread.finished.connect(worker.deleteLater, type=QtCore.Qt.ConnectionType.QueuedConnection)
        thread.finished.connect(thread.deleteLater, type=QtCore.Qt.ConnectionType.QueuedConnection)
        thread.finished.connect(broker.cleanup, type=QtCore.Qt.ConnectionType.QueuedConnection)
        self._io_thread = thread
        self._io_worker = worker
        self._io_broker = broker
        thread.start()

    @QtCore.Slot(object)
    def _on_async_io_finished(self, payload_obj) -> None:
        payload = dict(payload_obj or {})
        job = dict(getattr(self, "_active_io_job", {}) or {})
        try:
            kind = str(payload.get("kind") or job.get("kind") or "")
            if kind == "image":
                image = payload.get("image")
                if not isinstance(image, QtGui.QImage) or image.isNull():
                    raise ImageIoError(f"Không mở được ảnh: {job.get('path')}")
                self._apply_loaded_image(str(payload.get("path") or job.get("path") or ""), image, bool(job.get("switch_tab", True)))
            elif kind == "mask":
                mask = payload.get("mask")
                if not isinstance(mask, QtGui.QImage) or mask.isNull():
                    raise ImageIoError(f"Không mở được mask: {job.get('path')}")
                self._apply_loaded_mask(str(payload.get("path") or job.get("path") or ""), mask)
        except ImageIoError as e:
            QtWidgets.QMessageBox.critical(self, "Open Image" if job.get("kind") == "image" else "Open Mask", str(e))
        finally:
            self._hide_async_loading()

    @QtCore.Slot(str)
    def _on_async_io_failed(self, message: str) -> None:
        job = dict(getattr(self, "_active_io_job", {}) or {})
        self._hide_async_loading()
        QtWidgets.QMessageBox.critical(
            self,
            "Open Image" if job.get("kind") == "image" else "Open Mask",
            str(message or "Unknown IO error."),
        )

    @QtCore.Slot()
    def _cleanup_async_io_job(self) -> None:
        self._io_thread = None
        self._io_worker = None
        self._io_broker = None
        self._active_io_job = None

    def _apply_loaded_image(self, path: str, img: QtGui.QImage, switch_tab: bool = True) -> None:
        prev_image_path = self._state.image_path if self._state is not None else ""
        self._state = LoadedState(image_path=path, image_w=img.width(), image_h=img.height())
        self._overlay_canvas.set_image(img)
        self._image_canvas.set_image(img)

        blank = new_blank_mask((img.width(), img.height())).mask
        self._overlay_canvas.set_mask(blank)
        self._sync_mask_views()
        self._mask_path = None
        self._sync_path_labels()
        self._explorer_panel.select_path(path)
        if switch_tab:
            self._view_tabs.setCurrentIndex(0)

        self._overlay_canvas.set_highlight_boxes([])
        self._image_canvas.set_highlight_boxes([])
        self._active_highlight_boxes = []
        if os.path.normpath(str(prev_image_path)) != os.path.normpath(str(path)):
            if hasattr(self, "_roi_session_run_id"):
                self._roi_session_run_id = None
            if hasattr(self, "_roi_session_image_path"):
                self._roi_session_image_path = None
            if hasattr(self, "_roi_session_started_at"):
                self._roi_session_started_at = None
        self._populate_sidebar_history()
        self._populate_mask_list()
        self._refresh_ui_state()

    def _apply_loaded_mask(self, path: str, mask: QtGui.QImage) -> None:
        self._overlay_canvas.set_mask(mask)
        self._sync_mask_views()
        self._mask_path = Path(path)
        self._view_tabs.setCurrentIndex(0)
        self._sync_path_labels()
        self._refresh_ui_state()

    def _history_rows_from_csv(self, csv_path: Path) -> list[dict]:
        rows: list[dict] = []
        if not csv_path.is_file():
            return rows
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
        except Exception:
            return []
        return rows

    def _rewrite_csv_rows(self, path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fieldnames is None:
            fieldnames = list(rows[0].keys()) if rows else []
        with path.open("w", encoding="utf-8", newline="") as f:
            if not fieldnames:
                return
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _rewrite_jsonl_rows(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")

    def _delete_detection_persisted(self, det: dict) -> None:
        run_id = str(det.get("run_id") or "").strip()
        detection_id = str(det.get("detection_id") or "").strip()
        if not run_id:
            return

        mask_path = str(det.get("mask_path") or "").strip()
        if mask_path and os.path.isfile(mask_path):
            try:
                os.remove(mask_path)
            except OSError:
                pass

        results_root = getattr(self, "_results_dir", None)
        if not results_root:
            return
        run_dir = Path(results_root) / run_id
        csv_paths = [
            run_dir / "data" / "data.csv",
            run_dir / "data" / "detections.csv",
            run_dir / f"{run_id}_lan_quet_workspace.csv",
        ]

        def row_matches(row: dict) -> bool:
            row_run_id = str(row.get("run_id") or "").strip()
            row_det_id = str(row.get("detection_id") or "").strip()
            if row_run_id != run_id:
                return False
            if detection_id and row_det_id:
                return row_det_id == detection_id
            row_mask_path = self._resolve_results_asset_path(row.get("mask_rel") or "", row_run_id)
            return bool(mask_path) and os.path.normpath(row_mask_path) == os.path.normpath(mask_path)

        for csv_path in csv_paths:
            if not csv_path.is_file():
                continue
            rows = self._history_rows_from_csv(csv_path)
            if not rows:
                continue
            fieldnames = list(rows[0].keys())
            kept_rows = [row for row in rows if not row_matches(row)]
            self._rewrite_csv_rows(csv_path, kept_rows, fieldnames)

        jsonl_path = run_dir / "data" / "detections.jsonl"
        if jsonl_path.is_file():
            kept_jsonl: list[dict] = []
            try:
                with jsonl_path.open("r", encoding="utf-8", newline="") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if not row_matches(row):
                            kept_jsonl.append(row)
            except Exception:
                kept_jsonl = []
            self._rewrite_jsonl_rows(jsonl_path, kept_jsonl)

    def _resolve_results_asset_path(self, rel_path: str, run_id: str = "") -> str:
        rel_path = str(rel_path or "").strip()
        if not rel_path:
            return ""
        if os.path.isabs(rel_path):
            return rel_path
        results_root = Path(self._results_dir)
        if run_id and not rel_path.startswith(f"{run_id}/") and not rel_path.startswith(f"{run_id}\\"):
            return str(results_root / run_id / rel_path)
        return str(results_root / rel_path)

    def _row_image_path(self, row: dict) -> str:
        image_rel = (row.get("image_rel") or row.get("image_path") or "").strip()
        if not image_rel:
            return ""
        if os.path.isabs(image_rel):
            return image_rel
        return str(Path(self._workspace_root) / image_rel)

    def _iter_run_csv_files(self) -> list[Path]:
        results_root = Path(self._results_dir)
        if not results_root.exists():
            return []
        files = list(results_root.glob("*/*_lan_quet_workspace.csv"))
        files.extend(list(results_root.glob("*_lan_quet_workspace.csv")))
        files.sort(reverse=True)
        return files

    def _detections_from_history_rows(self, rows: list[dict]) -> dict[str, list[dict]]:
        by_image: dict[str, list[dict]] = {}
        for row in rows:
            image_path = self._row_image_path(row)
            if not image_path:
                continue

            run_id = (row.get("run_id") or "").strip()
            mask_path = self._resolve_results_asset_path(row.get("mask_rel") or "", run_id)
            overlay_path = self._resolve_results_asset_path(row.get("overlay_rel") or "", run_id)
            isolate_path = self._resolve_results_asset_path(row.get("isolate_rel") or "", run_id)
            image_asset_path = self._resolve_results_asset_path(row.get("image_asset_rel") or "", run_id)
            full_mask_path = self._resolve_results_asset_path(row.get("full_mask_rel") or "", run_id)

            score = 0.0
            try:
                score = float(row.get("score") or 0.0)
            except Exception:
                score = 0.0

            det = {
                "label": row.get("label") or "Mask",
                "score": score,
                "model_name": row.get("model") or row.get("model_name") or "Model",
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "isolate_path": isolate_path,
                "image_asset_path": image_asset_path,
                "full_mask_path": full_mask_path,
                "run_id": row.get("run_id") or "",
                "detection_id": row.get("detection_id") or "",
            }
            box = (row.get("box") or "").strip()
            if box:
                try:
                    parts = [float(x) for x in box.split(",")]
                    if len(parts) == 4:
                        det["box"] = parts
                except Exception:
                    pass

            by_image.setdefault(image_path, []).append(det)
        return by_image

    def _open_folder_history_dialog(self) -> None:
        from .dialogs import FolderHistoryDialog

        results_root = getattr(self, "_results_dir", None)
        if results_root is None:
            QtWidgets.QMessageBox.information(self, "Folder History", "No workspace results folder.")
            return
        results_root = Path(results_root)

        dlg = FolderHistoryDialog(self, results_root)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        res = dlg.get_result()
        if not res:
            return
        
        csv_path, image_rel = res

        def load_history_payload():
            rows = self._history_rows_from_csv(csv_path)
            by_image = self._detections_from_history_rows(rows)
            return rows, by_image

        rows, by_image = self._run_with_loading("History", "Loading history results...", load_history_payload)
        
        run_name = csv_path.name.replace("_lan_quet_workspace.csv", "")

        if image_rel is None:
            # Full Scan Mode: Load ALL results from this run into view
            if hasattr(self, "_history_view_detections"):
                self._history_view_detections = by_image
            
            # Refresh current image if it was affected
            if self._state and self._state.image_path in by_image:
                 self._populate_mask_list()
            
            if hasattr(self, "_history_current_label"):
                self._history_current_label.setText(f"Current Run: {run_name}")

            self.statusBar().showMessage(f"Loaded History Run: {run_name} ({len(by_image)} images)", 5000)
            return

        # Single Image Mode (legacy/specific)
        # Resolve image path from rel
        image_path = ""
        if os.path.isabs(image_rel):
            image_path = image_rel
        else:
            image_path = str(self._workspace_root / image_rel)

        self.load_image(image_path, switch_tab=True)
        
        # Show detections for this image from the selected run
        # Note: image_path from dialog might differ slightly from loaded if resolutions differ, 
        # but usually should match. We use the key from by_image that matches.
        
        # We need to find the key in by_image that corresponds to this image_path
        # The dialog gave us `image_rel`.
        
        # Let's try to find the exact match in by_image keys
        # The keys in by_image are absolute paths.
        
        target_key = None
        norm_target = os.path.normpath(image_path)
        
        for k in by_image.keys():
            if os.path.normpath(k) == norm_target:
                target_key = k
                break
        
        if not target_key and image_path in by_image:
             target_key = image_path

        dets = by_image.get(target_key, []) if target_key else []
        
        if hasattr(self, "_history_view_detections"):
            self._history_view_detections = {image_path: dets}
        if hasattr(self, "_object_list"):
            self._populate_mask_list()
            
        self.statusBar().showMessage(f"History: {run_name} -> {os.path.basename(image_path)}", 5000)

    def _populate_sidebar_history(self) -> None:
        if not hasattr(self, "_mask_history_tree") or not self._state:
            return
        
        self._mask_history_tree.clear()
        image_path = self._state.image_path
        rows = []
        norm_image = os.path.normpath(image_path)
        for csv_path in self._iter_run_csv_files():
            for row in self._history_rows_from_csv(csv_path):
                row_image = self._row_image_path(row)
                if row_image and os.path.normpath(row_image) == norm_image:
                    rows.append(row)

        if not rows:
            return

        # Group by Run
        by_run = {}
        for r in rows:
            rid = r.get("run_id") or "unknown"
            by_run.setdefault(rid, []).append(r)

        # Add "All" item
        all_item = QtWidgets.QTreeWidgetItem(self._mask_history_tree)
        all_item.setText(0, "All (No Filter)")
        all_item.setText(1, "-")
        all_item.setText(2, "-")
        all_item.setData(0, QtCore.Qt.UserRole, "ALL")

        # Sort runs new -> old
        for rid, r_rows in sorted(by_run.items(), reverse=True):
            model = "Unknown"
            max_score = 0.0
            for r in r_rows:
                if model == "Unknown":
                    model = r.get("model") or "Unknown"
                try:
                    s = float(r.get("score") or 0)
                    if s > max_score:
                        max_score = s
                except:
                    pass
            
            item = QtWidgets.QTreeWidgetItem(self._mask_history_tree)
            item.setText(0, rid)
            item.setText(1, model)
            item.setText(2, f"{max_score:.2f}")
            item.setData(0, QtCore.Qt.UserRole, rid)
            
        # Select "All" by default if no filter active
        # Or if we have a filter active?
        # self._mask_history_tree.setCurrentItem(all_item)

    def _on_sidebar_history_clicked(self, item: QtWidgets.QTreeWidgetItem, col: int) -> None:
        if not item:
            return
        run_id = item.data(0, QtCore.Qt.UserRole)
        if not self._state:
            return
            
        image_path = self._state.image_path
        rows = []
        norm_image = os.path.normpath(image_path)
        for csv_path in self._iter_run_csv_files():
            for row in self._history_rows_from_csv(csv_path):
                row_image = self._row_image_path(row)
                if row_image and os.path.normpath(row_image) == norm_image:
                    rows.append(row)
            
        target_rows = []
        if run_id == "ALL":
            target_rows = rows
            self.statusBar().showMessage("Showing all masks.", 3000)
        else:
            target_rows = [r for r in rows if (r.get("run_id") or "unknown") == run_id]
            self.statusBar().showMessage(f"Showing masks for run: {run_id}", 3000)
            
        # Convert to detections
        by_image = self._detections_from_history_rows(target_rows)
        
        # Extract for current image
        # Note: by_image keys are absolute paths.
        dets = []
        norm = os.path.normpath(image_path)
        for k, v in by_image.items():
            if os.path.normpath(k) == norm:
                dets = v
                break
        if not dets and image_path in by_image:
            dets = by_image[image_path]

        # Update view
        if hasattr(self, "_history_view_detections"):
            self._history_view_detections[image_path] = dets
            
        self._populate_mask_list()

    def _open_image_history_dialog(self) -> None:
        # Just focus the tab now
        if hasattr(self, "_focus_mask_tab"):
            self._focus_mask_tab()

    def _init_workspace(self) -> None:
        last = self._read_last_workspace_path()
        if last:
            try:
                self._setup_workspace(Path(last), create_root=False)
                self._save_workspace_path(last)
                return
            except Exception:
                pass
        self._prompt_create_workspace()

    def _setup_workspace(self, root_path: Path, *, create_root: bool) -> None:
        if create_root:
            root_path.mkdir(parents=True, exist_ok=True)
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"Workspace not found: {root_path}")
        self._workspace_root = root_path
        self._images_dir = self._workspace_root / "images"
        self._results_dir = self._workspace_root / "results"
        self._images_dir.mkdir(exist_ok=True)
        self._results_dir.mkdir(exist_ok=True)

        self.setWindowTitle(f"GroundTruth Editor - {self._workspace_root.name}")

        if hasattr(self, "_explorer_panel"):
            # Restore last session folder (VSCode style)
            self._load_initial_folder()
        self._load_workspace_results()
        self._populate_history_list()

    def _read_last_workspace_path(self) -> str:
        if hasattr(self, "_read_env_values"):
            data = self._read_env_values()
            return str(data.get("workspace_path") or "").strip()
        return ""

    def _save_workspace_path(self, path: str) -> None:
        if hasattr(self, "_update_env_values"):
            try:
                self._update_env_values({"workspace_path": str(path)})
            except Exception:
                pass

    def _read_last_folder_path(self) -> str:
        if hasattr(self, "_read_env_values"):
            data = self._read_env_values()
            return str(data.get("last_folder_path") or "").strip()
        return ""

    def _save_last_folder_path(self, path: str) -> None:
        if hasattr(self, "_update_env_values"):
            try:
                self._update_env_values({"last_folder_path": str(path)})
            except Exception:
                pass

    def _load_initial_folder(self) -> None:
        # Deprecated logic if we just want workspace images
        # But kept for reference if needed
        last = self._read_last_folder_path()
        if last and Path(last).is_dir():
            self.load_folder(last, append=False)
            return
        self.load_folder(str(self._images_dir), append=False)

    def _prompt_create_workspace(self) -> None:
        while True:
            start = str(Path.cwd())
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Create Workspace", start)
            if d:
                try:
                    self._setup_workspace(Path(d), create_root=True)
                    self._save_workspace_path(d)
                    return
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Create Workspace", str(e))
                    continue

            resp = QtWidgets.QMessageBox.question(
                self,
                "Workspace Required",
                "No workspace selected. The app needs a workspace to continue.\n\nQuit now?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if resp == QtWidgets.QMessageBox.StandardButton.Yes:
                QtCore.QTimer.singleShot(0, self.close)
                return

    def open_workspace_dialog(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Workspace", str(Path.cwd()))
        if d:
            try:
                self._setup_workspace(Path(d), create_root=False)
                self._save_workspace_path(d)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Open Workspace", str(e))

    def add_image_dialog(self) -> None:
        start = str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Add Image",
            start,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self.add_image(path)

    def add_folder_dialog(self) -> None:
        start = str(Path.cwd())
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Add Folder", start)
        if d:
            self.add_folder(d)

    def add_image(self, path: str) -> None:
        if not path:
            return
        dest = self._images_dir / os.path.basename(path)
        try:
            shutil.copy2(path, dest)
            self.load_image(str(dest), switch_tab=True)
            # Also refresh folder view to ensure it appears
            self.load_folder(str(self._images_dir), append=False)
            self._explorer_panel.select_path(str(dest))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Add Image", f"Failed to copy image: {e}")

    def add_folder(self, folder_path: str) -> None:
        folder = Path(folder_path)
        exts = {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"}
        files = set()
        for ext in exts:
            files.update(folder.glob(ext))
            files.update(folder.glob(ext.upper()))

        count = 0
        for f in files:
            try:
                dest = self._images_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    count += 1
            except Exception:
                pass

        if count > 0:
            self.statusBar().showMessage(f"Added {count} images to workspace.", 5000)
            self.load_folder(str(self._images_dir), append=False)
        else:
            self.statusBar().showMessage("No new images added.", 3000)

    def _load_workspace_results(self) -> None:
        if not hasattr(self, "_results_dir"):
            return
        if hasattr(self, "_image_detections_all"):
            self._image_detections_all = {}
        self._image_detections = {}
        if hasattr(self, "_history_view_detections"):
            self._history_view_detections = {}
        results_root = Path(self._results_dir)
        if not results_root.exists():
            return

        def load_all_results():
            rows: list[dict] = []
            for csv_path in self._iter_run_csv_files():
                rows.extend(self._history_rows_from_csv(csv_path))
            return self._detections_from_history_rows(rows)

        by_image = self._run_with_loading("Workspace", "Loading workspace results...", load_all_results)
        if hasattr(self, "_image_detections_all"):
            self._image_detections_all = {k: list(v) for k, v in by_image.items()}
        self._image_detections = {k: list(v) for k, v in by_image.items()}

        state = getattr(self, "_state", None)
        if state and hasattr(self, "_object_list"):
            if state.image_path in self._image_detections:
                self._populate_mask_list()

    def _populate_history_list(self) -> None:
        if not hasattr(self, "_history_list"):
            return
        self._history_list.clear()
        if not hasattr(self, "_results_dir"):
            return
        items = self._iter_run_csv_files()
        for p in items:
            item = QtWidgets.QListWidgetItem(p.name)
            item.setData(QtCore.Qt.UserRole, str(p))
            self._history_list.addItem(item)
            
        if hasattr(self, "_populate_isolate_list"):
            self._populate_isolate_list()

    def _on_history_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return
        path = item.data(QtCore.Qt.UserRole)
        if path:
            self.statusBar().showMessage(f"History: {path}", 5000)

    def load_folder(self, folder_path: str, append: bool = False) -> None:
        folder = Path(folder_path)
        exts = {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"}
        files = set()
        for ext in exts:
            files.update(folder.glob(ext))
            files.update(folder.glob(ext.upper()))

        new_files = sorted([str(f) for f in files])
        if append:
            pass  # We generally re-scan the workspace folder

        self._folder_images = new_files
        self._explorer_panel.set_images(self._folder_images)
        if self._folder_images:
            self.statusBar().showMessage(f"Workspace has {len(self._folder_images)} images.", 5000)
        self._explorer_panel.focus_filter()
        self._refresh_ui_state()
        if folder.exists():
            self._save_last_folder_path(str(folder))

    # Folder interactions are handled by ExplorerPanel signals.

    def load_image(self, path: str, switch_tab: bool = True) -> None:
        if not path:
            return
        self._start_async_io_job(kind="image", path=path, switch_tab=switch_tab)

    def open_mask_dialog(self) -> None:
        if self._state is None:
            QtWidgets.QMessageBox.critical(self, "Open Mask", "Load an image first.")
            return

        start = str(self._mask_path or Path(self._state.image_path).parent)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Mask",
            start,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        self._start_async_io_job(
            kind="mask",
            path=path,
            expected_size=(self._state.image_w, self._state.image_h),
        )

    def save_mask_dialog(self) -> None:
        if self._state is None or self._overlay_canvas.mask().isNull():
            QtWidgets.QMessageBox.critical(self, "Save Mask", "No mask to save.")
            return
        default_dir = self._mask_path or Path(self._state.image_path).parent

        suggested_name = ""
        current_item = self._object_list.currentItem()
        if current_item:
            text = current_item.text()
            # Clean up text for filename
            safe = "".join(c if c.isalnum() else "_" for c in text)
            while "__" in safe:
                safe = safe.replace("__", "_")
            suggested_name = f"_{safe.strip('_')}"

        base = Path(self._state.image_path).stem
        default_path = os.path.join(default_dir, f"{base}{suggested_name}.png")

        path, selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Mask As",
            str(default_path),
            "PNG (0/255) (*.png);;PNG (0/1, indexed) (*.png)",
        )
        if not path:
            return
        try:
            def do_save():
                if selected.startswith("PNG (0/1"):
                    save_mask_png_01_indexed(path, self._overlay_canvas.mask())
                else:
                    save_mask_png_0255(path, self._overlay_canvas.mask())

            self._run_with_loading("Export Mask", "Saving mask...", do_save)
        except ImageIoError as e:
            QtWidgets.QMessageBox.critical(self, "Save Mask", str(e))
            return
        self._mask_path = Path(path)
        self._sync_path_labels()
        self._refresh_ui_state()

    def save_image_dialog(self) -> None:
        if self._state is None:
            QtWidgets.QMessageBox.critical(self, "Export Image", "No image to export.")
            return

        image = self._image_canvas.image()
        if image.isNull():
            image = self._overlay_canvas.image()
        if image.isNull():
            QtWidgets.QMessageBox.critical(self, "Export Image", "No image to export.")
            return

        base = Path(self._state.image_path).stem
        default_dir = Path(self._state.image_path).parent
        default_path = os.path.join(default_dir, f"{base}_image.png")
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Image As",
            str(default_path),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return
        ok = self._run_with_loading("Export Image", "Saving image...", lambda: image.save(path))
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Export Image", f"Failed to save image: {path}")

    def _build_overlay_export_image(self) -> QtGui.QImage:
        base = self._overlay_canvas.image()
        if base.isNull():
            return QtGui.QImage()

        result = base.copy()
        overlay = self._overlay_canvas.overlay_visual()
        if not overlay.isNull():
            painter = QtGui.QPainter(result)
            painter.setOpacity(self._overlay_canvas.canvas_state().overlay_opacity / 255.0)
            painter.drawImage(0, 0, overlay)
            painter.end()
            return result

        mask = self._overlay_canvas.mask()
        if mask.isNull():
            return result

        overlay = mask.convertToFormat(QtGui.QImage.Format_Indexed8)
        color_table = []
        opacity = self._overlay_canvas.canvas_state().overlay_opacity
        for i in range(256):
            alpha = (i * opacity) // 255
            color_table.append(QtGui.QColor(255, 0, 0, alpha).rgba())
        overlay.setColorTable(color_table)

        painter = QtGui.QPainter(result)
        painter.drawImage(0, 0, overlay)
        painter.end()
        return result

    def _build_overlay_boxes_export_image(self) -> QtGui.QImage:
        result = self._build_overlay_export_image()
        if result.isNull():
            return result

        boxes = list(getattr(self, "_active_highlight_boxes", []) or [])
        if not boxes:
            return result

        painter = QtGui.QPainter(result)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        for rect, label in boxes:
            color = self._label_color(label)
            painter.setPen(QtGui.QPen(color, 2))
            painter.drawRect(rect)
            if label:
                font = painter.font()
                font.setPointSize(10)
                painter.setFont(font)
                fm = painter.fontMetrics()
                text_rect = fm.boundingRect(label)
                bg_rect = QtCore.QRectF(
                    rect.left(),
                    max(0.0, rect.top() - text_rect.height() - 4),
                    text_rect.width() + 6,
                    text_rect.height() + 4,
                )
                if bg_rect.top() < 0:
                    bg_rect.moveTop(rect.top())
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                bg_color = QtGui.QColor(color)
                bg_color.setAlpha(180)
                painter.setBrush(bg_color)
                painter.drawRect(bg_rect)
                painter.setPen(QtGui.QColor(0, 0, 0))
                painter.drawText(bg_rect, QtCore.Qt.AlignmentFlag.AlignCenter, label)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        painter.end()
        return result

    def save_overlay_dialog(self) -> None:
        if self._state is None:
            QtWidgets.QMessageBox.critical(self, "Export Overlay", "No overlay to export.")
            return

        overlay = self._run_with_loading("Export Overlay", "Preparing overlay...", self._build_overlay_export_image)
        if overlay.isNull():
            QtWidgets.QMessageBox.critical(self, "Export Overlay", "No overlay to export.")
            return

        base = Path(self._state.image_path).stem
        default_dir = Path(self._state.image_path).parent
        default_path = os.path.join(default_dir, f"{base}_overlay.png")
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Overlay As",
            str(default_path),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return
        ok = self._run_with_loading("Export Overlay", "Saving overlay...", lambda: overlay.save(path))
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Export Overlay", f"Failed to save overlay: {path}")

    def save_overlay_boxes_dialog(self) -> None:
        if self._state is None:
            QtWidgets.QMessageBox.critical(self, "Export Overlay + Boxes", "No image to export.")
            return

        image = self._run_with_loading(
            "Export Overlay + Boxes",
            "Preparing overlay and boxes...",
            self._build_overlay_boxes_export_image,
        )
        if image.isNull():
            QtWidgets.QMessageBox.critical(self, "Export Overlay + Boxes", "No overlay/boxes to export.")
            return

        base = Path(self._state.image_path).stem
        default_dir = Path(self._state.image_path).parent
        default_path = os.path.join(default_dir, f"{base}_overlay_boxes.png")
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Overlay + Boxes As",
            str(default_path),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return
        ok = self._run_with_loading(
            "Export Overlay + Boxes",
            "Saving overlay and boxes...",
            lambda: image.save(path),
        )
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Export Overlay + Boxes", f"Failed to save image: {path}")

    def _sync_brush_slider(self, radius: int) -> None:
        if self._brush_slider.value() == radius:
            return
        self._brush_slider.blockSignals(True)
        self._brush_slider.setValue(radius)
        self._brush_slider.blockSignals(False)
        if hasattr(self, "_brush_spin"):
            self._brush_spin.blockSignals(True)
            self._brush_spin.setValue(radius)
            self._brush_spin.blockSignals(False)
        self._brush_value.setText(f"{radius} px")

    def _sync_mask_views(self) -> None:
        if getattr(self, "_mask_canvas", None) is not None and getattr(self, "_overlay_canvas", None) is not None:
            self._mask_canvas.set_mask(self._overlay_canvas.mask())
            self._mask_canvas.set_overlay_visual(self._overlay_canvas.overlay_visual())
            # Sync highlight boxes to mask canvas if available
            if hasattr(self, "_active_highlight_boxes"):
                self._mask_canvas.set_highlight_boxes(self._active_highlight_boxes)
            self._mask_canvas.update()

    def _on_cursor_info(self, x: int, y: int, v: int) -> None:
        self._status_label.setText(f"x={x}  y={y}  mask={v}")

    def _sync_path_labels(self) -> None:
        if self._state is None:
            self._image_label.setText("")
            self._mask_label.setText("")
            return
        img_name = os.path.basename(self._state.image_path)
        self._image_label.setText(f"Image: {img_name}")
        if self._mask_path is None:
            self._mask_label.setText("Mask: (new)")
        else:
            self._mask_label.setText(f"Mask: {os.path.basename(str(self._mask_path))}")

    def _focus_folder_filter(self) -> None:
        if hasattr(self, "_left_tabs") and hasattr(self, "_explorer_tab"):
            self._left_tabs.setCurrentWidget(self._explorer_tab)
        self._explorer_panel.focus_filter()

    def _navigate_folder(self, delta: int) -> None:
        next_path = self._explorer_panel.navigate(delta, self._state.image_path if self._state is not None else None)
        if next_path:
            self.load_image(next_path, switch_tab=True)

    def _open_compare_tool_dialog(self) -> None:
        from .dialogs.compare_dialog import CompareDialog
        from .features.compare_utils import compute_dice_iou
        import cv2
        import numpy as np

        if not hasattr(self, "_results_dir") or not self._results_dir:
            QtWidgets.QMessageBox.warning(self, "Compare", "No workspace results folder found. Run a prediction first.")
            return

        dlg = CompareDialog(self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        gt_dir, prefix = dlg.get_result()
        if not gt_dir or not os.path.exists(gt_dir):
            QtWidgets.QMessageBox.warning(self, "Compare", "Invalid GT folder.")
            return

        def do_compare():
            def _match_gt_file(image_name: str, gt_files: list[Path], affix: str) -> Path | None:
                image_path = Path(str(image_name or ""))
                image_name_full = image_path.name
                image_stem = image_path.stem
                affix = str(affix or "").strip()
                for gt_file in gt_files:
                    if gt_file.name == image_name_full or gt_file.stem == image_stem:
                        return gt_file
                    if affix and (
                        gt_file.stem == f"{affix}{image_stem}" or gt_file.stem == f"{image_stem}{affix}"
                    ):
                        return gt_file
                return None

            def _load_pred_mask(rows: list[dict], run_id: str) -> np.ndarray | None:
                full_mask_rel = str(rows[0].get("full_mask_rel") or "").strip()
                if full_mask_rel:
                    full_mask_path = self._resolve_results_asset_path(full_mask_rel, run_id)
                    if os.path.isfile(full_mask_path):
                        full_mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
                        if full_mask is not None:
                            return full_mask

                merged_mask = None
                for row in rows:
                    mask_rel = str(row.get("mask_rel") or "").strip()
                    if not mask_rel:
                        continue
                    mask_path = self._resolve_results_asset_path(mask_rel, run_id)
                    if not os.path.isfile(mask_path):
                        continue
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    if merged_mask is None:
                        merged_mask = mask
                    else:
                        if merged_mask.shape != mask.shape:
                            mask = cv2.resize(
                                mask,
                                (merged_mask.shape[1], merged_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        merged_mask = np.maximum(merged_mask, mask)
                return merged_mask

            results_root = Path(self._results_dir)
            run_dirs = []
            for d in results_root.iterdir():
                if d.is_dir() and d.name != "models":
                    if (d / "data" / "data.csv").exists() or list(d.glob("*_lan_quet_workspace.csv")):
                        run_dirs.append(d)

            run_dirs = sorted(run_dirs, reverse=True)
            if not run_dirs:
                return None, "No runs found."

            latest_run = run_dirs[0]
            data_csv = latest_run / "data" / "data.csv"
            if not data_csv.is_file():
                return None, f"No compare data found in latest run ({latest_run.name})."

            rows = self._history_rows_from_csv(data_csv)
            if not rows:
                return None, f"No detection rows found in latest run ({latest_run.name})."

            rows_by_image: dict[str, list[dict]] = {}
            run_id = str(latest_run.name)
            for row in rows:
                image_rel = str(row.get("image_rel") or row.get("image_path") or "").strip()
                if not image_rel:
                    continue
                rows_by_image.setdefault(image_rel, []).append(row)

            results = []
            gt_root = Path(gt_dir)
            gt_files = list(gt_root.glob("*.png")) + list(gt_root.glob("*.jpg")) + list(gt_root.glob("*.jpeg"))
            for image_rel, image_rows in rows_by_image.items():
                best_match = _match_gt_file(image_rel, gt_files, prefix)
                if not best_match:
                    continue

                pm_img = _load_pred_mask(image_rows, run_id)
                gt_img = cv2.imread(str(best_match), cv2.IMREAD_GRAYSCALE)
                if pm_img is None or gt_img is None:
                    continue

                if pm_img.shape != gt_img.shape:
                    gt_img = cv2.resize(gt_img, (pm_img.shape[1], pm_img.shape[0]), interpolation=cv2.INTER_NEAREST)

                dice, iou = compute_dice_iou(pm_img, gt_img)
                results.append({
                    "image": Path(image_rel).name,
                    "gt_mask": best_match.name,
                    "dice": dice,
                    "iou": iou,
                })

            return results, None

        results, error = self._run_with_loading("Compare", "Comparing masks...", do_compare)
        if error:
            QtWidgets.QMessageBox.warning(self, "Compare", error)
            return
             
        if not results:
             QtWidgets.QMessageBox.information(self, "Compare", "No matching masks found between GT folder and predictions.")
             return
             
        if hasattr(self, "_compare_panel"):
             self._compare_panel.set_results(results)
             if hasattr(self, "_left_tabs") and hasattr(self, "_compare_tab"):
                 self._left_tabs.setCurrentWidget(self._compare_tab)
        
        QtWidgets.QMessageBox.information(self, "Compare", f"Compared {len(results)} masks successfully.")
