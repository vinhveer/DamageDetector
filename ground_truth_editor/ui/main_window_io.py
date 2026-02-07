from __future__ import annotations

import csv
import datetime
import os
import shutil
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from image_io import (
    ImageIoError,
    load_image,
    load_mask,
    new_blank_mask,
    save_mask_png_01_indexed,
    save_mask_png_0255,
)

from .state import LoadedState


class MainWindowIOMixin:
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

    def _detections_from_history_rows(self, rows: list[dict]) -> dict[str, list[dict]]:
        by_image: dict[str, list[dict]] = {}
        results_root = Path(self._results_dir)
        for row in rows:
            image_rel = (row.get("image_rel") or row.get("image_path") or "").strip()
            if not image_rel:
                continue
            if os.path.isabs(image_rel):
                image_path = image_rel
            else:
                image_path = str(Path(self._workspace_root) / image_rel)

            mask_rel = (row.get("mask_rel") or "").strip()
            if mask_rel:
                if os.path.isabs(mask_rel):
                    mask_path = mask_rel
                else:
                    mask_path = str(results_root / mask_rel)
            else:
                mask_path = ""

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
                "run_id": row.get("run_id") or "",
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
        
        # Load the run data so we can display it
        rows = self._history_rows_from_csv(csv_path)
        by_image = self._detections_from_history_rows(rows)
        
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
        
        # We need the data.csv for this image
        image_path = self._state.image_path
        img_name = getattr(self, "_sanitize_name", None)
        if callable(img_name):
            folder_name = self._sanitize_name(Path(image_path).stem or "image")
        else:
            folder_name = Path(image_path).stem or "image"

        results_root = getattr(self, "_results_dir", None)
        if not results_root:
            return

        data_csv = Path(results_root) / folder_name / "data.csv"
        rows = []
        if data_csv.is_file():
            rows = self._history_rows_from_csv(data_csv)

        # 1. Add "Current (Unsaved/Session)" if there are detections in memory?
        # Actually, self._image_detections might contain unsaved runs.
        # But our spec says "All masks are saved to disk on each detection".
        # So reading CSV is the source of truth.
        
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
        
        # We need to reload detections from CSV for this run (or all)
        # Because we might have filtered them out of memory.
        # Although _load_workspace_results loads everything into _image_detections_all.
        
        if not self._state:
            return
            
        image_path = self._state.image_path
        
        # Re-read CSV to be safe/consistent
        img_name = getattr(self, "_sanitize_name", None)
        if callable(img_name):
            folder_name = self._sanitize_name(Path(image_path).stem or "image")
        else:
            folder_name = Path(image_path).stem or "image"
        
        results_root = getattr(self, "_results_dir", None)
        if results_root:
            data_csv = Path(results_root) / folder_name / "data.csv"
            rows = self._history_rows_from_csv(data_csv)
        else:
            rows = []
            
        # Filter
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

        for sub in results_root.iterdir():
            if not sub.is_dir():
                continue
            data_csv = sub / "data.csv"
            if not data_csv.is_file():
                continue
            try:
                with data_csv.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        image_rel = (row.get("image_rel") or row.get("image_path") or "").strip()
                        if not image_rel:
                            continue
                        if os.path.isabs(image_rel):
                            image_path = image_rel
                        else:
                            image_path = str(Path(self._workspace_root) / image_rel)

                        mask_rel = (row.get("mask_rel") or "").strip()
                        if mask_rel:
                            if os.path.isabs(mask_rel):
                                mask_path = mask_rel
                            else:
                                mask_path = str(results_root / mask_rel)
                        else:
                            mask_path = ""

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
                            "run_id": row.get("run_id") or "",
                        }

                        box = (row.get("box") or "").strip()
                        if box:
                            try:
                                parts = [float(x) for x in box.split(",")]
                                if len(parts) == 4:
                                    det["box"] = parts
                            except Exception:
                                pass

                        if hasattr(self, "_image_detections_all"):
                            self._image_detections_all.setdefault(image_path, []).append(det)
                        self._image_detections.setdefault(image_path, []).append(det)
            except Exception:
                continue

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
        results_root = Path(self._results_dir)
        if not results_root.exists():
            return
        items = sorted(results_root.glob("*_lan_quet_workspace.csv"))
        for p in items:
            item = QtWidgets.QListWidgetItem(p.name)
            item.setData(QtCore.Qt.UserRole, str(p))
            self._history_list.addItem(item)

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
        try:
            img = load_image(path)
        except ImageIoError as e:
            QtWidgets.QMessageBox.critical(self, "Open Image", str(e))
            return

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
            self._view_tabs.setCurrentIndex(0)  # Overlay

        self._overlay_canvas.set_highlight_box(None)
        self._populate_sidebar_history()
        self._populate_mask_list()
        self._refresh_ui_state()

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
        try:
            loaded = load_mask(path, (self._state.image_w, self._state.image_h))
        except ImageIoError as e:
            QtWidgets.QMessageBox.critical(self, "Open Mask", str(e))
            return
        self._overlay_canvas.set_mask(loaded.mask)
        self._sync_mask_views()
        self._mask_path = Path(path)
        self._view_tabs.setCurrentIndex(0)
        self._sync_path_labels()
        self._refresh_ui_state()

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
            if selected.startswith("PNG (0/1"):
                save_mask_png_01_indexed(path, self._overlay_canvas.mask())
            else:
                save_mask_png_0255(path, self._overlay_canvas.mask())
        except ImageIoError as e:
            QtWidgets.QMessageBox.critical(self, "Save Mask", str(e))
            return
        self._mask_path = Path(path)
        self._sync_path_labels()
        self._refresh_ui_state()

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
        self._mask_canvas.set_mask(self._overlay_canvas.mask())
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
