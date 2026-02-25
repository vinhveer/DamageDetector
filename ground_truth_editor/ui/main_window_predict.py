from __future__ import annotations

import csv
import datetime
import os
from dataclasses import replace
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from image_io import ImageIoError, load_image, load_mask
from predict_sam_dino import SamDinoParams
from predict_unet import UnetParams

from .dialogs import PredictRunDialog, ProcessingDialog, IsolateDialog
from .features.predict.workers import (
    BatchSamDinoWorker,
    BatchUnetWorker,
    SamDinoIsolateWorker,
    SamDinoWorker,
    UnetDinoWorker,
    UnetWorker,
    WorkerBase,
)


class MainWindowPredictMixin:
    def _resolve_service_path(self, path: str | None) -> str:
        if not path:
            return ""
        p = Path(str(path))
        if p.is_absolute():
            return str(p)
        # Service processes run with repo root as CWD.
        repo_root = Path(__file__).resolve().parents[2]
        return str(repo_root / p)

    def _build_sam_dino_params(
        self,
        *,
        output_dir: str | None = None,
        invert_override: bool | None = None,
        force_use_delta: bool | None = None,
    ) -> SamDinoParams:
        ckpt = self._sd_sam_ckpt.text().strip()
        if not ckpt:
            raise ValueError("SAM checkpoint is required.")

        gdino_ckpt = self._sd_gdino_ckpt.text().strip()
        if not gdino_ckpt:
            raise ValueError("GroundingDINO checkpoint is required.")

        gdino_lower = gdino_ckpt.lower()
        if gdino_lower.endswith((".pth", ".pt", ".safetensors", ".bin")) and not os.path.exists(gdino_ckpt):
            raise ValueError(f"GroundingDINO checkpoint not found: {gdino_ckpt}")

        out_dir = output_dir 
        if not out_dir:
            # Use workspace results dir if available
            res_dir = getattr(self, "_results_dir", None)
            if res_dir:
                out_dir = str(res_dir)
            else:
                out_dir = "results_sam_dino" # Fallback

        queries = [q.strip() for q in self._sd_queries.text().split(",") if q.strip()]
        if not queries:
            raise ValueError("Text queries required (comma-separated).")

        delta_type = "none"
        delta_ckpt = self._sd_delta_ckpt.text().strip()
        middle_dim = int(self._sd_middle_dim.value())
        scaling_factor = float(self._sd_scaling_factor.value())
        rank = int(self._sd_rank.value())
        
        should_use_delta = False
        if force_use_delta is not None:
             should_use_delta = force_use_delta
        else:
             # Inferred logic: uses delta if path is set (not auto/empty)
             if delta_ckpt and delta_ckpt.lower() != "auto":
                 should_use_delta = True
        
        if should_use_delta:
            delta_type = str(self._sd_delta_type.currentText())
            if not delta_ckpt or delta_ckpt.lower() == "auto":
                 raise ValueError("Deta checkpoint path is required to use fine-tuned model.")
            
            if delta_ckpt.lower().endswith((".pth", ".pt", ".safetensors", ".bin")) and not os.path.exists(delta_ckpt):
                 raise ValueError(f"Delta checkpoint not found: {delta_ckpt}")

        invert_mask = bool(self._sd_invert.isChecked())
        if invert_override is not None:
            invert_mask = bool(invert_override)

        cfg_id = self._sd_gdino_cfg.currentData()
        if cfg_id is None or not str(cfg_id).strip():
            raise ValueError("GroundingDINO config is required.")

        return SamDinoParams(
            sam_checkpoint=ckpt,
            sam_model_type=str(self._sd_sam_type.currentText()),
            delta_type=delta_type,
            delta_checkpoint=delta_ckpt,
            middle_dim=middle_dim,
            scaling_factor=scaling_factor,
            rank=rank,
            gdino_checkpoint=gdino_ckpt,
            gdino_config_id=str(cfg_id),
            text_queries=queries,
            box_threshold=float(self._sd_box_thr.value()),
            text_threshold=float(self._sd_text_thr.value()),
            max_dets=int(self._sd_max_dets.value()),
            invert_mask=invert_mask,
            sam_min_component_area=int(self._sd_min_area.value()),
            sam_dilate_iters=int(self._sd_dilate.value()),
            device=str(self._sd_device.currentText()),
            output_dir=out_dir,
        )

    def _append_log(self, widget: QtWidgets.QPlainTextEdit, text: str) -> None:
        widget.appendPlainText(text)

    def _new_run_id(self) -> str:
        base = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = base
        results_root = getattr(self, "_results_dir", None)
        if results_root is None:
            return run_id
        try:
            root = Path(results_root)
        except Exception:
            return run_id
        suffix = 1
        while (root / run_id / f"{run_id}_lan_quet_workspace.csv").exists():
            suffix += 1
            run_id = f"{base}_{suffix:02d}"
        return run_id

    def _save_run_metadata(self, run_id: str, scope: str, params: object) -> None:
        results_root = getattr(self, "_results_dir", None)
        if not results_root:
            return

        out_dir = Path(results_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"{run_id}_info.txt"
        
        lines = [
            f"Run ID: {run_id}",
            f"Date: {datetime.datetime.now().isoformat()}",
            f"Scope: {scope}",
            "-" * 20,
            "Parameters:",
        ]

        if hasattr(params, "__dict__"):
            p_dict = params.__dict__
        elif isinstance(params, dict):
            p_dict = params
        else:
            p_dict = {}

        for k, v in p_dict.items():
            lines.append(f"{k}: {v}")

        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            print(f"Failed to save run info: {e}")

    def _begin_run(self, scope: str, params: object = None) -> None:
        if self._current_run_id is None:
            self._current_run_id = self._new_run_id()
            self._current_run_scope = str(scope or "current")
            self._current_run_started_at = datetime.datetime.now().isoformat(timespec="seconds")
            
            if params:
                self._save_run_metadata(self._current_run_id, scope, params)

    def _reset_run_context(self) -> None:
        self._current_run_id = None
        self._current_run_scope = None
        self._current_run_started_at = None

    def _sanitize_name(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_") or "image"

    def _mask_from_detection(self, det: dict) -> "tuple[object | None, str | None]":
        import base64
        import cv2
        import numpy as np

        mask_b64 = det.get("mask_b64")
        if mask_b64:
            try:
                mask_bytes = base64.b64decode(mask_b64)
                arr = np.frombuffer(mask_bytes, np.uint8)
                mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask, None
            except Exception:
                pass

        mask_path = det.get("mask_path") or det.get("mask") or ""
        if mask_path:
            path = self._resolve_service_path(mask_path)
            if os.path.isfile(path):
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    return mask, path
        return None, None

    def _append_csv_rows(self, path: Path, rows: list[dict], fieldnames: list[str]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _persist_detection_results(self, details: dict) -> None:
        results_root = getattr(self, "_results_dir", None)
        if results_root is None:
            return
        image_path = details.get("image_path") or (self._state.image_path if self._state else "")
        if not image_path:
            return

        run_id = self._current_run_id or self._new_run_id()
        started_at = self._current_run_started_at or datetime.datetime.now().isoformat(timespec="seconds")

        dets = details.get("detections") or []
        if not dets:
            return

        img_path = Path(image_path)
        out_dir = Path(results_root) / run_id
        mask_dir = out_dir / "mask"
        data_dir = out_dir / "data"
        mask_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        rel_image = ""
        try:
            rel_image = os.path.relpath(str(img_path), str(self._workspace_root))
            if rel_image.startswith(".."):
                rel_image = str(img_path)
        except Exception:
            rel_image = str(img_path)

        rows: list[dict] = []
        for idx, det in enumerate(dets):
            mask_arr, _src_path = self._mask_from_detection(det)
            if mask_arr is None:
                continue

            if int(mask_arr.max()) <= 1:
                import numpy as np

                mask_arr = (mask_arr.astype(np.uint8) * 255)

            # Mask trùng tên với ảnh
            mask_name = img_path.name
            if len(dets) > 1:
                 # If there are multiple detections, we need to save them separately so they don't overwrite each other,
                 # or maybe the user wants to combine them? "Mask trùng tên với ảnh nhé"
                 # I'll append the index just to be safe if there are multiple.
                 if idx > 0:
                     mask_name = f"{img_path.stem}_{idx}{img_path.suffix}"
                
            mask_path = mask_dir / mask_name
            try:
                import cv2

                cv2.imwrite(str(mask_path), mask_arr)
            except Exception:
                continue

            det["mask_path"] = str(mask_path)
            det["run_id"] = run_id
            model_name = det.get("model_name") or "Model"
            label = det.get("label") or "Mask"
            score = det.get("score")
            try:
                score = float(score)
            except Exception:
                score = 0.0

            box = det.get("box") or det.get("bbox")
            box_text = ""
            if box and isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    box_text = ",".join(str(float(x)) for x in box)
                except Exception:
                    box_text = ""

            rows.append(
                {
                    "run_id": run_id,
                    "created_at": started_at,
                    "model": str(model_name),
                    "label": str(label),
                    "score": score,
                    "mask_rel": f"mask/{mask_name}",
                    "image_rel": rel_image,
                    "box": box_text,
                }
            )

        if not rows:
            return

        data_csv = data_dir / "data.csv"
        fields = ["run_id", "created_at", "model", "label", "score", "mask_rel", "image_rel", "box"]
        self._append_csv_rows(data_csv, rows, fields)

        run_csv = out_dir / f"{run_id}_lan_quet_workspace.csv"
        self._append_csv_rows(run_csv, rows, fields)
        self._populate_history_list()

    def _store_detections_from_details(self, details: dict, *, update_view: bool) -> None:
        image_path = details.get("image_path") or (self._state.image_path if self._state else "")
        if not image_path:
            return

        detections = details.get("detections")
        if not detections:
            mask_path = details.get("mask_path") or details.get("mask") or ""
            if mask_path:
                detections = [
                    {
                        "label": "Mask",
                        "score": 1.0,
                        "model_name": details.get("model_name") or "Mask",
                        "mask_path": self._resolve_service_path(mask_path),
                    }
                ]
                details["detections"] = detections

        if detections:
            for det in detections:
                mp = det.get("mask_path")
                if mp:
                    det["mask_path"] = self._resolve_service_path(mp)
            self._image_detections[image_path] = detections
            if hasattr(self, "_image_detections_all"):
                self._image_detections_all[image_path] = detections
            if hasattr(self, "_history_view_detections"):
                self._history_view_detections.pop(image_path, None)
            if update_view and self._state:
                if os.path.normpath(str(image_path)) == os.path.normpath(str(self._state.image_path)):
                    self._populate_mask_list()

    def _apply_visual_result(self, details: dict) -> None:
        overlay_path = self._resolve_service_path(details.get("overlay_path") or "")
        if overlay_path and os.path.isfile(overlay_path):
            try:
                overlay_img = load_image(overlay_path)
                self._image_canvas.set_image(overlay_img)
                if self._active_log_widget is not None:
                    self._append_log(self._active_log_widget, f"Preview overlay: {overlay_path}")
            except Exception as e:
                if self._active_log_widget is not None:
                    self._append_log(self._active_log_widget, f"Could not load overlay preview: {e}")

        mask_path = self._resolve_service_path(details.get("mask_path") or details.get("mask") or "")
        if mask_path and self._state is not None and os.path.isfile(mask_path):
            try:
                loaded = load_mask(mask_path, (self._state.image_w, self._state.image_h))
                self._overlay_canvas.set_mask(loaded.mask)
                self._sync_mask_views()
                self._mask_path = Path(mask_path)
                self._sync_path_labels()
                self._view_tabs.setCurrentIndex(0)
                if self._active_log_widget is not None:
                    self._append_log(self._active_log_widget, f"Loaded mask: {mask_path}")
            except ImageIoError as e:
                if self._active_log_widget is not None:
                    self._append_log(self._active_log_widget, f"Could not load output mask: {e}")

    @QtCore.Slot(str)
    def _on_worker_log(self, text: str) -> None:
        if self._active_log_widget is not None:
            self._append_log(self._active_log_widget, text)

    @QtCore.Slot(str)
    def _on_worker_failed_slot(self, msg: str) -> None:
        self._post_run_action = None
        if self._active_log_widget is not None:
            self._append_log(self._active_log_widget, f"FAILED: {msg}")
        QtWidgets.QMessageBox.critical(self, "Run", msg)

    @QtCore.Slot(object)
    def _on_worker_finished_slot(self, details_obj) -> None:
        details = dict(details_obj or {})
        if self._active_log_widget is not None and details.get("stopped"):
            self._append_log(self._active_log_widget, "STOPPED")
            self._post_run_action = None
            return

        if details.get("batch_done"):
            results = details.get("results") or []
            current = self._state.image_path if self._state else ""
            current_norm = os.path.normpath(str(current)) if current else ""
            current_result = None
            if results:
                for res in results:
                    img_path = str(res.get("image_path") or "")
                    is_current = bool(current_norm) and os.path.normpath(img_path) == current_norm
                    self._store_detections_from_details(res, update_view=is_current)
                    self._persist_detection_results(res)
                    if is_current:
                        current_result = res
                if current_result is not None:
                    self._apply_visual_result(current_result)
            if self._active_log_widget is not None:
                self._append_log(self._active_log_widget, "Batch Processing COMPLETE.")
            return

        self._store_detections_from_details(details, update_view=True)
        self._persist_detection_results(details)
        self._apply_visual_result(details)

        if self._active_log_widget is not None:
            self._append_log(self._active_log_widget, "DONE")

        post_action = self._post_run_action
        self._post_run_action = None
        if post_action and post_action.get("type") in {"extract", "isolate"}:
            self._handle_extract_result(details, post_action.get("source_image"))

        self._refresh_ui_state()

    def _open_predict_mode_dialog(self) -> None:
        if self._thread is not None:
            QtWidgets.QMessageBox.information(self, "Predict", "Already running. Please wait.")
            return

        dlg = PredictRunDialog(
            self,
            has_image=self._state is not None,
            has_folder=bool(self._folder_images),
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        mode, scope = dlg.get_result()
        # Delay slightly to allow dialog to close
        QtCore.QTimer.singleShot(50, lambda: self._execute_prediction(mode, scope))

    def _predict_folder_dialog(self) -> None:
        if self._thread is not None:
            QtWidgets.QMessageBox.information(self, "Predict", "Already running. Please wait.")
            return

        start = str(Path.cwd())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Predict Folder", start)
        if not folder:
            return

        # Load folder into explorer so results can be viewed later.
        self.load_folder(folder, append=False)
        images = list(self._folder_images)
        if not images:
            QtWidgets.QMessageBox.information(self, "Predict Folder", "No images found in selected folder.")
            return

        dlg = PredictRunDialog(self, has_image=False, has_folder=True)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        mode, _scope = dlg.get_result()
        QtCore.QTimer.singleShot(50, lambda: self._execute_prediction(mode, "folder", folder_images=images))

    def _predict_with_scope(self, mode: str) -> None:
        if self._thread is not None:
            QtWidgets.QMessageBox.information(self, "Predict", "Already running. Please wait.")
            return
        has_image = self._state is not None
        has_folder = bool(self._folder_images)
        if not has_image and not has_folder:
            QtWidgets.QMessageBox.information(self, "Predict", "No image or folder loaded.")
            return

        scope = "current"
        if has_image and has_folder:
            resp = QtWidgets.QMessageBox.question(
                self,
                "Predict Scope",
                "Run on current image?\n\nChoose No to run on whole folder.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if resp == QtWidgets.QMessageBox.StandardButton.Cancel:
                return
            scope = "current" if resp == QtWidgets.QMessageBox.StandardButton.Yes else "folder"
        elif has_folder and not has_image:
            scope = "folder"
        else:
            scope = "current"

        self._execute_prediction(mode, scope)

    def _execute_prediction(self, mode: str, scope: str, *, folder_images: list[str] | None = None) -> None:
        mode = str(mode or "").strip().lower()
        if mode == "sam_dino_ft":
            title = "Predict SAM + DINO + Finetune"
        elif mode == "sam_dino":
            title = "Predict SAM + DINO"
        elif mode == "unet":
            title = "Predict UNet + DINO"
        else:
            QtWidgets.QMessageBox.critical(self, "Predict", f"Unknown mode: {mode}")
            return

        if not self._ensure_settings_ready(mode):
            return

        if scope == "folder":
            images = folder_images if folder_images is not None else self._folder_images
            if not images:
                QtWidgets.QMessageBox.information(self, "Predict", "No images in folder list. Open a folder first.")
                self._focus_folder_filter()
                return

        if mode in {"sam_dino", "sam_dino_ft"}:
            # Explicitly set use_delta. 
            # If mode is sam_dino, we force False to ignore any lingering path in settings.
            use_delta = (mode == "sam_dino_ft")
            if scope == "current":
                self._run_sam_dino(force_use_delta=use_delta)
            else:
                self._run_batch_sam_dino(images, force_use_delta=use_delta)
            return

        if mode == "unet":
            if scope == "current":
                self._run_unet()
            else:
                self._run_batch_unet(images)
            return

    def _ensure_image_loaded(self) -> bool:
        return self._state is not None

    def _create_cutout_from_mask(self, image_path: str, mask_path: str, output_dir: str | None) -> str:
        import cv2

        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        if mask.shape[:2] != bgr.shape[:2]:
            mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask

        out_dir = output_dir or os.path.dirname(mask_path) or os.getcwd()
        base = Path(image_path).stem
        out_path = os.path.join(out_dir, f"{base}_cutout.png")
        if not cv2.imwrite(out_path, bgra):
            raise RuntimeError(f"Failed to save cutout: {out_path}")
        return out_path

    def _handle_extract_result(self, details: dict, source_image: str | None) -> None:
        if not source_image:
            return
        isolate_path = details.get("isolate_path") or ""
        if isolate_path and os.path.isfile(isolate_path):
            cutout_path = isolate_path
        else:
            mask_path = details.get("mask_path") or details.get("mask") or ""
            if not mask_path or not os.path.isfile(mask_path):
                return

            out_dir = details.get("output_dir") or os.path.dirname(mask_path)
            try:
                cutout_path = self._create_cutout_from_mask(source_image, mask_path, out_dir)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Isolate object", f"Could not create isolated image: {e}")
                return

        resp = QtWidgets.QMessageBox.question(
            self,
            "Isolate object",
            "Use the isolated image for detection?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if resp == QtWidgets.QMessageBox.StandardButton.Yes:
            self.load_image(cutout_path)
            self.statusBar().showMessage(f"Loaded cutout: {cutout_path}", 5000)

    def _show_processing_dialog(self, title: str) -> ProcessingDialog:
        if self._progress_dialog is not None:
            self._progress_dialog.allow_close()
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        dialog = ProcessingDialog(self, title)
        dialog.stopRequested.connect(self._stop_current)
        dialog.show()
        self._progress_dialog = dialog
        return dialog

    def _start_worker(self, worker: WorkerBase, *, title: str, pre_logs: list[str] | None = None) -> None:
        if self._thread is not None:
            return
        self._thread = QtCore.QThread(self)
        self._worker = worker

        dialog = self._show_processing_dialog(title)
        self._active_log_widget = dialog.log_widget()
        self._active_stop_btn = dialog.stop_button()
        if pre_logs:
            for line in pre_logs:
                self._append_log(self._active_log_widget, line)

        # Create a broker in the GUI thread to safely route signals.
        # This bypasses PySide6's metaclass bugs with bound methods on Mixins.
        class SignalBroker(QtCore.QObject):
            log = QtCore.Signal(str)
            failed = QtCore.Signal(str)
            finished = QtCore.Signal(object)

        # Keep a reference to the broker to prevent GC
        self._worker_broker = SignalBroker(self)
        self._worker_broker.log.connect(self._on_worker_log)
        self._worker_broker.failed.connect(self._on_worker_failed_slot)
        self._worker_broker.finished.connect(self._on_worker_finished_slot)

        worker.moveToThread(self._thread)

        self._thread.started.connect(worker.run)
        
        worker.log.connect(self._worker_broker.log)
        worker.failed.connect(self._worker_broker.failed)
        worker.finished.connect(self._worker_broker.finished)

        # thread.quit can be directly connected since QThread is a proper QObject
        worker.finished.connect(self._thread.quit, type=QtCore.Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._thread.quit, type=QtCore.Qt.ConnectionType.QueuedConnection)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._cleanup_worker)

        self._set_running(True)
        self._thread.start()

    def _cleanup_worker(self) -> None:
        if self._active_stop_btn is not None:
            self._active_stop_btn.setEnabled(False)
        if self._progress_dialog is not None:
            self._progress_dialog.allow_close()
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        self._thread = None
        self._worker = None
        self._active_stop_btn = None
        self._active_log_widget = None
        self._reset_run_context()
        self._set_running(False)

    def _run_unet(self, start_run: bool = True) -> None:
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "UNet", "Load an image first.")
            return

        image_path = self._state.image_path
        model_path = self._unet_model_edit.text().strip()
        if not model_path:
            QtWidgets.QMessageBox.critical(self, "UNet", "UNet model is required.")
            return
        if not os.path.isfile(model_path):
            QtWidgets.QMessageBox.critical(self, "UNet", f"UNet model not found: {model_path}")
            return
            
        res_dir = getattr(self, "_results_dir", None)
        out_dir = str(res_dir) if res_dir else "results_unet"

        unet_params = UnetParams(
            model_path=model_path,
            output_dir=out_dir,
            threshold=float(self._unet_threshold.value()),
            apply_postprocessing=bool(self._unet_post.isChecked()),
            mode=str(self._unet_mode.currentText()),
            input_size=int(self._unet_input_size.value()),
            tile_overlap=int(self._unet_overlap.value()),
            tile_batch_size=int(self._unet_tile_batch.value()),
        )

        # Check if we should use DINO first
        # If user explicitly calls "UNet + DINO", they want DINO detection first.
        # But we need to know if they manually selected a ROI.
        # Use existing _pending_unet logic? No, let's check if DINO settings are needed.
        # We assume if no ROI is pending, and the mode is "Predict UNet + DINO" (from dialog context),
        # we should use DINO.
        # However, _run_unet is called from _execute_prediction which knows the mode.
        # _execute_prediction sets 'title' but passes nothing else.
        # But we know mode="unet" in PredictDialog means "UNet + DINO".
        # Let's check DINO checkpoint availability.

        # If Manual ROI is active (pending unet), use that.
        if self._pending_unet:
            # Already handled by _on_roi_selected
            return

        # If no pending ROI, we default to DINO detection unless user wants full image.
        # But wait, UNet can also run on full image.
        # The user requested "Unet -> DINO run object -> Unet know region".
        # This implies "UNet + DINO" is THE way this button works now.
        # So we should run UnetDinoWorker.

        # Ask user? Or just do it?
        # Let's assume default behavior for "UNet + DINO" button is DINO-guided.
        # If they want full image, maybe standard UNet running...
        # But "Predict UNet + DINO" is the label.

        try:
            dino_params = self._build_sam_dino_params()
        except Exception:
            dino_params = None

        if dino_params:
            if start_run:
                self._begin_run("current", {"unet": unet_params, "dino": dino_params})
            # Run DINO then UNet
            worker = UnetDinoWorker(image_path, unet_params, dino_params)
            self._start_worker(worker, title="UNet + DINO Processing")
            return

        # Fallback to standard UNet (tiled/full)
        if start_run:
            self._begin_run("current", unet_params)
        self._pending_unet = (image_path, unet_params)
        self._view_tabs.setCurrentIndex(0)  # Overlay
        self._overlay_canvas.set_editable(False)
        self._overlay_canvas.start_roi_selection()
        self._set_roi_selecting(True)
        self.statusBar().showMessage("Select ROI for UNet (or click for full image). DINO params invalid/missing.")

    @QtCore.Slot(object)
    def _on_roi_selected(self, roi_box_obj) -> None:
        if self._pending_unet is None:
            return
        image_path, params = self._pending_unet
        self._pending_unet = None

        roi_box = roi_box_obj if roi_box_obj is None else tuple(int(x) for x in roi_box_obj)
        self._overlay_canvas.set_editable(True)
        self._set_roi_selecting(False)

        log_lines: list[str] = []
        if roi_box is None:
            log_lines.append("ROI: full image")
        else:
            l, t, r, b = roi_box
            log_lines.append(f"ROI: left={l}, top={t}, right={r}, bottom={b}")

        params2 = replace(params, roi_box=roi_box)
        worker = UnetWorker(image_path, params2)
        self._start_worker(worker, title="UNet processing", pre_logs=log_lines)

    @QtCore.Slot()
    def _on_roi_canceled(self) -> None:
        if self._pending_unet is None:
            return
        self._pending_unet = None
        self._overlay_canvas.set_editable(True)
        self._set_roi_selecting(False)
        self.statusBar().showMessage("ROI selection canceled.", 4000)

    def _isolate_object(self) -> None:
        if self._thread is not None:
            QtWidgets.QMessageBox.information(self, "Isolate object", "Already running. Please wait.")
            return
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "Isolate object", "Load an image first.")
            return

        try:
            # Force base SAM+DINO (no delta) for isolation to ensure consistent behavior
            params = self._build_sam_dino_params(force_use_delta=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Isolate object", str(e))
            return
            
        last_labels = getattr(self, "_isolate_last_labels", "")
        last_crop = getattr(self, "_isolate_last_crop", False)
        last_white = getattr(self, "_isolate_last_white", False)
        
        dlg = IsolateDialog(self, labels=last_labels, crop=last_crop, white=last_white)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
            
        values = dlg.get_values()
        labels_text = values["labels"]
        crop_to_bbox = values["crop"]
        outside_value = 255 if values["white"] else 0
        
        # Update settings for persistence
        self._isolate_last_labels = labels_text
        self._isolate_last_crop = crop_to_bbox
        self._isolate_last_white = values["white"]
        self._schedule_save_settings()

        target_labels = [t.strip() for t in labels_text.split(",") if t.strip()]
        
        # If user provided labels, use them as the detection queries for this run.
        # This ensures we actually look for what we want to isolate.
        if target_labels:
            params = replace(params, text_queries=target_labels)

        self._post_run_action = {"type": "isolate", "source_image": self._state.image_path}
        worker = SamDinoIsolateWorker(
            self._state.image_path,
            params,
            target_labels=target_labels,
            outside_value=outside_value,
            crop_to_bbox=crop_to_bbox,
        )
        self._start_worker(worker, title="Isolate object")

    def _run_sam_dino(self, checked: bool = False, *, force_use_delta: bool | None = None) -> None:
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "SAM+DINO", "Load an image first.")
            return

        try:
            params = self._build_sam_dino_params(force_use_delta=force_use_delta)
            self._begin_run("current", params)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM+DINO", str(e))
            return

        self._post_run_action = None
        worker = SamDinoWorker(self._state.image_path, params)
        self._start_worker(worker, title="SAM + DINO processing")

    def _run_batch_unet(self, image_paths: list[str] | None = None) -> None:
        images = image_paths if image_paths is not None else self._folder_images
        if not images:
            QtWidgets.QMessageBox.information(self, "Batch UNet", "No images in folder list. Open a folder first.")
            return

        model_path = self._unet_model_edit.text().strip()
        if not model_path:
            QtWidgets.QMessageBox.critical(self, "Batch UNet", "UNet model is required.")
            return
        if not os.path.isfile(model_path):
             QtWidgets.QMessageBox.critical(self, "Batch UNet", f"UNet model not found: {model_path}")
             return

        res_dir = getattr(self, "_results_dir", None)
        out_dir = str(res_dir) if res_dir else "results_unet"

        params = UnetParams(
            model_path=model_path,
            output_dir=out_dir,
            threshold=float(self._unet_threshold.value()),
            apply_postprocessing=bool(self._unet_post.isChecked()),
            mode=str(self._unet_mode.currentText()),
            input_size=int(self._unet_input_size.value()),
            tile_overlap=int(self._unet_overlap.value()),
            tile_batch_size=int(self._unet_tile_batch.value()),
        )

        # Ensure output dir exists
        os.makedirs(out_dir, exist_ok=True)

        self._begin_run("folder", params)
        worker = BatchUnetWorker(images, params)
        self._start_worker(worker, title="Batch UNet Processing")

    def _run_batch_sam_dino(
        self,
        image_paths: list[str] | None = None,
        *,
        force_use_delta: bool | None = None,
    ) -> None:
        images = image_paths if image_paths is not None else self._folder_images
        if not images:
            QtWidgets.QMessageBox.information(self, "Batch SAM+DINO", "No images in folder list. Open a folder first.")
            return

        try:
            params = self._build_sam_dino_params(force_use_delta=force_use_delta)
            self._begin_run("folder", params)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM+DINO", str(e))
            return

        os.makedirs(params.output_dir, exist_ok=True)

        worker = BatchSamDinoWorker(images, params)
        self._start_worker(worker, title="Batch SAM+DINO Processing")

    def _stop_current(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()
