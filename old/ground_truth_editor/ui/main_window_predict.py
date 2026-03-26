from __future__ import annotations

import csv
import datetime
import json
import os
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from image_io import ImageIoError, load_image, load_mask
from inference_api.contracts import InferenceRequest
from inference_api.editor_bridge import build_editor_request, prediction_title

from .dialogs import PredictRunDialog, ProcessingDialog, IsolateDialog


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

    def _editor_settings(self) -> dict:
        return self._collect_settings()

    def _default_output_dir(self, mode: str) -> str:
        res_dir = getattr(self, "_results_dir", None)
        if res_dir:
            return str(res_dir)
        return "results_unet" if str(mode).strip().lower().startswith("unet") else "results_sam_dino"

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

        json_path = out_dir / "run.json"
        try:
            payload = {
                "run_id": run_id,
                "created_at": datetime.datetime.now().isoformat(),
                "scope": scope,
                "params": p_dict,
            }
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save run json: {e}")

    def _begin_run(self, scope: str, params: object = None) -> None:
        state = getattr(self, "_state", None)
        current_image = state.image_path if state else ""
        is_roi_run = bool(getattr(self, "_current_roi_box", None)) and str(scope or "current") == "current"

        if is_roi_run and current_image:
            if getattr(self, "_roi_session_image_path", None) != current_image:
                self._roi_session_run_id = None
                self._roi_session_started_at = None
                self._roi_session_image_path = current_image
            if self._current_run_id is None and getattr(self, "_roi_session_run_id", None):
                self._current_run_id = self._roi_session_run_id
                self._current_run_scope = str(scope or "current")
                self._current_run_started_at = self._roi_session_started_at

        if self._current_run_id is None:
            self._current_run_id = self._new_run_id()
            self._current_run_scope = str(scope or "current")
            self._current_run_started_at = datetime.datetime.now().isoformat(timespec="seconds")
            
            if params:
                self._save_run_metadata(self._current_run_id, scope, params)

            if is_roi_run and current_image:
                self._roi_session_run_id = self._current_run_id
                self._roi_session_image_path = current_image
                self._roi_session_started_at = self._current_run_started_at

    def _reset_run_context(self) -> None:
        self._current_run_id = None
        self._current_run_scope = None
        self._current_run_started_at = None

    def _sanitize_name(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_") or "image"

    def _run_asset_key(self, image_path: str, rel_image: str) -> str:
        source = rel_image or str(image_path or "")
        source = source.replace("\\", "__").replace("/", "__")
        return self._sanitize_name(source)

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

    def _append_jsonl_rows(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")

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

        image_key = self._run_asset_key(str(img_path), rel_image)
        existing_count = 0
        try:
            data_csv = data_dir / "data.csv"
            for row in self._history_rows_from_csv(data_csv):
                if (row.get("run_id") or "") != run_id:
                    continue
                if (row.get("image_rel") or "") != rel_image:
                    continue
                existing_count += 1
        except Exception:
            existing_count = 0

        image_asset_rel = ""

        overlay_rel = ""
        overlay_src = details.get("overlay_path")
        if overlay_src:
            overlay_rel = self._resolve_service_path(overlay_src)

        isolate_rel = ""
        isolate_src = details.get("isolate_path")
        if isolate_src:
            isolate_rel = self._resolve_service_path(isolate_src)

        full_mask_rel = ""
        full_mask_src = details.get("mask_path") or details.get("mask") or ""
        if full_mask_src:
            full_mask_rel = self._resolve_service_path(full_mask_src)

        rows: list[dict] = []
        for idx, det in enumerate(dets):
            mask_arr, _src_path = self._mask_from_detection(det)
            if mask_arr is None:
                continue

            if int(mask_arr.max()) <= 1:
                import numpy as np

                mask_arr = (mask_arr.astype(np.uint8) * 255)

            det_id = f"det_{existing_count + idx:03d}"
            label_safe = self._sanitize_name(det.get("label") or "mask")
            mask_name = f"{image_key}__{det_id}__{label_safe}.png"
                
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
                    "detection_id": det_id,
                    "model": str(model_name),
                    "label": str(label),
                    "score": score,
                    "image_rel": rel_image,
                    "image_asset_rel": image_asset_rel,
                    "mask_rel": f"{run_id}/mask/{mask_name}",
                    "full_mask_rel": full_mask_rel,
                    "overlay_rel": overlay_rel,
                    "isolate_rel": isolate_rel,
                    "box": box_text,
                }
            )

        if not rows:
            return

        data_csv = data_dir / "data.csv"
        detections_csv = data_dir / "detections.csv"
        detections_jsonl = data_dir / "detections.jsonl"
        fields = [
            "run_id",
            "created_at",
            "detection_id",
            "model",
            "label",
            "score",
            "image_rel",
            "image_asset_rel",
            "mask_rel",
            "full_mask_rel",
            "overlay_rel",
            "isolate_rel",
            "box",
        ]
        self._append_csv_rows(data_csv, rows, fields)
        self._append_csv_rows(detections_csv, rows, fields)
        self._append_jsonl_rows(detections_jsonl, rows)

        run_csv = out_dir / f"{run_id}_lan_quet_workspace.csv"
        self._append_csv_rows(run_csv, rows, fields)
        self._populate_history_list()

    def _store_detections_from_details(self, details: dict, *, update_view: bool, append: bool = False) -> None:
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
                det["_checked"] = True
            if append:
                merged_detections = list(self._image_detections.get(image_path, []))
                merged_detections.extend(detections)
            else:
                merged_detections = detections
            self._image_detections[image_path] = merged_detections
            if hasattr(self, "_image_detections_all"):
                if append:
                    merged_all = list(self._image_detections_all.get(image_path, []))
                    merged_all.extend(detections)
                    self._image_detections_all[image_path] = merged_all
                else:
                    self._image_detections_all[image_path] = merged_detections
            if hasattr(self, "_history_view_detections"):
                self._history_view_detections.pop(image_path, None)
            if update_view and self._state:
                if os.path.normpath(str(image_path)) == os.path.normpath(str(self._state.image_path)):
                    self._populate_mask_list()

    def _apply_visual_result(self, details: dict) -> None:
        # Load and set DINO boxes if available
        detections = details.get("detections", [])
        
        boxes_to_draw = []
        if hasattr(self, "_active_highlight_boxes"):
            boxes_to_draw = list(self._active_highlight_boxes)
            # Remove the temporary 'ROI' box if it's there
            boxes_to_draw = [b for b in boxes_to_draw if b[1] != "ROI"]
            
        if detections and len(detections) > 0:
            from PySide6.QtCore import QRectF
            for det in detections:
                box = det.get("box")
                label = det.get("label", "")
                score = det.get("score", 0.0)
                display_text = f"{label} {score:.2f}" if label else ""
                 
                if box and len(box) == 4:
                    hb = QRectF(box[0], box[1], box[2]-box[0], box[3]-box[1])
                    boxes_to_draw.append((hb, display_text))

        # Keep the Image tab anchored to the original loaded image.
        base_image = self._overlay_canvas.image()
        if not base_image.isNull():
            self._image_canvas.set_image(base_image)

        self._active_highlight_boxes = boxes_to_draw
        self._image_canvas.set_highlight_boxes(boxes_to_draw)
        self._overlay_canvas.set_highlight_boxes(boxes_to_draw)

        overlay_path = self._resolve_service_path(details.get("overlay_path") or "")
        if overlay_path and os.path.isfile(overlay_path):
            if self._active_log_widget is not None:
                self._append_log(self._active_log_widget, f"Saved overlay: {overlay_path}")

        mask_path = self._resolve_service_path(details.get("mask_path") or details.get("mask") or "")
        if mask_path and self._state is not None and os.path.isfile(mask_path):
            try:
                self._mask_path = Path(mask_path)
                self._sync_path_labels()
                self._view_tabs.setCurrentIndex(0)
                if detections and hasattr(self, "_update_composite_mask"):
                    self._update_composite_mask()
                else:
                    loaded = load_mask(mask_path, (self._state.image_w, self._state.image_h))
                    self._overlay_canvas.set_mask(loaded.mask)
                    self._sync_mask_views()
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

    def _merge_with_existing_mask(self, details: dict) -> None:
        if not self._state or self._overlay_canvas.mask().isNull():
            return
            
        image_path = details.get("image_path")
        if not image_path or os.path.normpath(str(image_path)) != os.path.normpath(str(self._state.image_path)):
            return
            
        mask_path = details.get("mask_path")
        if not mask_path or not os.path.isfile(mask_path):
            return
            
        import cv2
        import numpy as np
        from PySide6 import QtGui
        
        existing_qimage = self._overlay_canvas.mask().convertToFormat(QtGui.QImage.Format_Grayscale8)
        w, h = existing_qimage.width(), existing_qimage.height()
        bpl = existing_qimage.bytesPerLine()
        ptr = existing_qimage.constBits()
        existing_arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))[:, :w].copy()
        
        new_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if new_mask is None:
            return
            
        if new_mask.shape != existing_arr.shape:
            new_mask = cv2.resize(new_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        merged_mask = np.maximum(existing_arr, new_mask)
        cv2.imwrite(mask_path, merged_mask)
        
        overlay_path = details.get("overlay_path")
        if overlay_path and os.path.isfile(overlay_path):
            bgr = cv2.imread(image_path)
            if bgr is not None:
                alpha = 0.45
                bool_mask = merged_mask > 0
                overlay = bgr.copy()
                overlay[bool_mask] = (overlay[bool_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8)
                cv2.imwrite(overlay_path, overlay)
                
        isolate_path = details.get("isolate_path")
        if isolate_path and os.path.isfile(isolate_path):
            bgr = cv2.imread(image_path)
            if bgr is not None:
                bool_mask = merged_mask > 0
                isolate = np.zeros_like(bgr)
                isolate[bool_mask] = bgr[bool_mask]
                cv2.imwrite(isolate_path, isolate)

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
                    if is_current:
                        self._merge_with_existing_mask(res)
                    self._store_detections_from_details(res, update_view=is_current, append=False)
                    self._persist_detection_results(res)
                    if is_current:
                        current_result = res
                if current_result is not None:
                    self._apply_visual_result(current_result)
            if self._active_log_widget is not None:
                self._append_log(self._active_log_widget, "Batch Processing COMPLETE.")
            return

        append_results = bool(getattr(self, "_append_results_mode", False))
        self._merge_with_existing_mask(details)
        self._store_detections_from_details(details, update_view=True, append=append_results)
        self._persist_detection_results(details)
        self._apply_visual_result(details)
        self._append_results_mode = False

        if self._active_log_widget is not None:
            self._append_log(self._active_log_widget, "DONE")

        post_action = self._post_run_action
        self._post_run_action = None
        if post_action and post_action.get("type") in {"extract", "isolate"}:
            self._handle_extract_result(details, post_action.get("source_image"))

        self._refresh_ui_state()

    def _start_predict_roi(self) -> None:
        if self._active_job_id is not None:
            QtWidgets.QMessageBox.information(self, "Predict", "Already running. Please wait.")
            return
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.information(self, "Predict", "Load an image first.")
            return

        self._pending_predict_roi = True
        self._view_tabs.setCurrentIndex(0)  # Overlay
        self._overlay_canvas.set_editable(False)
        self._overlay_canvas.start_roi_selection()
        self._set_roi_selecting(True)
        self.statusBar().showMessage("Select ROI for prediction. Esc to cancel.")

    def _open_predict_mode_dialog(self) -> None:
        if self._active_job_id is not None:
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
        if self._active_job_id is not None:
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
        if self._active_job_id is not None:
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
        title = prediction_title(mode)

        if not self._ensure_settings_ready(mode):
            return

        if scope == "folder":
            images = folder_images if folder_images is not None else self._folder_images
            if not images:
                QtWidgets.QMessageBox.information(self, "Predict", "No images in folder list. Open a folder first.")
                self._focus_folder_filter()
                return

        if mode in {"sam_dino", "sam_dino_ft"}:
            use_delta = (mode == "sam_dino_ft")
            if scope == "current":
                self._run_sam_dino(force_use_delta=use_delta)
            else:
                self._run_batch_sam_dino(images, force_use_delta=use_delta)
            return

        if mode in {"sam_only", "sam_only_ft"}:
            use_delta = (mode == "sam_only_ft")
            if scope == "current":
                self._run_sam_only(force_use_delta=use_delta)
            else:
                self._run_batch_sam_only(images, force_use_delta=use_delta)
            return

        if mode == "sam_tiled":
            if scope == "current":
                self._run_sam_tiled()
            else:
                self._run_batch_sam_tiled(images)
            return

        if mode == "unet_only":
            if scope == "current":
                self._run_unet()
            else:
                self._run_batch_unet(images)
            return
        if mode == "unet_dino":
            if scope == "current":
                self._run_unet_dino()
            else:
                self._run_batch_unet_dino(images)
            return
        QtWidgets.QMessageBox.critical(self, "Predict", f"Unknown mode: {mode}")

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

    def _submit_inference_request(self, request: InferenceRequest, *, title: str, pre_logs: list[str] | None = None) -> None:
        if self._active_job_id is not None:
            return
        dialog = self._show_processing_dialog(title)
        self._active_log_widget = dialog.log_widget()
        self._active_stop_btn = dialog.stop_button()
        if pre_logs:
            for line in pre_logs:
                self._append_log(self._active_log_widget, line)
        self._active_job_id = self._inference_api.submit(request)
        self._active_job_workflow = request.workflow
        self._set_running(True)
        self._predict_poll_timer.start()

    @QtCore.Slot()
    def _cleanup_job(self) -> None:
        self._predict_poll_timer.stop()
        if self._active_stop_btn is not None:
            self._active_stop_btn.setEnabled(False)
        if self._progress_dialog is not None:
            self._progress_dialog.allow_close()
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        self._active_job_id = None
        self._active_job_workflow = None
        self._active_stop_btn = None
        self._active_log_widget = None
        self._reset_run_context()
        self._set_running(False)

    @QtCore.Slot()
    def _poll_inference_events(self) -> None:
        job_id = self._active_job_id
        if not job_id:
            self._predict_poll_timer.stop()
            return
        for event in self._inference_api.drain_events(job_id):
            if event.type == "progress" and event.message:
                self._on_worker_log(event.message)
                continue
            if event.type == "partial_result" and event.result is not None:
                details = event.result.to_dict()
                if details:
                    image_path = details.get("image_path") or (self._state.image_path if self._state else "")
                    is_current = bool(self._state) and os.path.normpath(str(image_path)) == os.path.normpath(str(self._state.image_path))
                    self._store_detections_from_details(details, update_view=is_current, append=False)
                    if is_current:
                        self._apply_visual_result(details)
                continue
            if event.type == "completed" and event.result is not None:
                self._on_worker_finished_slot(event.result.to_dict())
                self._cleanup_job()
                return
            if event.type == "failed":
                self._on_worker_failed_slot(event.error or event.message or "Unknown inference error.")
                self._cleanup_job()
                return
            if event.type == "cancelled":
                self._on_worker_finished_slot({"stopped": True})
                self._cleanup_job()
                return

    def _submit_editor_mode(
        self,
        mode: str,
        *,
        title: str,
        image_path: str | None = None,
        image_paths: list[str] | None = None,
        roi_box: tuple[int, int, int, int] | None = None,
        pre_logs: list[str] | None = None,
        target_labels: list[str] | None = None,
        outside_value: int | None = None,
        crop_to_bbox: bool | None = None,
        max_depth: int | None = None,
        min_box_px: int | None = None,
    ) -> dict:
        request = build_editor_request(
            mode,
            self._editor_settings(),
            image_path=image_path,
            image_paths=image_paths,
            roi_box=roi_box,
            output_dir=self._default_output_dir(mode),
            target_labels=target_labels,
            outside_value=outside_value,
            crop_to_bbox=crop_to_bbox,
            max_depth=max_depth,
            min_box_px=min_box_px,
        )
        self._submit_inference_request(request, title=title, pre_logs=pre_logs)
        return dict(request.params)

    def _run_unet(self, start_run: bool = True) -> None:
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "UNet", "Load an image first.")
            return
        roi_box = getattr(self, "_current_roi_box", None)
        try:
            params = self._submit_editor_mode(
                "unet_only",
                title="UNet processing",
                image_path=self._state.image_path,
                roi_box=roi_box,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "UNet", str(e))
            return
        if start_run:
            self._begin_run("current", params)

    def _run_unet_dino(self, start_run: bool = True) -> None:
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "UNet + DINO", "Load an image first.")
            return
        roi_box = getattr(self, "_current_roi_box", None)
        try:
            params = self._submit_editor_mode(
                "unet_dino",
                title="UNet + DINO Processing",
                image_path=self._state.image_path,
                roi_box=roi_box,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "UNet + DINO", str(e))
            return
        if start_run:
            self._begin_run("current", params)

    @QtCore.Slot(object)
    def _on_roi_selected(self, roi_box_obj) -> None:
        if getattr(self, "_pending_predict_roi", False):
            self._pending_predict_roi = False
            self._overlay_canvas.set_editable(True)
            self._set_roi_selecting(False)
            
            if roi_box_obj is None:
                self.statusBar().showMessage("ROI selection canceled.", 4000)
                return
                
            roi_box = tuple(int(x) for x in roi_box_obj)
            
            # Sync highlight box to other views
            from PySide6.QtCore import QRectF
            hb = QRectF(roi_box[0], roi_box[1], roi_box[2]-roi_box[0], roi_box[3]-roi_box[1])
            
            # We add a green box to show the ROI being predicted, it is just visual 
            # and will be replaced by the results (or we can just append it).
            # For now just set it as the single active box until prediction completes.
            boxes_to_draw = [(hb, "ROI")]
            self._active_highlight_boxes = boxes_to_draw
            self._image_canvas.set_highlight_boxes(boxes_to_draw)
            self._overlay_canvas.set_highlight_boxes(boxes_to_draw)
            
            dlg = PredictRunDialog(
                self,
                has_image=True,
                has_folder=False,
            )
            dlg.rb_scope_folder.setVisible(False)
            dlg.rb_scope_current.setChecked(True)
            dlg.rb_scope_current.setText(f"Current Image (ROI: {roi_box})")
            
            if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return

            mode, _ = dlg.get_result()
            self._current_roi_box = roi_box
            self._append_results_mode = True
            
            def do_run():
                try:
                    self._execute_prediction(mode, "current")
                finally:
                    self._current_roi_box = None
                    
            QtCore.QTimer.singleShot(50, do_run)
            return

    @QtCore.Slot()
    def _on_roi_canceled(self) -> None:
        if getattr(self, "_pending_predict_roi", False):
            self._pending_predict_roi = False
            self._append_results_mode = False
            self._overlay_canvas.set_editable(True)
            self._set_roi_selecting(False)
            self.statusBar().showMessage("ROI selection canceled.", 4000)
            return

    def _isolate_object(self) -> None:
        if self._active_job_id is not None:
            QtWidgets.QMessageBox.information(self, "Isolate object", "Already running. Please wait.")
            return
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "Isolate object", "Load an image first.")
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

        self._post_run_action = {"type": "isolate", "source_image": self._state.image_path}
        try:
            params = self._submit_editor_mode(
                "isolate",
                title="Isolate object",
                image_path=self._state.image_path,
                target_labels=target_labels,
                outside_value=outside_value,
                crop_to_bbox=crop_to_bbox,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Isolate object", str(e))
            return
        self._begin_run("current", params)

    def _run_sam_dino(self, checked: bool = False, *, force_use_delta: bool | None = None) -> None:
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "SAM+DINO", "Load an image first.")
            return

        try:
            mode = "sam_dino_ft" if force_use_delta else "sam_dino"
            params = self._submit_editor_mode(mode, title="SAM + DINO processing", image_path=self._state.image_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM+DINO", str(e))
            return

        self._begin_run("current", params)
        self._post_run_action = None

    def _run_batch_unet(self, image_paths: list[str] | None = None) -> None:
        images = image_paths if image_paths is not None else self._folder_images
        if not images:
            QtWidgets.QMessageBox.information(self, "Batch UNet", "No images in folder list. Open a folder first.")
            return

        try:
            params = self._submit_editor_mode("unet_only", title="Batch UNet Processing", image_paths=list(images))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Batch UNet", str(e))
            return

        self._begin_run("folder", params)

    def _run_batch_unet_dino(self, image_paths: list[str] | None = None) -> None:
        images = image_paths if image_paths is not None else self._folder_images
        if not images:
            QtWidgets.QMessageBox.information(self, "Batch UNet + DINO", "No images in folder list. Open a folder first.")
            return
        try:
            params = self._submit_editor_mode("unet_dino", title="Batch UNet + DINO Processing", image_paths=list(images))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Batch UNet + DINO", str(e))
            return
        self._begin_run("folder", params)

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
            mode = "sam_dino_ft" if force_use_delta else "sam_dino"
            params = self._submit_editor_mode(mode, title="Batch SAM+DINO Processing", image_paths=list(images))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM+DINO", str(e))
            return

        self._begin_run("folder", params)

    def _run_sam_only(self, *, force_use_delta: bool | None = None) -> None:
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "SAM Only", "Load an image first.")
            return

        try:
            mode = "sam_only_ft" if force_use_delta else "sam_only"
            label = "SAM Only + Finetune" if force_use_delta else "SAM Only"
            params = self._submit_editor_mode(mode, title=f"{label} processing", image_path=self._state.image_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM Only", str(e))
            return

        self._begin_run("current", params)
        self._post_run_action = None

    def _run_batch_sam_only(
        self,
        image_paths: list[str] | None = None,
        *,
        force_use_delta: bool | None = None,
    ) -> None:
        images = image_paths if image_paths is not None else self._folder_images
        if not images:
            QtWidgets.QMessageBox.information(self, "Batch SAM Only", "No images in folder list. Open a folder first.")
            return

        try:
            label = "SAM Only + Finetune" if force_use_delta else "SAM Only"
            mode = "sam_only_ft" if force_use_delta else "sam_only"
            params = self._submit_editor_mode(mode, title=f"Batch {label} Processing", image_paths=list(images))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM Only", str(e))
            return

        self._begin_run("folder", params)

    def _stop_current(self) -> None:
        if self._active_job_id is None:
            return
        self._inference_api.cancel(self._active_job_id)

    def _run_sam_tiled(
        self,
        *,
        target_labels: list[str] | None = None,
        tile_size: int = 640,
        tile_overlap: float = 0.25,
    ) -> None:
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "SAM Tiled", "Load an image first.")
            return

        try:
            params = self._submit_editor_mode(
                "sam_tiled",
                title=f"SAM+DINO Tiled (target={target_labels or ['crack']})",
                image_path=self._state.image_path,
                target_labels=target_labels or ["crack"],
                max_depth=3,
                min_box_px=48,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM Tiled", str(e))
            return

        self._begin_run("current", params)
        self._post_run_action = None

    def _run_batch_sam_tiled(
        self,
        image_paths: list[str] | None = None,
        *,
        target_labels: list[str] | None = None,
        tile_size: int = 640,
        tile_overlap: float = 0.25,
    ) -> None:
        images = image_paths if image_paths is not None else self._folder_images
        if not images:
            QtWidgets.QMessageBox.information(self, "Batch SAM Tiled", "No images in folder list.")
            return

        try:
            labels = target_labels or ["crack"]
            params = self._submit_editor_mode(
                "sam_tiled",
                title=f"Batch SAM+DINO Tiled (target={labels})",
                image_paths=list(images),
                target_labels=labels,
                max_depth=3,
                min_box_px=48,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM Tiled", str(e))
            return

        self._begin_run("folder", params)
