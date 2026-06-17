from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from inference_api.contracts import InferenceRequest
from inference_api.prediction_models import (
    DETECTION_NONE,
    SEGMENTATION_SAM,
    SEGMENTATION_SAM_LORA,
    SEGMENTATION_UNET,
    TASK_GROUP_CRACK_ONLY,
    TASK_GROUP_MORE_DAMAGE,
    PredictionConfig,
)
from inference_api.request_builder import build_prediction_request
from inference_api.workflow_resolver import resolve_workflow
from ui.canvas.items.mask_item import MaskPixmapItem
from ui.models.job import JobKind, JobUpdate
from ui.models.layer import MaskRef
from ui.services.box_segment_pipeline import run_unet_boxes


class SegmentMixin:
    def _wire_segment_signals(self) -> None:
        self._segment_panel.runRequested.connect(self._run_segment)
        self._segment_panel.cancelRequested.connect(self._cancel_segment)
        self._segment_panel.opacity.valueChanged.connect(self._on_mask_opacity)

        # Wire InferenceClient signals for segment jobs
        self._infer.jobCompleted.connect(self._on_infer_completed)
        self._infer.jobFailed.connect(self._on_infer_failed)
        self._infer.jobCancelled.connect(self._on_infer_cancelled)
        self._infer.logEmitted.connect(self._on_infer_log)
        self._infer.progressEmitted.connect(self._on_infer_progress)

    def _run_segment(self) -> None:
        if self._image_path is None:
            QtWidgets.QMessageBox.information(self, "No image", "Open an image first.")
            return

        group = self._active_group()
        rows = list(group.rows) if group is not None else []
        self._seg_group_id = group.layer_id if group is not None and rows else None

        backend = self._segment_panel.current_backend()
        task_group_data = self._segment_panel.task_group_combo.currentData()

        # Update settings with current form values per active backend.
        if backend == "sam":
            sam_ckpt = self._segment_panel.sam_options.ckpt.text().strip()
            if sam_ckpt:
                self._settings.sam_checkpoint = sam_ckpt
            self._settings.sam_model_type = self._segment_panel.sam_options.model_type.currentText()
        elif backend == "sam_lora":
            opts = self._segment_panel.sam_lora_options
            base_ckpt = opts.base_ckpt.text().strip()
            lora_ckpt = opts.lora_ckpt.text().strip()
            refine_ckpt = opts.refine_ckpt.text().strip()
            predict_mode = opts.predict_mode.currentData() or "tile_full_box"
            if base_ckpt:
                self._settings.sam_checkpoint = base_ckpt
                self._settings.sam_lora_base_checkpoint = base_ckpt
            if lora_ckpt:
                self._settings.sam_lora_checkpoint = lora_ckpt
            self._settings.sam_lora_rank = int(opts.lora_rank.value())
            self._settings.sam_lora_predict_mode = str(predict_mode)
            self._settings.sam_lora_refine_checkpoint = refine_ckpt
            self._settings.sam_lora_refine_rank = int(opts.refine_rank.value())
            if predict_mode == "coarse_refine" and not refine_ckpt:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Refine checkpoint required",
                    "coarse_refine mode requires a refine delta checkpoint.",
                )
                return
        elif backend == "unet":
            unet_ckpt = self._segment_panel.unet_options.ckpt.text().strip()
            if unet_ckpt:
                self._settings.unet_checkpoint = unet_ckpt
            self._settings.unet_threshold = float(self._segment_panel.unet_options.threshold.value())

            if not rows:
                self._append_log("No boxes selected: running UNet full-image segment.")
            else:
                job = self._jobs.submit(
                    JobKind.segment,
                    label="segment unet boxes",
                    params={"backend": backend, "boxes": len(rows)},
                )
                self._seg_job_id = job.id
                self._jobs.mark_running(job.id)
                self._segment_panel.run_button.setEnabled(False)
                self._segment_panel.cancel_button.setEnabled(True)
                self._append_log("Segment started: unet box-scoped")
                try:
                    self._run_unet_box_segment(job.id, rows)
                except Exception as exc:
                    self._jobs.fail(job.id, str(exc))
                    self._segment_panel.run_button.setEnabled(True)
                    self._segment_panel.cancel_button.setEnabled(False)
                    self._append_log(f"Segment ERROR: {exc}")
                    QtWidgets.QMessageBox.warning(self, "Segment failed", str(exc))
                return

        # Build PredictionConfig
        seg_model = {
            "sam": SEGMENTATION_SAM,
            "sam_lora": SEGMENTATION_SAM_LORA,
            "unet": SEGMENTATION_UNET,
        }.get(backend, SEGMENTATION_SAM)

        task_group = task_group_data or TASK_GROUP_CRACK_ONLY

        config = PredictionConfig(
            task_group=task_group,
            segmentation_model=seg_model,
            detection_model=DETECTION_NONE,
        )

        # Build output dir for masks
        self._seg_tmp_dir = tempfile.mkdtemp(prefix="dd_seg_")

        try:
            request = build_prediction_request(
                config,
                self._settings.as_inference_settings(),
                image_path=str(self._image_path),
                output_dir=self._seg_tmp_dir,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Configuration error", str(exc))
            return

        # If boxes exist, prompt box-capable segmenters. Without boxes, keep the
        # request box-free so the backend runs whole-image segmentation.
        if rows:
            request = self._inject_boxes_into_request(request)
        else:
            self._append_log(f"No boxes selected: running {backend} full-image segment.")

        job = self._jobs.submit(
            JobKind.segment,
            label=f"segment {backend}",
            params={"backend": backend},
        )
        self._seg_job_id = job.id
        self._seg_infer_id = self._infer.submit(request)
        self._seg_infer_to_job[self._seg_infer_id] = job.id
        self._jobs.mark_running(job.id)

        self._segment_panel.run_button.setEnabled(False)
        self._segment_panel.cancel_button.setEnabled(True)
        self._append_log(f"Segment started: {backend}")

    def _run_unet_box_segment(self, job_id: str, rows: list) -> None:
        if self._image_path is None:
            raise RuntimeError("No image path")
        opts = self._segment_panel.unet_options
        output_dir = tempfile.mkdtemp(prefix="dd_seg_unet_boxes_")
        self._seg_tmp_dir = output_dir
        boxes = [
            {
                "box": [float(r.x1), float(r.y1), float(r.x2), float(r.y2)],
                "label": r.group_name or r.label or "crack",
                "score": float(r.score),
                "detector_name": getattr(r, "detector_name", "detect"),
                "det_idx": idx,
            }
            for idx, r in enumerate(rows)
        ]
        result = run_unet_boxes(
            self._image_path,
            boxes,
            output_dir,
            model_path=opts.ckpt.text().strip() or None,
            device=str(self._settings.device),
            threshold=float(opts.threshold.value()),
            input_size=int(opts.input_size.value()),
            tile_overlap=int(round(float(opts.overlap.value()) * float(opts.input_size.value()))),
            apply_postprocessing=True,
            log_fn=self._append_log,
        )
        stats = list(result.get("stats") or [])
        for item in stats:
            self._append_log(
                "UNet mask px: "
                f"area={item.get('area_px')} outside={item.get('outside_px')} "
                f"outside_ratio={float(item.get('outside_ratio') or 0):.4f} "
                f"full_image_ratio={float(item.get('full_image_ratio') or 0):.4f} "
                f"box_fill_ratio={float(item.get('box_fill_ratio') or 0):.4f}"
            )
        self._jobs.complete(job_id, result=result)
        self._segment_panel.run_button.setEnabled(True)
        self._segment_panel.cancel_button.setEnabled(False)
        self._apply_segment_payload(result, f"unet_{job_id[:6]}")

    def _inject_boxes_into_request(self, request: InferenceRequest) -> InferenceRequest:
        """Pass the active layer's detection boxes as segmentation prompts."""
        group = self._active_group()
        rows = group.rows if group is not None else []
        if not rows:
            return request
        force_crack = request.resolved.get("segmentation_model") == SEGMENTATION_SAM_LORA
        boxes = []
        for r in rows:
            label = "crack" if force_crack else str(getattr(r, "label", "") or r.group_name or "object")
            boxes.append(
                {
                    "box": [int(r.x1), int(r.y1), int(r.x2), int(r.y2)],
                    "label": label,
                    "score": float(r.score),
                    "detector_name": str(getattr(r, "detector_name", "") or "detect"),
                    "group_name": str(getattr(r, "group_name", "") or ""),
                }
            )
        self._append_log("Segment boxes: " + ", ".join(str(item["label"]) for item in boxes[:8]))
        params = dict(request.params)
        # SAM / SAM-LoRA read from params["sam"]; UNet reads from params["unet"].
        for key in ("sam", "unet"):
            if key in params and isinstance(params[key], dict):
                sub = dict(params[key])
                sub["boxes"] = boxes
                if force_crack and key == "sam":
                    sub["task_group"] = TASK_GROUP_CRACK_ONLY
                params[key] = sub
        return InferenceRequest(
            workflow=request.workflow,
            image_path=request.image_path,
            image_paths=request.image_paths,
            roi_box=request.roi_box,
            params=params,
            selection=request.selection,
            resolved=request.resolved,
            client_tag=request.client_tag,
            source=request.source,
        )

    def _cancel_segment(self) -> None:
        if hasattr(self, "_seg_infer_id") and self._seg_infer_id:
            self._infer.cancel(self._seg_infer_id)
            self._append_log("Segment cancel requested...")

    def _on_mask_opacity(self, value: float) -> None:
        self._canvas.set_masks_opacity(value)

    @QtCore.Slot(str, object)
    def _on_infer_completed(self, infer_id: str, result: object) -> None:
        job_id = self._seg_infer_to_job.pop(infer_id, None)
        if job_id is None:
            return

        self._jobs.complete(job_id, result=result)
        self._segment_panel.run_button.setEnabled(True)
        self._segment_panel.cancel_button.setEnabled(False)

        if result is None:
            self._append_log("Segment finished (no result).")
            return

        # Detections live in the payload dict (list of dicts), not the dataclass field.
        payload = result.to_dict() if hasattr(result, "to_dict") else {}
        self._apply_segment_payload(payload, infer_id)

    def _apply_segment_payload(self, payload: dict, infer_id: str) -> None:
        detections = list(payload.get("detections") or [])
        opacity = float(self._segment_panel.opacity.value())

        # Masks attach to the group that was segmented (box + mask same layer).
        group_id = getattr(self, "_seg_group_id", None)
        group = self._group_by_id(group_id)

        image_rect = self._canvas.image_rect()

        count = 0
        for det in detections:
            mask_path = self._resolve_det_mask_path(det, payload, infer_id, count)
            if not mask_path or not Path(mask_path).exists():
                continue
            bbox = det.get("box") or [0, 0, 0, 0]
            bbox_rect = self._mask_render_rect(mask_path, bbox, image_rect, bool(det.get("mask_is_crop")))

            item = MaskPixmapItem(mask_path, bbox_rect, opacity=opacity)
            if group_id is not None:
                self._canvas.add_group_mask(group_id, item)
            else:
                self._canvas.add_mask_item(item)
            if group is not None:
                group.mask_refs.append(MaskRef(mask_path=mask_path, box=[float(v) for v in bbox[:4]]))
            count += 1

        self._append_log(f"Segment done: {count} mask(s) applied.")

    def _mask_render_rect(
        self,
        mask_path: str,
        bbox: list | tuple,
        image_rect: QtCore.QRectF,
        mask_is_crop: bool,
    ) -> QtCore.QRectF:
        """Choose image-space placement for a mask PNG.

        New per-box outputs save cropped mask PNGs and mark `mask_is_crop=True`,
        so they render at the detection box. Legacy/merged outputs are full image
        masks and render at the image rect.
        """
        if mask_is_crop and len(bbox) >= 4 and (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
            return QtCore.QRectF(float(bbox[0]), float(bbox[1]), float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))

        image = QtGui.QImage(str(mask_path))
        if not image.isNull() and len(bbox) >= 4 and (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
            box_w = max(1, int(round(float(bbox[2]) - float(bbox[0]))))
            box_h = max(1, int(round(float(bbox[3]) - float(bbox[1]))))
            if abs(image.width() - box_w) <= 2 and abs(image.height() - box_h) <= 2:
                return QtCore.QRectF(float(bbox[0]), float(bbox[1]), float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
        return image_rect

    def _resolve_det_mask_path(self, det: dict, payload: dict, infer_id: str, index: int) -> str | None:
        """Resolve a usable mask file path for a detection dict.

        Order: per-detection mask_path -> decode mask_b64 to a temp PNG ->
        fall back to the top-level payload mask_path (single-mask workflows).
        """
        mask_path = str(det.get("mask_path") or "").strip()
        if mask_path and Path(mask_path).exists():
            return mask_path
        mask_b64 = det.get("mask_b64")
        if mask_b64:
            try:
                import base64

                data = base64.b64decode(mask_b64)
                out_dir = Path(self._seg_tmp_dir) if getattr(self, "_seg_tmp_dir", None) else Path(tempfile.gettempdir())
                out_path = out_dir / f"mask_{infer_id[:6]}_{index}.png"
                out_path.write_bytes(data)
                return str(out_path)
            except Exception:
                pass
        fallback = str(payload.get("mask_path") or "").strip()
        detections = list(payload.get("detections") or [])
        if fallback and Path(fallback).exists() and len(detections) <= 1:
            return fallback
        return None

    @QtCore.Slot(str, str)
    def _on_infer_failed(self, infer_id: str, error: str) -> None:
        job_id = self._seg_infer_to_job.pop(infer_id, None)
        if job_id is None:
            return
        self._jobs.fail(job_id, error)
        self._segment_panel.run_button.setEnabled(True)
        self._segment_panel.cancel_button.setEnabled(False)
        self._append_log(f"Segment ERROR: {error}")
        QtWidgets.QMessageBox.warning(self, "Segment failed", error)

    @QtCore.Slot(str)
    def _on_infer_cancelled(self, infer_id: str) -> None:
        job_id = self._seg_infer_to_job.pop(infer_id, None)
        if job_id is None:
            return
        self._jobs.cancel(job_id)
        self._segment_panel.run_button.setEnabled(True)
        self._segment_panel.cancel_button.setEnabled(False)
        self._append_log("Segment cancelled.")

    @QtCore.Slot(str, str)
    def _on_infer_log(self, infer_id: str, message: str) -> None:
        if infer_id in self._seg_infer_to_job:
            if self._seg_infer_to_job.get(infer_id):
                self._jobs.update(self._seg_infer_to_job[infer_id], JobUpdate(log_line=message))
            self._append_log(message)

    @QtCore.Slot(str, float)
    def _on_infer_progress(self, infer_id: str, progress: float) -> None:
        job_id = self._seg_infer_to_job.get(infer_id)
        if job_id:
            self._jobs.update(job_id, JobUpdate(progress=progress))
