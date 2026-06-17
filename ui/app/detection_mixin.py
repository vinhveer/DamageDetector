from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ui.models.job import JobKind, JobUpdate
from ui.services.detect_process import DetectProcess, DetectionRow
from ui.services.box_segment_pipeline import run_yolo_detection


class DetectionMixin:
    def _wire_detection_signals(self) -> None:
        self._detect_panel.runRequested.connect(self._run_detection)
        self._detect_panel.cancelRequested.connect(self._cancel_detection)
        self._detect_panel.filtersChanged.connect(self._refresh_results)
        self._detect_panel.table.itemSelectionChanged.connect(self._on_table_selected)
        self._jobs.jobAdded.connect(self._jobs_panel.upsert)
        self._jobs.jobStatusChanged.connect(self._jobs_panel.upsert)
        self._jobs.logEmitted.connect(lambda _jid, line: self._append_log(line))
        self._jobs_panel.cancelRequested.connect(self._cancel_job)

    def _set_stabledino_checkpoint(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "StableDINO checkpoint (.pth)",
            str(self._stabledino_checkpoint or Path.cwd()),
            "Checkpoint (*.pth)",
        )
        if path:
            self._stabledino_checkpoint = path
            self._settings.stabledino_checkpoint = path
            self.statusBar().showMessage(f"StableDINO checkpoint: {path}")

    def _run_detection(self) -> None:
        if self._image_path is None:
            QtWidgets.QMessageBox.information(self, "No image", "Open an image first.")
            return
        rois = self._canvas.roi_rects()
        if not rois:
            full = self._canvas.image_rect()
            if full.isEmpty():
                QtWidgets.QMessageBox.information(self, "No image", "Open an image first.")
                return
            rois = [full]

        detector_key = str(self._detect_panel.detector_combo.currentData() or "gdino")
        detector = self._detect_panel.detector_combo.currentText()
        sd_ckpt = ""
        if detector_key == "stabledino":
            sd_ckpt = self._stabledino_checkpoint
            self._settings.stabledino_checkpoint = sd_ckpt
            if not sd_ckpt:
                QtWidgets.QMessageBox.information(self, "StableDINO checkpoint", "Set StableDINO checkpoint first (Detector menu).")
                return

        # Accumulate: each detect run becomes its own layer/group. Don't clear
        # previous detections.
        self._last_detector = detector
        self._log_panel.clear()
        self._set_busy(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(rois))
        self._progress.setValue(0)

        job = self._jobs.submit(
            JobKind.detect,
            label=f"detect {detector} ({len(rois)} ROI)",
            params={"detector": detector, "rois": len(rois)},
        )
        self._proc_job_id = job.id
        self._jobs.mark_running(job.id)

        if detector_key == "yolo":
            try:
                opts = self._detect_panel.yolo_options
                self._settings.yolo_checkpoint = opts.checkpoint.text().strip()
                self._settings.yolo_conf = float(opts.conf.value())
                self._settings.yolo_iou = float(opts.iou.value())
                self._settings.yolo_imgsz = int(opts.imgsz.value())
                self._settings.yolo_max_dets = int(opts.max_dets.value())
                boxes = run_yolo_detection(
                    self._image_path,
                    Path.cwd() / ".tmp" / "ui_yolo_detect" / job.id,
                    weight_path=opts.checkpoint.text().strip() or None,
                    device=str(self._settings.device),
                    conf=float(opts.conf.value()),
                    iou=float(opts.iou.value()),
                    imgsz=int(opts.imgsz.value()),
                    max_dets=int(opts.max_dets.value()),
                    fallback=False,
                    log_fn=self._on_detect_log,
                )
                rows = []
                for idx, item in enumerate(boxes):
                    box = item.get("box") or [0, 0, 0, 0]
                    if len(box) != 4:
                        continue
                    rows.append(
                        DetectionRow(
                            roi_index=idx,
                            detector_name="YOLO",
                            group_name="YOLO",
                            label=str(item.get("label") or "crack"),
                            score=float(item.get("score") or 0.0),
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                        )
                    )
                self._on_detect_finished(rows)
            except Exception as exc:
                self._on_detect_failed(str(exc))
            return

        self._proc = DetectProcess(
            image_path=self._image_path,
            rois=rois,
            detector_name=detector,
            conf=float(self._detect_panel.run_conf.value()),
            stabledino_checkpoint=sd_ckpt,
        )
        self._active_processes.append(self._proc)
        self._proc.log.connect(self._on_detect_log)
        self._proc.progress.connect(self._on_detect_progress)
        self._proc.finished.connect(self._on_detect_finished)
        self._proc.failed.connect(self._on_detect_failed)
        self._proc.start()

    def _cancel_detection(self) -> None:
        if self._proc is not None:
            self._proc.cancel()
            self._append_log("Cancel requested...")

    def _cancel_job(self, job_id: str) -> None:
        if job_id == self._proc_job_id:
            self._cancel_detection()

    @QtCore.Slot(str)
    def _on_detect_log(self, message: str) -> None:
        if self._proc_job_id is not None:
            self._jobs.update(self._proc_job_id, JobUpdate(log_line=str(message)))
        else:
            self._append_log(str(message))

    @QtCore.Slot(int, int)
    def _on_detect_progress(self, done: int, total: int) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(done)
        if self._proc_job_id is not None:
            self._jobs.progress(self._proc_job_id, done, total)

    @QtCore.Slot(list)
    def _on_detect_finished(self, rows: list[DetectionRow]) -> None:
        detector = getattr(self, "_last_detector", "detect")
        if rows:
            self._add_detection_group(detector, list(rows))
        self._refresh_layers_panel()
        self._refresh_results()
        self._set_busy(False)
        self._progress.setVisible(False)
        if self._proc_job_id is not None:
            self._jobs.complete(self._proc_job_id, result=len(rows))
        if self._proc in self._active_processes:
            self._active_processes.remove(self._proc)
        self._proc = None
        self._proc_job_id = None
        self._append_log(f"Detect done: {len(self._canvas.roi_rects())} ROI, {len(rows)} boxes.")

    @QtCore.Slot(str)
    def _on_detect_failed(self, message: str) -> None:
        self._set_busy(False)
        self._progress.setVisible(False)
        if self._proc_job_id is not None:
            if str(message).startswith("Cancelled"):
                self._jobs.cancel(self._proc_job_id)
            else:
                self._jobs.fail(self._proc_job_id, message)
        if self._proc in self._active_processes:
            self._active_processes.remove(self._proc)
        self._proc = None
        self._proc_job_id = None
        self._append_log(f"ERROR: {message}")
        if not str(message).startswith("Cancelled"):
            QtWidgets.QMessageBox.warning(self, "Detect failed", str(message))
