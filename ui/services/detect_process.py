from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore

from ui.services.cli_job import CliJob


@dataclass(frozen=True)
class DetectionRow:
    roi_index: int
    detector_name: str
    group_name: str
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


class DetectProcess(QtCore.QObject):
    """ROI detection process managed by the UI.

    The actual work runs in a fresh Python subprocess. All temp files live at
    `.tmp/roi_detect/<job_id>/`; cancelling kills the process group.
    """

    log = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        *,
        image_path: Path,
        rois: list[QtCore.QRectF],
        detector_name: str,
        conf: float,
        tmp_dir: Path | None = None,
        stabledino_checkpoint: str = "",
    ) -> None:
        super().__init__()
        self._image_path = Path(image_path)
        self._rois = [QtCore.QRectF(r) for r in rois]
        self._detector_name = str(detector_name)
        self._conf = float(conf)
        self._stabledino_checkpoint = str(stabledino_checkpoint or "")
        self._job: CliJob | None = None
        self._out_json: Path | None = None
        self._rois_json: Path | None = None
        del tmp_dir  # legacy argument; CliJob owns .tmp/<workflow>/<job_id>/ now.

    @property
    def job_id(self) -> str | None:
        return self._job.job_id if self._job is not None else None

    @property
    def job_dir(self) -> Path | None:
        return self._job.job_dir if self._job is not None else None

    def start(self) -> None:
        job = CliJob(workflow="roi_detect", module="pineline.roi_detect_app.detect_job")
        job.job_dir.mkdir(parents=True, exist_ok=True)
        self._rois_json = job.job_dir / "rois_in.json"
        self._out_json = job.job_dir / "detections_out.json"

        rois_payload = [[float(r.left()), float(r.top()), float(r.right()), float(r.bottom())] for r in self._rois]
        self._rois_json.write_text(json.dumps(rois_payload), encoding="utf-8")
        if self._out_json.exists():
            self._out_json.unlink()

        args = [
            "--image",
            str(self._image_path),
            "--detector",
            self._detector_name,
            "--conf",
            f"{self._conf}",
            "--rois-json",
            str(self._rois_json),
            "--tmp-dir",
            str(job.job_dir),
            "--out-json",
            str(self._out_json),
        ]
        if self._stabledino_checkpoint:
            args += ["--stabledino-checkpoint", self._stabledino_checkpoint]
        job.args = args
        job.log.connect(self.log)
        job.progress.connect(self.progress)
        job.failed.connect(self.failed)
        job.finished.connect(self._on_job_finished)
        self._job = job
        self.log.emit(f"Launching detect process: {self._detector_name}, rois={len(self._rois)}")
        job.start()

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancel()

    def is_running(self) -> bool:
        return self._job is not None and self._job.is_running()

    @QtCore.Slot(int)
    def _on_job_finished(self, exit_code: int) -> None:
        if int(exit_code) != 0:
            self.failed.emit(f"Detect process exited with code {exit_code}")
            return
        try:
            if self._out_json is None:
                raise RuntimeError("missing output json path")
            payload = json.loads(self._out_json.read_text(encoding="utf-8"))
        except Exception as exc:
            self.failed.emit(f"Cannot read results: {exc}")
            return
        rows = [
            DetectionRow(
                roi_index=int(item["roi_index"]),
                detector_name=str(item["detector_name"]),
                group_name=str(item["group_name"]),
                label=str(item["label"]),
                score=float(item["score"]),
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                x2=float(item["x2"]),
                y2=float(item["y2"]),
            )
            for item in payload
        ]
        self.finished.emit(rows)
