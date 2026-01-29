from __future__ import annotations

import os
import sys
import threading
import json
import subprocess
import tempfile
from dataclasses import replace
from dataclasses import dataclass
from dataclasses import asdict
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
from predict_sam_dino import SamDinoParams, SamDinoRunner
from predict_unet import UnetParams, UnetRunner


class _WorkerBase(QtCore.QObject):
    log = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    @QtCore.Slot()
    def stop(self) -> None:
        self._stop_event.set()
        self.log.emit("Stop requested...")

    def _stop_checker(self) -> bool:
        return self._stop_event.is_set()


class ProcessingDialog(QtWidgets.QDialog):
    stopRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None, title: str) -> None:
        super().__init__(parent)
        self._allow_close = False
        self.setWindowTitle(title)
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self.resize(720, 420)

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(title, self)
        header.setWordWrap(True)
        layout.addWidget(header)

        self._log = QtWidgets.QPlainTextEdit(self)
        self._log.setReadOnly(True)
        layout.addWidget(self._log, 1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self._stop_btn = QtWidgets.QPushButton("Stop", self)
        self._stop_btn.clicked.connect(self.stopRequested.emit)
        row.addWidget(self._stop_btn)
        layout.addLayout(row)

    def log_widget(self) -> QtWidgets.QPlainTextEdit:
        return self._log

    def stop_button(self) -> QtWidgets.QPushButton:
        return self._stop_btn

    def allow_close(self) -> None:
        self._allow_close = True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape and not self._allow_close:
            event.ignore()
            return
        super().keyPressEvent(event)


class UnetWorker(_WorkerBase):
    def __init__(self, runner: UnetRunner, image_path: str, params: UnetParams) -> None:
        super().__init__()
        self._runner = runner
        self._image_path = image_path
        self._params = params

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.log.emit(f"UNet: image={self._image_path}")
            self.log.emit(f"UNet: model={self._params.model_path}")
            details = self._runner.run(
                self._image_path,
                self._params,
                stop_checker=self._stop_checker,
                log_fn=self.log.emit,
            )
            self.finished.emit(details)
        except Exception as e:
            if e.__class__.__name__ == "StopRequested" or str(e) == "Stopped":
                self.log.emit("STOPPED")
                self.finished.emit({"stopped": True})
                return
            self.failed.emit(str(e))


class SamDinoWorker(_WorkerBase):
    def __init__(self, runner: SamDinoRunner, image_path: str, params: SamDinoParams) -> None:
        super().__init__()
        self._runner = runner
        self._image_path = image_path
        self._params = params
        self._proc: subprocess.Popen[str] | None = None

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.log.emit(f"SAM+DINO: image={self._image_path}")
            self.log.emit(f"SAM+DINO: sam_checkpoint={self._params.sam_checkpoint}")
            self.log.emit(f"SAM+DINO: gdino_checkpoint={self._params.gdino_checkpoint}")
            self.log.emit(f"SAM+DINO: gdino_config_id={self._params.gdino_config_id}")

            repo_root = Path(__file__).resolve().parents[1]
            payload = {
                "mode": "run",
                "image_path": self._image_path,
                "params": asdict(self._params),
            }

            with tempfile.TemporaryDirectory(prefix="sam_dino_") as td:
                payload_path = os.path.join(td, "payload.json")
                out_path = os.path.join(td, "result.json")
                with open(payload_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    f.write("\n")

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                py_path = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{py_path}" if py_path else str(repo_root)

                cmd = [sys.executable, "-m", "sam_dino.cli", "--payload", payload_path, "--output", out_path]
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                last_lines: list[str] = []
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    if self._stop_checker():
                        break
                    msg = line.rstrip("\n")
                    last_lines.append(msg)
                    if len(last_lines) > 60:
                        last_lines.pop(0)
                    self.log.emit(msg)

                if self._stop_checker():
                    try:
                        if self._proc.poll() is None:
                            self._proc.terminate()
                            self._proc.wait(timeout=5)
                    except Exception:
                        try:
                            if self._proc.poll() is None:
                                self._proc.kill()
                        except Exception:
                            pass
                    self.finished.emit({"stopped": True})
                    return

                rc = self._proc.wait()
                self._proc = None
                if rc != 0 and not os.path.isfile(out_path):
                    tail = "\n".join(last_lines[-10:])
                    raise RuntimeError(f"SAM+DINO subprocess exited with code {rc}\n\nLast output:\n{tail}")
                if not os.path.isfile(out_path):
                    tail = "\n".join(last_lines[-10:])
                    raise RuntimeError(f"SAM+DINO subprocess did not produce result.json\n\nLast output:\n{tail}")

                with open(out_path, "r", encoding="utf-8") as f:
                    details = json.load(f)
                if isinstance(details, dict) and details.get("error"):
                    raise RuntimeError(f"{details.get('error_type')}: {details.get('error')}")
                self.finished.emit(details)
        except Exception as e:
            if e.__class__.__name__ == "StopRequested" or str(e) == "Stopped":
                self.log.emit("STOPPED")
                self.finished.emit({"stopped": True})
                return
            self.failed.emit(str(e))


class SamDinoIsolateWorker(_WorkerBase):
    def __init__(
        self,
        runner: SamDinoRunner,
        image_path: str,
        params: SamDinoParams,
        *,
        target_labels: list[str],
        outside_value: int,
        crop_to_bbox: bool,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._image_path = image_path
        self._params = params
        self._target_labels = target_labels
        self._outside_value = outside_value
        self._crop_to_bbox = crop_to_bbox
        self._proc: subprocess.Popen[str] | None = None

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.log.emit(f"SAM+DINO isolate: image={self._image_path}")
            self.log.emit(f"SAM+DINO isolate: sam_checkpoint={self._params.sam_checkpoint}")
            self.log.emit(f"SAM+DINO isolate: gdino_checkpoint={self._params.gdino_checkpoint}")
            self.log.emit(f"SAM+DINO isolate: gdino_config_id={self._params.gdino_config_id}")
            repo_root = Path(__file__).resolve().parents[1]
            payload = {
                "mode": "isolate",
                "image_path": self._image_path,
                "params": asdict(self._params),
                "target_labels": list(self._target_labels),
                "outside_value": int(self._outside_value),
                "crop_to_bbox": bool(self._crop_to_bbox),
            }

            with tempfile.TemporaryDirectory(prefix="sam_dino_") as td:
                payload_path = os.path.join(td, "payload.json")
                out_path = os.path.join(td, "result.json")
                with open(payload_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    f.write("\n")

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                py_path = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{py_path}" if py_path else str(repo_root)

                cmd = [sys.executable, "-m", "sam_dino.cli", "--payload", payload_path, "--output", out_path]
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                last_lines: list[str] = []
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    if self._stop_checker():
                        break
                    msg = line.rstrip("\n")
                    last_lines.append(msg)
                    if len(last_lines) > 60:
                        last_lines.pop(0)
                    self.log.emit(msg)

                if self._stop_checker():
                    try:
                        if self._proc.poll() is None:
                            self._proc.terminate()
                            self._proc.wait(timeout=5)
                    except Exception:
                        try:
                            if self._proc.poll() is None:
                                self._proc.kill()
                        except Exception:
                            pass
                    self.finished.emit({"stopped": True})
                    return

                rc = self._proc.wait()
                self._proc = None
                if rc != 0 and not os.path.isfile(out_path):
                    tail = "\n".join(last_lines[-10:])
                    raise RuntimeError(f"SAM+DINO subprocess exited with code {rc}\n\nLast output:\n{tail}")
                if not os.path.isfile(out_path):
                    tail = "\n".join(last_lines[-10:])
                    raise RuntimeError(f"SAM+DINO subprocess did not produce result.json\n\nLast output:\n{tail}")

                with open(out_path, "r", encoding="utf-8") as f:
                    details = json.load(f)
                if isinstance(details, dict) and details.get("error"):
                    raise RuntimeError(f"{details.get('error_type')}: {details.get('error')}")
                self.finished.emit(details)
        except Exception as e:
            if e.__class__.__name__ == "StopRequested" or str(e) == "Stopped":
                self.log.emit("STOPPED")
                self.finished.emit({"stopped": True})
                return
            self.failed.emit(str(e))


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
        self._worker: _WorkerBase | None = None
        self._active_stop_btn: QtWidgets.QPushButton | None = None
        self._active_log_widget: QtWidgets.QPlainTextEdit | None = None
        self._progress_dialog: ProcessingDialog | None = None
        self._post_run_action: dict | None = None

        self._state: LoadedState | None = None
        self._mask_path: Path | None = None
        self._pending_unet: tuple[str, UnetParams] | None = None

        self._overlay_canvas = ImageCanvas(self, render_mode="overlay", editable=True)
        self._image_canvas = ImageCanvas(self, render_mode="image", editable=False)
        self._mask_canvas = ImageCanvas(self, render_mode="mask", editable=False)

        self._overlay_canvas.cursorInfo.connect(self._on_cursor_info)
        self._overlay_canvas.brushRadiusChanged.connect(self._sync_brush_slider)
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

        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_view_tabs(), 1)
        self.setCentralWidget(central)

    def _build_sidebar(self) -> QtWidgets.QWidget:
        sidebar = QtWidgets.QWidget(self)
        sidebar.setFixedWidth(420)
        layout = QtWidgets.QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)

        self._method_tabs = QtWidgets.QTabWidget(sidebar)
        self._method_tabs.addTab(self._build_tab_sam_dino(), "SAM + DINO")
        self._method_tabs.addTab(self._build_tab_unet(), "UNet")
        self._method_tabs.addTab(self._build_tab_editor(), "Editor")
        layout.addWidget(self._method_tabs, 1)
        return sidebar

    def _build_tab_editor(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        row = QtWidgets.QHBoxLayout()
        open_mask_btn = QtWidgets.QToolButton(tab)
        open_mask_btn.setDefaultAction(self._act_open_mask)
        save_mask_btn = QtWidgets.QToolButton(tab)
        save_mask_btn.setDefaultAction(self._act_save_mask)
        row.addWidget(open_mask_btn)
        row.addWidget(save_mask_btn)
        row.addStretch(1)
        layout.addLayout(row)


        layout.addWidget(QtWidgets.QLabel("Brush size:", tab))
        self._brush_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, tab)
        self._brush_slider.setRange(1, 250)
        self._brush_slider.setValue(self._overlay_canvas.canvas_state().brush_radius)
        self._brush_slider.valueChanged.connect(self._overlay_canvas.set_brush_radius)
        layout.addWidget(self._brush_slider)
        self._brush_value = QtWidgets.QLabel(f"{self._brush_slider.value()} px", tab)
        self._brush_slider.valueChanged.connect(lambda v: self._brush_value.setText(f"{v} px"))
        layout.addWidget(self._brush_value)

        layout.addSpacing(6)
        layout.addWidget(QtWidgets.QLabel("Overlay opacity:", tab))
        self._overlay_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, tab)
        self._overlay_slider.setRange(0, 255)
        self._overlay_slider.setValue(self._overlay_canvas.canvas_state().overlay_opacity)
        self._overlay_slider.valueChanged.connect(self._overlay_canvas.set_overlay_opacity)
        layout.addWidget(self._overlay_slider)

        layout.addSpacing(6)
        layout.addWidget(
            QtWidgets.QLabel(
                "Paint: LMB\n"
                "Erase: Ctrl + LMB\n"
                "Zoom: Ctrl + Wheel\n"
                "Brush: Ctrl + Shift + Wheel\n"
                "Pan: Wheel (Shift+Wheel = horizontal)",
                tab,
            )
        )
        layout.addStretch(1)
        return tab

    def _build_view_tabs(self) -> QtWidgets.QWidget:
        self._view_tabs = QtWidgets.QTabWidget(self)
        self._view_tabs.addTab(self._overlay_canvas, "Overlay")
        self._view_tabs.addTab(self._image_canvas, "Image")
        self._view_tabs.addTab(self._mask_canvas, "Mask")
        return self._view_tabs

    def _build_tab_unet(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        self._unet_model_edit = QtWidgets.QLineEdit(tab)
        self._unet_model_edit.setText(str(Path("PredictTools/Unet/best_model.pth")))
        model_browse = QtWidgets.QPushButton("Browse...", tab)
        model_browse.clicked.connect(lambda: self._browse_file(self._unet_model_edit, "Model (*.pth);;All files (*)"))

        self._unet_outdir_edit = QtWidgets.QLineEdit(tab)
        self._unet_outdir_edit.setText("results_unet")
        out_browse = QtWidgets.QPushButton("Browse...", tab)
        out_browse.clicked.connect(lambda: self._browse_dir(self._unet_outdir_edit))

        self._unet_threshold = QtWidgets.QDoubleSpinBox(tab)
        self._unet_threshold.setRange(0.0, 1.0)
        self._unet_threshold.setSingleStep(0.01)
        self._unet_threshold.setValue(0.5)

        self._unet_post = QtWidgets.QCheckBox("Apply postprocessing", tab)
        self._unet_post.setChecked(True)

        self._unet_mode = QtWidgets.QComboBox(tab)
        self._unet_mode.addItems(["tile", "letterbox", "resize"])

        self._unet_input_size = QtWidgets.QSpinBox(tab)
        self._unet_input_size.setRange(64, 4096)
        self._unet_input_size.setValue(256)

        self._unet_overlap = QtWidgets.QSpinBox(tab)
        self._unet_overlap.setRange(0, 4096)
        self._unet_overlap.setValue(0)
        self._unet_overlap.setToolTip("0 = recommended (input_size//2)")

        self._unet_tile_batch = QtWidgets.QSpinBox(tab)
        self._unet_tile_batch.setRange(1, 128)
        self._unet_tile_batch.setValue(4)

        r = 0
        grid.addWidget(QtWidgets.QLabel("Model (.pth)"), r, 0)
        grid.addWidget(self._unet_model_edit, r, 1)
        grid.addWidget(model_browse, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Output dir"), r, 0)
        grid.addWidget(self._unet_outdir_edit, r, 1)
        grid.addWidget(out_browse, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Threshold"), r, 0)
        grid.addWidget(self._unet_threshold, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Mode"), r, 0)
        grid.addWidget(self._unet_mode, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Input size"), r, 0)
        grid.addWidget(self._unet_input_size, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Tile overlap"), r, 0)
        grid.addWidget(self._unet_overlap, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Tile batch size"), r, 0)
        grid.addWidget(self._unet_tile_batch, r, 1)
        r += 1
        grid.addWidget(self._unet_post, r, 1)
        grid.setColumnStretch(1, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self._unet_run_btn = QtWidgets.QPushButton("Run UNet", tab)
        self._unet_run_btn.clicked.connect(self._run_unet)
        btn_row.addWidget(self._unet_run_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)
        return tab

    def _build_tab_sam_dino(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        self._sd_sam_ckpt = QtWidgets.QLineEdit(tab)
        self._sd_sam_ckpt.setText(str(Path("PredictTools/SAM/sam_vit_h_4b8939.pth")))
        ckpt_browse = QtWidgets.QPushButton("Browse...", tab)
        ckpt_browse.clicked.connect(lambda: self._browse_file(self._sd_sam_ckpt, "SAM checkpoint (*.pth);;All files (*)"))

        self._sd_sam_type = QtWidgets.QComboBox(tab)
        self._sd_sam_type.addItems(["auto", "vit_b", "vit_l", "vit_h"])

        self._sd_use_delta = QtWidgets.QCheckBox("Use fine-tuned (delta) model", tab)
        self._sd_use_delta.setChecked(False)

        self._sd_delta_group = QtWidgets.QGroupBox("Fine-tune (delta)", tab)
        self._sd_delta_group.setVisible(False)
        dg = QtWidgets.QGridLayout(self._sd_delta_group)

        self._sd_delta_type = QtWidgets.QComboBox(self._sd_delta_group)
        self._sd_delta_type.addItems(["adapter", "lora", "both"])
        self._sd_delta_type.setCurrentText("lora")

        self._sd_delta_ckpt = QtWidgets.QLineEdit(self._sd_delta_group)
        self._sd_delta_ckpt.setPlaceholderText("auto or path/to/delta.pth")
        self._sd_delta_ckpt.setText("auto")
        delta_browse = QtWidgets.QPushButton("Browse...", self._sd_delta_group)
        delta_browse.clicked.connect(
            lambda: self._browse_file(self._sd_delta_ckpt, "Delta checkpoint (*.pth);;All files (*)")
        )

        self._sd_middle_dim = QtWidgets.QSpinBox(self._sd_delta_group)
        self._sd_middle_dim.setRange(1, 4096)
        self._sd_middle_dim.setValue(32)

        self._sd_scaling_factor = QtWidgets.QDoubleSpinBox(self._sd_delta_group)
        self._sd_scaling_factor.setRange(0.0, 10.0)
        self._sd_scaling_factor.setDecimals(4)
        self._sd_scaling_factor.setSingleStep(0.05)
        self._sd_scaling_factor.setValue(0.2)

        self._sd_rank = QtWidgets.QSpinBox(self._sd_delta_group)
        self._sd_rank.setRange(1, 1024)
        self._sd_rank.setValue(4)

        dr = 0
        dg.addWidget(QtWidgets.QLabel("Delta type"), dr, 0)
        dg.addWidget(self._sd_delta_type, dr, 1, 1, 2)
        dr += 1
        dg.addWidget(QtWidgets.QLabel("Delta checkpoint"), dr, 0)
        dg.addWidget(self._sd_delta_ckpt, dr, 1)
        dg.addWidget(delta_browse, dr, 2)
        dr += 1
        dg.addWidget(QtWidgets.QLabel("Adapter middle_dim"), dr, 0)
        dg.addWidget(self._sd_middle_dim, dr, 1)
        dr += 1
        dg.addWidget(QtWidgets.QLabel("Adapter scaling_factor"), dr, 0)
        dg.addWidget(self._sd_scaling_factor, dr, 1)
        dr += 1
        dg.addWidget(QtWidgets.QLabel("LoRA rank"), dr, 0)
        dg.addWidget(self._sd_rank, dr, 1)
        dg.setColumnStretch(1, 1)

        self._sd_gdino_ckpt = QtWidgets.QLineEdit(tab)
        self._sd_gdino_ckpt.setPlaceholderText("path/to/groundingdino.pth or HF model id")
        self._sd_gdino_ckpt.setText("IDEA-Research/grounding-dino-base")
        gdino_browse = QtWidgets.QPushButton("Browse...", tab)
        gdino_browse.clicked.connect(
            lambda: self._browse_file(self._sd_gdino_ckpt, "GroundingDINO checkpoint (*.pth);;All files (*)")
        )

        self._sd_gdino_cfg = QtWidgets.QComboBox(tab)
        self._sd_gdino_cfg.addItem("Auto (infer from filename)", "auto")
        self._sd_gdino_cfg.addItem("grounding-dino-base", "IDEA-Research/grounding-dino-base")
        self._sd_gdino_cfg.addItem("grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny")

        self._sd_queries = QtWidgets.QLineEdit(tab)
        self._sd_queries.setText("crack,mold,stain,spall,damage,column")

        self._sd_isolate_labels = QtWidgets.QLineEdit(tab)
        self._sd_isolate_labels.setPlaceholderText("Target labels (comma, optional)")

        self._sd_isolate_crop = QtWidgets.QCheckBox("Crop to bbox", tab)
        self._sd_isolate_outside_white = QtWidgets.QCheckBox("Outside white (255)", tab)

        self._sd_box_thr = QtWidgets.QDoubleSpinBox(tab)
        self._sd_box_thr.setRange(0.0, 1.0)
        self._sd_box_thr.setSingleStep(0.01)
        self._sd_box_thr.setValue(0.25)

        self._sd_text_thr = QtWidgets.QDoubleSpinBox(tab)
        self._sd_text_thr.setRange(0.0, 1.0)
        self._sd_text_thr.setSingleStep(0.01)
        self._sd_text_thr.setValue(0.25)

        self._sd_max_dets = QtWidgets.QSpinBox(tab)
        self._sd_max_dets.setRange(0, 999)
        self._sd_max_dets.setValue(20)

        self._sd_invert = QtWidgets.QCheckBox("Invert mask", tab)
        self._sd_invert.setChecked(False)

        self._sd_min_area = QtWidgets.QSpinBox(tab)
        self._sd_min_area.setRange(0, 10_000_000)
        self._sd_min_area.setValue(0)

        self._sd_dilate = QtWidgets.QSpinBox(tab)
        self._sd_dilate.setRange(0, 50)
        self._sd_dilate.setValue(0)

        self._sd_outdir = QtWidgets.QLineEdit(tab)
        self._sd_outdir.setText("results_sam_dino")
        out_browse = QtWidgets.QPushButton("Browse...", tab)
        out_browse.clicked.connect(lambda: self._browse_dir(self._sd_outdir))

        r = 0
        grid.addWidget(QtWidgets.QLabel("SAM checkpoint"), r, 0)
        grid.addWidget(self._sd_sam_ckpt, r, 1)
        grid.addWidget(ckpt_browse, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("SAM type"), r, 0)
        grid.addWidget(self._sd_sam_type, r, 1)
        r += 1
        grid.addWidget(self._sd_use_delta, r, 1, 1, 2)
        r += 1
        grid.addWidget(self._sd_delta_group, r, 0, 1, 3)
        r += 1
        grid.addWidget(QtWidgets.QLabel("GroundingDINO checkpoint / model id"), r, 0)
        grid.addWidget(self._sd_gdino_ckpt, r, 1)
        grid.addWidget(gdino_browse, r, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("GroundingDINO config"), r, 0)
        grid.addWidget(self._sd_gdino_cfg, r, 1, 1, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Text queries (comma)"), r, 0)
        grid.addWidget(self._sd_queries, r, 1, 1, 2)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Isolate labels (comma)"), r, 0)
        grid.addWidget(self._sd_isolate_labels, r, 1, 1, 2)
        r += 1
        grid.addWidget(self._sd_isolate_crop, r, 1)
        r += 1
        grid.addWidget(self._sd_isolate_outside_white, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Box thr"), r, 0)
        grid.addWidget(self._sd_box_thr, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Text thr"), r, 0)
        grid.addWidget(self._sd_text_thr, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Max dets"), r, 0)
        grid.addWidget(self._sd_max_dets, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Min component area"), r, 0)
        grid.addWidget(self._sd_min_area, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Dilate iters"), r, 0)
        grid.addWidget(self._sd_dilate, r, 1)
        r += 1
        grid.addWidget(self._sd_invert, r, 1)
        r += 1
        grid.addWidget(QtWidgets.QLabel("Output dir"), r, 0)
        grid.addWidget(self._sd_outdir, r, 1)
        grid.addWidget(out_browse, r, 2)
        grid.setColumnStretch(1, 1)

        self._sd_use_delta.toggled.connect(self._sd_delta_group.setVisible)
        self._sd_delta_type.currentTextChanged.connect(self._sync_delta_controls)
        self._sd_delta_ckpt.textChanged.connect(self._auto_enable_delta_if_path)
        self._sync_delta_controls(self._sd_delta_type.currentText())

        btn_row = QtWidgets.QHBoxLayout()
        self._sd_run_btn = QtWidgets.QPushButton("Run SAM + DINO", tab)
        self._sd_run_btn.clicked.connect(self._run_sam_dino)
        self._sd_isolate_btn = QtWidgets.QPushButton("Tách vật thể", tab)
        self._sd_isolate_btn.clicked.connect(self._isolate_object)
        btn_row.addWidget(self._sd_run_btn)
        btn_row.addWidget(self._sd_isolate_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)
        return tab

    def _sync_delta_controls(self, delta_type: str) -> None:
        dt = str(delta_type).strip().lower()
        use_adapter = dt in {"adapter", "both"}
        use_lora = dt in {"lora", "both"}
        self._sd_middle_dim.setEnabled(use_adapter)
        self._sd_scaling_factor.setEnabled(use_adapter)
        self._sd_rank.setEnabled(use_lora)

    def _auto_enable_delta_if_path(self, text: str) -> None:
        t = str(text or "").strip()
        if not t:
            return
        if t.lower() == "auto":
            return
        low = t.lower().replace("\\", "/")
        base = low.rsplit("/", 1)[-1]
        has_adapter = "adapter" in base
        has_lora = "lora" in base
        if has_adapter and has_lora:
            self._sd_delta_type.setCurrentText("both")
        elif has_adapter:
            self._sd_delta_type.setCurrentText("adapter")
        elif has_lora:
            self._sd_delta_type.setCurrentText("lora")
        if not self._sd_use_delta.isChecked():
            self._sd_use_delta.setChecked(True)

    def _browse_file(self, line_edit: QtWidgets.QLineEdit, filter_str: str) -> None:
        start = line_edit.text().strip() or str(Path.cwd())
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", start, filter_str)
        if f:
            line_edit.setText(f)

    def _browse_dir(self, line_edit: QtWidgets.QLineEdit) -> None:
        start = line_edit.text().strip() or str(Path.cwd())
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        if d:
            line_edit.setText(d)

    def _build_sam_dino_params(
        self,
        *,
        queries_override: list[str] | None = None,
        invert_override: bool | None = None,
    ) -> SamDinoParams:
        ckpt = self._sd_sam_ckpt.text().strip()
        gdino_ckpt = self._sd_gdino_ckpt.text().strip()
        if not gdino_ckpt:
            raise ValueError("GroundingDINO checkpoint is required.")
        if not os.path.exists(gdino_ckpt):
            lower = gdino_ckpt.lower()
            if lower.endswith((".pth", ".pt", ".safetensors", ".bin")):
                raise ValueError(f"GroundingDINO checkpoint not found: {gdino_ckpt}")

        out_dir = self._sd_outdir.text().strip() or "results_sam_dino"
        if queries_override is None:
            queries = [q.strip() for q in self._sd_queries.text().split(",") if q.strip()]
            if not queries:
                queries = ["crack"]
        else:
            queries = queries_override

        delta_type = "none"
        delta_ckpt = "auto"
        middle_dim = 32
        scaling_factor = 0.2
        rank = 4
        if self._sd_use_delta.isChecked():
            delta_type = str(self._sd_delta_type.currentText())
            delta_ckpt = self._sd_delta_ckpt.text().strip() or "auto"
            middle_dim = int(self._sd_middle_dim.value())
            scaling_factor = float(self._sd_scaling_factor.value())
            rank = int(self._sd_rank.value())

        invert_mask = bool(self._sd_invert.isChecked())
        if invert_override is not None:
            invert_mask = bool(invert_override)

        cfg_id = self._sd_gdino_cfg.currentData()
        if cfg_id is None:
            cfg_id = "auto"

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
            output_dir=out_dir,
        )

    def _append_log(self, widget: QtWidgets.QPlainTextEdit, text: str) -> None:
        widget.appendPlainText(text)

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

        overlay_path = details.get("overlay_path") or ""
        if overlay_path and os.path.isfile(overlay_path):
            try:
                overlay_img = load_image(overlay_path)
                self._image_canvas.set_image(overlay_img)
                if self._active_log_widget is not None:
                    self._append_log(self._active_log_widget, f"Preview overlay: {overlay_path}")
            except Exception as e:
                if self._active_log_widget is not None:
                    self._append_log(self._active_log_widget, f"Could not load overlay preview: {e}")

        mask_path = details.get("mask_path") or details.get("mask") or ""
        if mask_path and self._state is not None and os.path.isfile(mask_path):
            try:
                loaded = load_mask(mask_path, (self._state.image_w, self._state.image_h))
                self._overlay_canvas.set_mask(loaded.mask)
                self._sync_mask_views()
                self._mask_path = Path(mask_path)
                self._view_tabs.setCurrentIndex(0)
                if self._active_log_widget is not None:
                    self._append_log(self._active_log_widget, f"Loaded mask: {mask_path}")
            except ImageIoError as e:
                if self._active_log_widget is not None:
                    self._append_log(self._active_log_widget, f"Could not load output mask: {e}")

        if self._active_log_widget is not None:
            self._append_log(self._active_log_widget, "DONE")

        post_action = self._post_run_action
        self._post_run_action = None
        if post_action and post_action.get("type") in {"extract", "isolate"}:
            self._handle_extract_result(details, post_action.get("source_image"))

    def _set_running(self, running: bool) -> None:
        self._unet_run_btn.setEnabled(not running)
        self._sd_run_btn.setEnabled(not running)
        if hasattr(self, "_sd_isolate_btn"):
            self._sd_isolate_btn.setEnabled(not running)
        if self._active_stop_btn is not None:
            self._active_stop_btn.setEnabled(running)
        self._act_open_image.setEnabled(not running)
        self._act_open_mask.setEnabled(not running)
        self._act_save_mask.setEnabled(not running)

    def _set_roi_selecting(self, selecting: bool) -> None:
        if selecting:
            self._unet_run_btn.setEnabled(False)
            self._sd_run_btn.setEnabled(False)
            if hasattr(self, "_sd_isolate_btn"):
                self._sd_isolate_btn.setEnabled(False)
            self._act_open_image.setEnabled(False)
            self._act_open_mask.setEnabled(False)
            self._act_save_mask.setEnabled(False)
            self.statusBar().showMessage("Select ROI: drag on Overlay tab. Esc = cancel. Click = full image.")
        else:
            self.statusBar().clearMessage()
            if self._thread is None:
                self._unet_run_btn.setEnabled(True)
                self._sd_run_btn.setEnabled(True)
                if hasattr(self, "_sd_isolate_btn"):
                    self._sd_isolate_btn.setEnabled(True)
                self._act_open_image.setEnabled(True)
                self._act_open_mask.setEnabled(True)
                self._act_save_mask.setEnabled(True)

    def _stop_current(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()

    def open_image_dialog(self) -> None:
        start = str(Path(self._state.image_path).parent) if self._state is not None else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            start,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str) -> None:
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
        self._view_tabs.setCurrentIndex(0)

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

    def save_mask_dialog(self) -> None:
        if self._state is None or self._overlay_canvas.mask().isNull():
            QtWidgets.QMessageBox.critical(self, "Save Mask", "No mask to save.")
            return
        default_dir = self._mask_path or Path(self._state.image_path).parent
        path, selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Mask As",
            str(default_dir),
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

    def _sync_brush_slider(self, radius: int) -> None:
        if self._brush_slider.value() == radius:
            return
        self._brush_slider.blockSignals(True)
        self._brush_slider.setValue(radius)
        self._brush_slider.blockSignals(False)
        self._brush_value.setText(f"{radius} px")

    def _sync_mask_views(self) -> None:
        self._mask_canvas.set_mask(self._overlay_canvas.mask())
        self._mask_canvas.update()

    def _on_cursor_info(self, x: int, y: int, v: int) -> None:
        self._status_label.setText(f"x={x}  y={y}  mask={v}")

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

    def _start_worker(self, worker: _WorkerBase, *, title: str, pre_logs: list[str] | None = None) -> None:
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

        worker.moveToThread(self._thread)

        self._thread.started.connect(worker.run)
        worker.log.connect(self._on_worker_log)
        worker.failed.connect(self._on_worker_failed_slot)
        worker.finished.connect(self._on_worker_finished_slot)

        worker.finished.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
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
        self._set_running(False)

    def _run_unet(self) -> None:
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "UNet", "Load an image first.")
            return

        image_path = self._state.image_path
        model_path = self._unet_model_edit.text().strip()
        out_dir = self._unet_outdir_edit.text().strip() or "results_unet"

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

        self._pending_unet = (image_path, params)

        self._view_tabs.setCurrentIndex(0)
        self._overlay_canvas.set_editable(False)
        self._overlay_canvas.start_roi_selection()
        self._set_roi_selecting(True)

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
        worker = UnetWorker(self._unet_runner, image_path, params2)
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
            params = self._build_sam_dino_params()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Isolate object", str(e))
            return

        labels_text = self._sd_isolate_labels.text().strip()
        target_labels = [t.strip() for t in labels_text.split(",") if t.strip()]
        outside_value = 255 if self._sd_isolate_outside_white.isChecked() else 0
        crop_to_bbox = bool(self._sd_isolate_crop.isChecked())

        self._post_run_action = {"type": "isolate", "source_image": self._state.image_path}
        worker = SamDinoIsolateWorker(
            self._samdino_runner,
            self._state.image_path,
            params,
            target_labels=target_labels,
            outside_value=outside_value,
            crop_to_bbox=crop_to_bbox,
        )
        self._start_worker(worker, title="Isolate object")

    def _run_sam_dino(self) -> None:
        if not self._ensure_image_loaded():
            QtWidgets.QMessageBox.critical(self, "SAM+DINO", "Load an image first.")
            return

        try:
            params = self._build_sam_dino_params()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM+DINO", str(e))
            return

        self._post_run_action = None
        worker = SamDinoWorker(self._samdino_runner, self._state.image_path, params)
        self._start_worker(worker, title="SAM + DINO processing")


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    raise SystemExit(app.exec())
