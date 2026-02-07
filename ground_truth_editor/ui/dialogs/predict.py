from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..utils import browse_dir, browse_file, set_combo_by_data


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


class PredictRunDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, *, has_image: bool, has_folder: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Prediction")
        self.setModal(True)
        self.resize(400, 320)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Model Group
        model_group = QtWidgets.QGroupBox("Model", self)
        mg_layout = QtWidgets.QVBoxLayout(model_group)
        mg_layout.setSpacing(8)

        self.rb_sam_dino = QtWidgets.QRadioButton("SAM + DINO", model_group)
        self.rb_sam_dino_ft = QtWidgets.QRadioButton("SAM + DINO + Finetune (LoRA)", model_group)
        self.rb_unet = QtWidgets.QRadioButton("UNet + DINO", model_group)
        self.rb_sam_dino.setChecked(True)

        mg_layout.addWidget(self.rb_sam_dino)
        mg_layout.addWidget(self.rb_sam_dino_ft)
        mg_layout.addWidget(self.rb_unet)
        layout.addWidget(model_group)

        # Scope Group
        scope_group = QtWidgets.QGroupBox("Scope", self)
        sg_layout = QtWidgets.QVBoxLayout(scope_group)
        sg_layout.setSpacing(8)

        self.rb_scope_current = QtWidgets.QRadioButton("Current Image", scope_group)
        self.rb_scope_folder = QtWidgets.QRadioButton("Whole Folder", scope_group)

        self.rb_scope_current.setEnabled(has_image)
        self.rb_scope_folder.setEnabled(has_folder)

        if has_image:
            self.rb_scope_current.setChecked(True)
        elif has_folder:
            self.rb_scope_folder.setChecked(True)
        else:
            self.rb_scope_current.setChecked(True)

        sg_layout.addWidget(self.rb_scope_current)
        sg_layout.addWidget(self.rb_scope_folder)
        layout.addWidget(scope_group)

        layout.addStretch(1)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel, self
        )
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Run")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_result(self) -> tuple[str, str]:
        if self.rb_sam_dino_ft.isChecked():
            mode = "sam_dino_ft"
        elif self.rb_unet.isChecked():
            mode = "unet"
        else:
            mode = "sam_dino"

        scope = "folder" if self.rb_scope_folder.isChecked() else "current"
        return mode, scope


class PredictDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        title: str,
        mode: str,
        settings: dict,
        has_image: bool,
        has_folder: bool,
        show_scope: bool = True,
        pages: list[str] | None = None,
        ok_text: str = "Run",
        initial_page: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(860, 720)

        self._mode = str(mode or "").strip().lower()
        self._show_scope = bool(show_scope)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QtWidgets.QLabel(title, self)
        header.setStyleSheet("font-weight: 600; font-size: 14px;")
        root.addWidget(header)

        self._scope_current: QtWidgets.QRadioButton | None = None
        self._scope_folder: QtWidgets.QRadioButton | None = None
        if self._show_scope:
            scope_group = QtWidgets.QGroupBox("Scope", self)
            scope_layout = QtWidgets.QHBoxLayout(scope_group)
            scope_layout.setContentsMargins(10, 10, 10, 10)
            self._scope_current = QtWidgets.QRadioButton("Current image", scope_group)
            self._scope_folder = QtWidgets.QRadioButton("Whole folder", scope_group)
            self._scope_current.setEnabled(bool(has_image))
            self._scope_folder.setEnabled(bool(has_folder))
            scope_layout.addWidget(self._scope_current)
            scope_layout.addWidget(self._scope_folder)
            scope_layout.addStretch(1)
            root.addWidget(scope_group)

            if has_image:
                self._scope_current.setChecked(True)
            elif has_folder:
                self._scope_folder.setChecked(True)
            else:
                self._scope_current.setChecked(True)

        self._toolbox = QtWidgets.QTabWidget(self)
        self._toolbox.setDocumentMode(True)
        self._toolbox.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        root.addWidget(self._toolbox, 1)

        self._w: dict[str, QtWidgets.QWidget] = {}

        if pages is None:
            show_sam = self._mode in {"sam_dino", "sam_dino_ft"}
            show_unet = self._mode == "unet"
            show_dino = True
            pages = []
            if show_sam:
                pages.append("SAM")
            if show_dino:
                pages.append("DINO")
            if show_unet:
                pages.append("UNet")

        page_builders = {
            "sam": lambda: self._build_sam_page(settings),
            "dino": lambda: self._build_dino_page(settings),
            "unet": lambda: self._build_unet_page(settings),
        }
        for label in pages:
            key = str(label or "").strip().lower()
            builder = page_builders.get(key)
            if builder is None:
                continue
            self._toolbox.addTab(builder(), str(label))

        if self._toolbox.count() > 0:
            self._toolbox.setCurrentIndex(0)
            if initial_page:
                want = str(initial_page).strip().lower()
                for i in range(self._toolbox.count()):
                    if self._toolbox.tabText(i).strip().lower() == want:
                        self._toolbox.setCurrentIndex(i)
                        break

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(str(ok_text or "OK"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def selected_scope(self) -> str:
        if not self._show_scope or self._scope_folder is None:
            return "current"
        return "folder" if self._scope_folder.isChecked() else "current"

    def settings_dict(self) -> dict:
        out: dict = {}

        def _txt(key: str) -> str | None:
            w = self._w.get(key)
            if w is None:
                return None
            if isinstance(w, QtWidgets.QLineEdit):
                return w.text().strip()
            return None

        def _bool(key: str) -> bool | None:
            w = self._w.get(key)
            if w is None:
                return None
            if isinstance(w, QtWidgets.QAbstractButton):
                return bool(w.isChecked())
            return None

        def _int(key: str) -> int | None:
            w = self._w.get(key)
            if w is None:
                return None
            if isinstance(w, QtWidgets.QSpinBox):
                return int(w.value())
            return None

        def _float(key: str) -> float | None:
            w = self._w.get(key)
            if w is None:
                return None
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                return float(w.value())
            return None

        def _combo_text(key: str) -> str | None:
            w = self._w.get(key)
            if w is None:
                return None
            if isinstance(w, QtWidgets.QComboBox):
                return str(w.currentText())
            return None

        def _combo_data(key: str) -> str | None:
            w = self._w.get(key)
            if w is None:
                return None
            if isinstance(w, QtWidgets.QComboBox):
                d = w.currentData()
                return str(d) if d is not None else str(w.currentText())
            return None

        # SAM
        v = _txt("sam_checkpoint")
        if v is not None:
            out["sam_checkpoint"] = v
        v = _combo_text("sam_model_type")
        if v is not None:
            out["sam_model_type"] = v
        v = _combo_text("delta_type")
        if v is not None:
            out["delta_type"] = v
        v = _txt("delta_checkpoint")
        if v is not None:
            out["delta_checkpoint"] = v
        v = _int("middle_dim")
        if v is not None:
            out["middle_dim"] = v
        v = _float("scaling_factor")
        if v is not None:
            out["scaling_factor"] = v
        v = _int("rank")
        if v is not None:
            out["rank"] = v
        v = _bool("invert_mask")
        if v is not None:
            out["invert_mask"] = v
        v = _int("min_area")
        if v is not None:
            out["min_area"] = v
        v = _int("dilate")
        if v is not None:
            out["dilate"] = v

        # DINO
        v = _txt("dino_checkpoint")
        if v is not None:
            out["dino_checkpoint"] = v
        v = _combo_data("dino_config_id")
        if v is not None:
            out["dino_config_id"] = v
        v = _combo_text("device")
        if v is not None:
            out["device"] = v
        v = _txt("text_queries")
        if v is not None:
            out["text_queries"] = v
        v = _float("box_threshold")
        if v is not None:
            out["box_threshold"] = v
        v = _float("text_threshold")
        if v is not None:
            out["text_threshold"] = v
        v = _int("max_dets")
        if v is not None:
            out["max_dets"] = v

        # UNet
        v = _txt("unet_model")
        if v is not None:
            out["unet_model"] = v
        v = _float("unet_threshold")
        if v is not None:
            out["unet_threshold"] = v
        v = _bool("unet_post")
        if v is not None:
            out["unet_post"] = v
        v = _combo_text("unet_mode")
        if v is not None:
            out["unet_mode"] = v
        v = _int("unet_input_size")
        if v is not None:
            out["unet_input_size"] = v
        v = _int("unet_overlap")
        if v is not None:
            out["unet_overlap"] = v
        v = _int("unet_tile_batch")
        if v is not None:
            out["unet_tile_batch"] = v

        return out

    def _build_sam_page(self, settings: dict) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        sam_ckpt = QtWidgets.QLineEdit(page)
        sam_ckpt.setText(str(settings.get("sam_checkpoint") or ""))
        sam_ckpt_btn = QtWidgets.QPushButton("Browse...", page)
        sam_ckpt_btn.clicked.connect(lambda: browse_file(self, sam_ckpt, "SAM checkpoint (*.pth);;All files (*)"))

        sam_type = QtWidgets.QComboBox(page)
        sam_type.addItems(["auto", "vit_b", "vit_l", "vit_h"])
        if settings.get("sam_model_type"):
            sam_type.setCurrentText(str(settings.get("sam_model_type")))

        sam_group = QtWidgets.QGroupBox("SAM", page)
        sam_layout = QtWidgets.QVBoxLayout(sam_group)
        sam_layout.setContentsMargins(10, 10, 10, 10)
        sam_layout.setSpacing(6)
        sam_layout.addWidget(QtWidgets.QLabel("SAM checkpoint", sam_group))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(sam_ckpt, 1)
        row.addWidget(sam_ckpt_btn)
        sam_layout.addLayout(row)
        sam_layout.addWidget(QtWidgets.QLabel("SAM type", sam_group))
        sam_layout.addWidget(sam_type)
        layout.addWidget(sam_group)

        # --- Delta ---
        # In this dialog, user can configure delta settings.
        # Usage depends on mode selection in main window, but here we just show config.
        # If we want to mimic main window logic: "Set path to enable".
        
        delta_group = QtWidgets.QGroupBox("Fine-tune (delta) - Set path to enable", page)
        delta_group.setVisible(True)
        dg = QtWidgets.QVBoxLayout(delta_group)
        dg.setContentsMargins(10, 10, 10, 10)
        dg.setSpacing(6)

        delta_type = QtWidgets.QComboBox(delta_group)
        delta_type.addItems(["adapter", "lora", "both"])
        if settings.get("delta_type"):
            delta_type.setCurrentText(str(settings.get("delta_type")))

        delta_ckpt = QtWidgets.QLineEdit(delta_group)
        delta_ckpt.setText(str(settings.get("delta_checkpoint") or "auto"))
        delta_ckpt.setPlaceholderText("auto or path/to/delta.pth")
        delta_btn = QtWidgets.QPushButton("Browse...", delta_group)
        delta_btn.clicked.connect(lambda: browse_file(self, delta_ckpt, "Delta checkpoint (*.pth);;All files (*)"))

        middle_dim = QtWidgets.QSpinBox(delta_group)
        middle_dim.setRange(1, 4096)
        middle_dim.setValue(int(settings.get("middle_dim", 32)))

        scaling = QtWidgets.QDoubleSpinBox(delta_group)
        scaling.setRange(0.0, 10.0)
        scaling.setDecimals(4)
        scaling.setSingleStep(0.05)
        scaling.setValue(float(settings.get("scaling_factor", 0.2)))

        rank = QtWidgets.QSpinBox(delta_group)
        rank.setRange(1, 1024)
        rank.setValue(int(settings.get("rank", 4)))

        dg.addWidget(QtWidgets.QLabel("Delta type", delta_group))
        dg.addWidget(delta_type)
        dg.addWidget(QtWidgets.QLabel("Delta checkpoint", delta_group))
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(delta_ckpt, 1)
        row2.addWidget(delta_btn)
        dg.addLayout(row2)
        dg.addWidget(QtWidgets.QLabel("Adapter middle_dim", delta_group))
        dg.addWidget(middle_dim)
        dg.addWidget(QtWidgets.QLabel("Adapter scaling_factor", delta_group))
        dg.addWidget(scaling)
        dg.addWidget(QtWidgets.QLabel("LoRA rank", delta_group))
        dg.addWidget(rank)

        layout.addWidget(delta_group)

        invert = QtWidgets.QCheckBox("Invert mask", page)
        invert.setChecked(bool(settings.get("invert_mask", False)))
        min_area = QtWidgets.QSpinBox(page)
        min_area.setRange(0, 10_000_000)
        min_area.setValue(int(settings.get("min_area", 0)))
        dilate = QtWidgets.QSpinBox(page)
        dilate.setRange(0, 50)
        dilate.setValue(int(settings.get("dilate", 0)))

        mask_group = QtWidgets.QGroupBox("Mask postprocess", page)
        mask_layout = QtWidgets.QVBoxLayout(mask_group)
        mask_layout.setContentsMargins(10, 10, 10, 10)
        mask_layout.setSpacing(6)
        mask_layout.addWidget(invert)
        mask_layout.addWidget(QtWidgets.QLabel("Min component area", mask_group))
        mask_layout.addWidget(min_area)
        mask_layout.addWidget(QtWidgets.QLabel("Dilate iters", mask_group))
        mask_layout.addWidget(dilate)
        layout.addWidget(mask_group)

        layout.addStretch(1)

        self._w["sam_checkpoint"] = sam_ckpt
        self._w["sam_model_type"] = sam_type
        self._w["delta_type"] = delta_type
        self._w["delta_checkpoint"] = delta_ckpt
        self._w["middle_dim"] = middle_dim
        self._w["scaling_factor"] = scaling
        self._w["rank"] = rank
        self._w["invert_mask"] = invert
        self._w["min_area"] = min_area
        self._w["dilate"] = dilate
        self._w["delta_checkpoint"] = delta_ckpt
        self._w["middle_dim"] = middle_dim
        self._w["scaling_factor"] = scaling
        self._w["rank"] = rank
        self._w["invert_mask"] = invert
        self._w["min_area"] = min_area
        self._w["dilate"] = dilate
        return page

    def _build_dino_page(self, settings: dict) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        dino_ckpt = QtWidgets.QLineEdit(page)
        dino_ckpt.setText(str(settings.get("dino_checkpoint") or ""))
        dino_ckpt.setPlaceholderText("path/to/groundingdino.pth or HF model id")
        dino_btn = QtWidgets.QPushButton("Browse...", page)
        dino_btn.clicked.connect(
            lambda: browse_file(self, dino_ckpt, "GroundingDINO checkpoint (*.pth);;All files (*)")
        )
        dino_dl_btn = QtWidgets.QPushButton("Download...", page)
        dino_dl_btn.setToolTip("Download a HuggingFace GroundingDINO repo to a local folder for offline use.")

        dino_cfg = QtWidgets.QComboBox(page)
        dino_cfg.addItem("Auto (infer from filename)", "auto")
        dino_cfg.addItem("grounding-dino-base", "IDEA-Research/grounding-dino-base")
        dino_cfg.addItem("grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny")
        cfg_id = str(settings.get("dino_config_id") or "auto")
        set_combo_by_data(dino_cfg, cfg_id)

        gdino_group = QtWidgets.QGroupBox("GroundingDINO", page)
        gdino_layout = QtWidgets.QVBoxLayout(gdino_group)
        gdino_layout.setContentsMargins(10, 10, 10, 10)
        gdino_layout.setSpacing(6)
        gdino_layout.addWidget(QtWidgets.QLabel("Checkpoint / model id", gdino_group))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(dino_ckpt, 1)
        row.addWidget(dino_btn)
        row.addWidget(dino_dl_btn)
        gdino_layout.addLayout(row)
        gdino_layout.addWidget(QtWidgets.QLabel("Config", gdino_group))
        gdino_layout.addWidget(dino_cfg)

        device = QtWidgets.QComboBox(page)
        device.addItems(["cpu", "auto", "cuda"])
        device.setCurrentText(str(settings.get("device") or "cpu"))
        gdino_layout.addWidget(QtWidgets.QLabel("Device (SAM + DINO)", gdino_group))
        gdino_layout.addWidget(device)
        layout.addWidget(gdino_group)

        def _download() -> None:
            items = ["IDEA-Research/grounding-dino-base", "IDEA-Research/grounding-dino-tiny"]
            current = str(dino_cfg.currentData() or dino_ckpt.text() or items[0])
            if current not in items:
                current = items[0]
            model_id, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Download GroundingDINO",
                "Model repo id:",
                items,
                items.index(current) if current in items else 0,
                False,
            )
            if not ok:
                return

            repo_root = Path(__file__).resolve().parents[2]
            safe_name = str(model_id).replace("/", "__")
            default_dir = str(repo_root / "checkpoints" / "hf" / safe_name)
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select download folder", default_dir)
            if not out_dir:
                return

            dlg = ProcessingDialog(self, f"Downloading {model_id}")
            dlg.stop_button().setEnabled(False)
            dlg.show()

            class _DL(QtCore.QObject):
                log = QtCore.Signal(str)
                failed = QtCore.Signal(str)
                done = QtCore.Signal(str)

                @QtCore.Slot()
                def run(self) -> None:
                    try:
                        from predict.dino.download import download_hf_model

                        path = download_hf_model(str(model_id), out_dir, log_fn=self.log.emit)
                        self.done.emit(path)
                    except Exception as e:
                        self.failed.emit(str(e))

            thread = QtCore.QThread(self)
            worker = _DL()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)

            def _append(s: str) -> None:
                dlg.log_widget().appendPlainText(str(s))

            worker.log.connect(_append)
            worker.failed.connect(lambda m: _append(f"FAILED: {m}"))

            def _finish(path: str) -> None:
                dino_ckpt.setText(str(path))
                set_combo_by_data(dino_cfg, "auto")
                _append("DONE")

            worker.done.connect(_finish)
            worker.done.connect(thread.quit)
            worker.failed.connect(thread.quit)

            def _cleanup() -> None:
                dlg.allow_close()
                dlg.close()
                worker.deleteLater()
                thread.deleteLater()

            thread.finished.connect(_cleanup)
            thread.start()

        dino_dl_btn.clicked.connect(_download)

        queries = QtWidgets.QLineEdit(page)
        queries.setText(str(settings.get("text_queries") or ""))
        layout.addWidget(QtWidgets.QLabel("Text queries (comma)", page))
        layout.addWidget(queries)

        box_thr = QtWidgets.QDoubleSpinBox(page)
        box_thr.setRange(0.0, 1.0)
        box_thr.setSingleStep(0.01)
        box_thr.setValue(float(settings.get("box_threshold", 0.25)))
        text_thr = QtWidgets.QDoubleSpinBox(page)
        text_thr.setRange(0.0, 1.0)
        text_thr.setSingleStep(0.01)
        text_thr.setValue(float(settings.get("text_threshold", 0.25)))
        max_dets = QtWidgets.QSpinBox(page)
        max_dets.setRange(0, 999)
        max_dets.setValue(int(settings.get("max_dets", 20)))

        thr_group = QtWidgets.QGroupBox("Thresholds", page)
        thr_layout = QtWidgets.QVBoxLayout(thr_group)
        thr_layout.setContentsMargins(10, 10, 10, 10)
        thr_layout.setSpacing(6)
        thr_layout.addWidget(QtWidgets.QLabel("Box thr", thr_group))
        thr_layout.addWidget(box_thr)
        thr_layout.addWidget(QtWidgets.QLabel("Text thr", thr_group))
        thr_layout.addWidget(text_thr)
        thr_layout.addWidget(QtWidgets.QLabel("Max dets", thr_group))
        thr_layout.addWidget(max_dets)
        layout.addWidget(thr_group)


        layout.addStretch(1)

        self._w["dino_checkpoint"] = dino_ckpt
        self._w["dino_config_id"] = dino_cfg
        self._w["device"] = device
        self._w["text_queries"] = queries
        self._w["box_threshold"] = box_thr
        self._w["text_threshold"] = text_thr
        self._w["max_dets"] = max_dets
        return page

    def _build_unet_page(self, settings: dict) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        model = QtWidgets.QLineEdit(page)
        model.setText(str(settings.get("unet_model") or ""))
        model_btn = QtWidgets.QPushButton("Browse...", page)
        model_btn.clicked.connect(lambda: browse_file(self, model, "Model (*.pth);;All files (*)"))
        layout.addWidget(QtWidgets.QLabel("Model (.pth)", page))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(model, 1)
        row.addWidget(model_btn)
        layout.addLayout(row)

        thr = QtWidgets.QDoubleSpinBox(page)
        thr.setRange(0.0, 1.0)
        thr.setSingleStep(0.01)
        thr.setValue(float(settings.get("unet_threshold", 0.5)))

        post = QtWidgets.QCheckBox("Apply postprocessing", page)
        post.setChecked(bool(settings.get("unet_post", True)))

        mode = QtWidgets.QComboBox(page)
        mode.addItems(["tile", "letterbox", "resize"])
        if settings.get("unet_mode"):
            mode.setCurrentText(str(settings.get("unet_mode")))

        input_size = QtWidgets.QSpinBox(page)
        input_size.setRange(64, 4096)
        input_size.setValue(int(settings.get("unet_input_size", 256)))

        overlap = QtWidgets.QSpinBox(page)
        overlap.setRange(0, 4096)
        overlap.setValue(int(settings.get("unet_overlap", 0)))

        tile_batch = QtWidgets.QSpinBox(page)
        tile_batch.setRange(1, 64)
        tile_batch.setValue(int(settings.get("unet_tile_batch", 4)))

        layout.addWidget(QtWidgets.QLabel("Threshold", page))
        layout.addWidget(thr)
        layout.addWidget(QtWidgets.QLabel("Mode", page))
        layout.addWidget(mode)
        layout.addWidget(QtWidgets.QLabel("Input size", page))
        layout.addWidget(input_size)
        layout.addWidget(QtWidgets.QLabel("Tile overlap", page))
        layout.addWidget(overlap)
        layout.addWidget(QtWidgets.QLabel("Tile batch size", page))
        layout.addWidget(tile_batch)
        layout.addWidget(post)
        layout.addStretch(1)

        self._w["unet_model"] = model
        self._w["unet_threshold"] = thr
        self._w["unet_post"] = post
        self._w["unet_mode"] = mode
        self._w["unet_input_size"] = input_size
        self._w["unet_overlap"] = overlap
        self._w["unet_tile_batch"] = tile_batch
        return page
