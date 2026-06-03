from __future__ import annotations

import json

from PySide6 import QtWidgets

from ...services.step_runner import STEP_MODULES
from ...stores.run_store import RunStore
from ..widgets.step_log import StepLog
from ..widgets.ui_kit import Card, Toolbar, danger_button, primary_button


class RunStepsWorkspace(QtWidgets.QWidget):
    def __init__(self, run_store: RunStore, controller, settings: dict, parent=None) -> None:
        super().__init__(parent)
        self.store = run_store
        self.controller = controller
        self.settings = settings
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Toolbar: pick a step + run + chain ──────────────────────────
        toolbar = Toolbar(self)
        toolbar.add_label("Bước")
        self.step = QtWidgets.QComboBox(self)
        self.step.addItems(["step04", "step05", "step06", "step07", "step08", "step09", "export_dataset"])
        self.step.setMinimumWidth(150)
        toolbar.add(self.step)
        self.run_btn = primary_button("Chạy bước", self)
        toolbar.add(self.run_btn)
        toolbar.add_separator()
        self.chain_btn = QtWidgets.QPushButton("Chạy chuỗi 05→08", self)
        toolbar.add(self.chain_btn)
        toolbar.add_stretch()
        root.addWidget(toolbar)

        # ── step09 caution card ─────────────────────────────────────────
        step09_card = Card("step09 — Tự huấn luyện (cẩn trọng)", self)
        step09_row = QtWidgets.QHBoxLayout()
        step09_row.setSpacing(8)
        self.audit_btn = QtWidgets.QPushButton("Kiểm tra (không ghi)", step09_card)
        self.apply_btn = danger_button("Áp dụng promotions", step09_card)
        warn = QtWidgets.QLabel("Áp dụng sẽ ghi đè cleaned_labels + semantic_decisions.", step09_card)
        warn.setObjectName("InfoKey")
        step09_row.addWidget(self.audit_btn)
        step09_row.addWidget(self.apply_btn)
        step09_row.addWidget(warn, 1)
        step09_card.body().addLayout(step09_row)
        root.addWidget(step09_card)

        # ── Flags editor ────────────────────────────────────────────────
        flags_card = Card("Tham số (JSON)", self)
        self.flags = QtWidgets.QPlainTextEdit(flags_card)
        self.flags.setMaximumHeight(150)
        self.flags.setPlainText(json.dumps(self._default_flags("step06"), indent=2))
        flags_card.add(self.flags)
        root.addWidget(flags_card)

        # ── Log ─────────────────────────────────────────────────────────
        log_card = Card("Nhật ký", self)
        self.log = StepLog(log_card)
        log_card.add(self.log, 1)
        root.addWidget(log_card, 1)

        # ── Wiring ──────────────────────────────────────────────────────
        self.step.currentTextChanged.connect(
            lambda step: self.flags.setPlainText(json.dumps(self._default_flags(step), indent=2))
        )
        self.run_btn.clicked.connect(self._run)
        self.audit_btn.clicked.connect(lambda: self._run_step09(False))
        self.apply_btn.clicked.connect(lambda: self._run_step09(True))
        self.chain_btn.clicked.connect(self.controller.run_chain_05_08)
        self.log.stopRequested.connect(self.controller.stop)
        self.store.logChanged.connect(lambda: self.log.log.setPlainText(self.store.log_text))
        self.store.statusChanged.connect(self.log.set_status)

    def _default_flags(self, step: str) -> dict:
        flags = {"--db": self.settings["db_path"], "--run-id": self.settings.get("run_id", "myrun")}
        if step in {"step04", "step05", "step08"}:
            flags["--model-name"] = self.settings.get("model_name", "facebook/dinov2-giant")
        if step in {"step04", "step05"}:
            flags["--view-name"] = self.settings.get("view_name", "tight")
        if step == "export_dataset":
            flags.update(
                {
                    "--image-root": self.settings.get("image_root", ""),
                    "--output-dir": self.settings.get("export_dir", ""),
                    "--format": self.settings.get("export_format", "yolo"),
                }
            )
        return flags

    def _run(self) -> None:
        step = self.step.currentText()
        if step not in STEP_MODULES:
            return
        try:
            flags = json.loads(self.flags.toPlainText() or "{}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Flags", str(exc))
            return
        self.controller.run_step(step, flags)

    def _run_step09(self, apply: bool) -> None:
        flags = self._default_flags("step09")
        if apply:
            ok = QtWidgets.QMessageBox.question(
                self,
                "Áp dụng promotions",
                "Áp dụng promotions sẽ ghi đè cleaned_labels và semantic_decisions. Tiếp tục?",
            )
            if ok != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            flags["--apply-promotions"] = True
        self.controller.run_step("step09", flags)
