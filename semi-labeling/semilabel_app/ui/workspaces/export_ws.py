from __future__ import annotations

from PySide6 import QtWidgets

from ...stores.run_store import RunStore
from ..widgets.step_log import StepLog
from ..widgets.ui_kit import Card, primary_button


class ExportWorkspace(QtWidgets.QWidget):
    def __init__(self, run_store: RunStore, controller, settings: dict, parent=None) -> None:
        super().__init__(parent)
        self.store = run_store
        self.controller = controller
        self.settings = settings
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Export form ─────────────────────────────────────────────────
        form_card = Card("Xuất dataset", self)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QtWidgets.QLabel("Thư mục đích", form_card), 0, 0)
        self.out_dir = QtWidgets.QLineEdit(str(self.settings.get("export_dir", "")), form_card)
        self.out_dir.setPlaceholderText("Chọn thư mục xuất…")
        grid.addWidget(self.out_dir, 0, 1)
        self.browse_btn = QtWidgets.QPushButton("Chọn…", form_card)
        grid.addWidget(self.browse_btn, 0, 2)

        grid.addWidget(QtWidgets.QLabel("Định dạng", form_card), 1, 0)
        self.format = QtWidgets.QComboBox(form_card)
        self.format.addItems(["yolo", "coco"])
        self.format.setFixedWidth(140)
        grid.addWidget(self.format, 1, 1)

        form_card.body().addLayout(grid)
        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch(1)
        self.run_btn = primary_button("Xuất", form_card)
        action_row.addWidget(self.run_btn)
        form_card.body().addLayout(action_row)
        root.addWidget(form_card)

        # ── Log ─────────────────────────────────────────────────────────
        log_card = Card("Nhật ký", self)
        self.log = StepLog(log_card)
        log_card.add(self.log, 1)
        root.addWidget(log_card, 1)

        # ── Wiring ──────────────────────────────────────────────────────
        self.browse_btn.clicked.connect(self._browse)
        self.run_btn.clicked.connect(
            lambda: self.controller.export_dataset(self.out_dir.text().strip(), self.format.currentText())
        )
        self.store.logChanged.connect(lambda: self.log.log.setPlainText(self.store.log_text))
        self.store.statusChanged.connect(self.log.set_status)

    def _browse(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Chọn thư mục xuất", self.out_dir.text().strip() or ""
        )
        if directory:
            self.out_dir.setText(directory)
