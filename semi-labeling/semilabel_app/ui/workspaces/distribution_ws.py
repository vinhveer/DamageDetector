from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ...config.defaults import LABELS
from ...services import db_service
from ...stores.review_store import ReviewStore
from ..widgets.thumb_grid import ThumbGrid
from ..widgets.ui_kit import Card, Chip, PercentBar, Toolbar, primary_button


class DistributionWorkspace(QtWidgets.QWidget):
    def __init__(self, review_store: ReviewStore, review_controller, settings: dict, parent=None) -> None:
        super().__init__(parent)
        self.store = review_store
        self.controller = review_controller
        self.settings = settings
        self._workers: list[object] = []
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Toolbar ─────────────────────────────────────────────────────
        toolbar = Toolbar(self)
        self.refresh_btn = QtWidgets.QPushButton("Làm mới", self)
        toolbar.add(self.refresh_btn)
        self.total_chip = Chip("Tổng 0", self)
        toolbar.add(self.total_chip)
        toolbar.add_stretch()
        self.sel_chip = Chip("0 đã chọn", self)
        toolbar.add(self.sel_chip)
        toolbar.add_label("Đổi sang")
        self.relabel = QtWidgets.QComboBox(self)
        self.relabel.addItems(LABELS)
        toolbar.add(self.relabel)
        self.apply_btn = QtWidgets.QPushButton("Đổi nhãn hàng loạt", self)
        toolbar.add(self.apply_btn)
        self.commit_btn = primary_button("Lưu chỉnh sửa (0)", self)
        toolbar.add(self.commit_btn)
        root.addWidget(toolbar)

        # ── Body: distribution table | grid ─────────────────────────────
        splitter = QtWidgets.QSplitter(self)
        dist_card = Card("Phân bố nhãn", splitter)
        self.table = QtWidgets.QTableWidget(0, 3, dist_card)
        self.table.setHorizontalHeaderLabels(["Nhãn", "Số lượng", "Tỷ lệ"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        dist_card.add(self.table, 1)
        splitter.addWidget(dist_card)

        grid_card = Card("Ảnh theo nhãn", splitter)
        self.grid = ThumbGrid(grid_card)
        grid_card.add(self.grid, 1)
        splitter.addWidget(grid_card)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        # ── Wiring ──────────────────────────────────────────────────────
        self.refresh_btn.clicked.connect(self.refresh)
        self.table.cellClicked.connect(lambda row, _col: self._load_label(self.table.item(row, 0).text()))
        self.apply_btn.clicked.connect(self._bulk_relabel)
        self.commit_btn.clicked.connect(self._commit)
        self.grid.selectionChanged.connect(lambda items: self.sel_chip.setText(f"{len(items)} đã chọn"))
        self.store.cleanedChanged.connect(lambda: self.grid.set_items(self.store.cleaned_items))
        self.store.pendingChanged.connect(self._render_commit)
        self._render_commit()

    def refresh(self) -> None:
        worker = db_service.DbWorker(
            db_service.cleaned_distribution, self.settings["db_path"], self.settings.get("run_id", "myrun")
        )
        worker.signals.finished.connect(self._render_dist)
        worker.signals.error.connect(lambda msg: QtWidgets.QMessageBox.warning(self, "Distribution", msg))
        worker.signals.finished.connect(lambda _result, w=worker: self._release_worker(w))
        worker.signals.error.connect(lambda _message, w=worker: self._release_worker(w))
        self._workers.append(worker)
        QtCore.QThreadPool.globalInstance().start(worker)

    def _release_worker(self, worker: object) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def _render_dist(self, dist) -> None:
        rows = list(dist.by_label)
        total = sum(count for _label, count, _pct in rows) or 0
        self.total_chip.setText(f"Tổng {total}")
        self.table.setRowCount(len(rows))
        for row, (label, count, pct) in enumerate(rows):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(label)))
            count_item = QtWidgets.QTableWidgetItem(str(count))
            count_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, count_item)
            self.table.setCellWidget(row, 2, PercentBar(float(pct), f"{pct:.1%}"))

    def _load_label(self, label: str) -> None:
        self.controller.load_cleaned(final_label=label, limit=500)

    def _bulk_relabel(self) -> None:
        label = self.relabel.currentText()
        selected = self.grid.selected_items()
        if not selected:
            QtWidgets.QMessageBox.information(self, "Đổi nhãn", "Chưa chọn ảnh nào.")
            return
        for item in selected:
            self.controller.update_cleaned(item, label)
        self.controller.load_cleaned(limit=500)

    def _commit(self) -> None:
        payload = self.controller.commit_pending_corrections()
        if payload.get("error"):
            QtWidgets.QMessageBox.warning(self, "Commit", payload["error"])
            return
        QtWidgets.QMessageBox.information(
            self,
            "Commit",
            f"Committed {payload.get('decisionCount', 0)} corrections",
        )

    def _render_commit(self) -> None:
        count = len(self.store.pending_corrections)
        self.commit_btn.setText(f"Lưu chỉnh sửa ({count})")
        self.commit_btn.setEnabled(count > 0)
