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
        self.prev_page_btn = QtWidgets.QPushButton("Prev", self)
        self.prev_page_btn.setFixedWidth(58)
        toolbar.add(self.prev_page_btn)
        self.page_chip = Chip("0/0", self)
        toolbar.add(self.page_chip)
        self.next_page_btn = QtWidgets.QPushButton("Next", self)
        self.next_page_btn.setFixedWidth(58)
        toolbar.add(self.next_page_btn)
        toolbar.add_label("Page size")
        self.page_size = QtWidgets.QSpinBox(self)
        self.page_size.setRange(50, 5000)
        self.page_size.setSingleStep(50)
        self.page_size.setValue(500)
        self.page_size.setFixedWidth(86)
        toolbar.add(self.page_size)
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
        self.prev_page_btn.clicked.connect(self._prev_page)
        self.next_page_btn.clicked.connect(self._next_page)
        self.page_size.valueChanged.connect(self._reset_page_size)
        self.apply_btn.clicked.connect(self._bulk_relabel)
        self.commit_btn.clicked.connect(self._commit)
        self.grid.selectionChanged.connect(lambda items: self.sel_chip.setText(f"{len(items)} đã chọn"))
        self.store.cleanedChanged.connect(self._on_cleaned_changed)
        self.store.pendingChanged.connect(self._render_commit)
        self._render_commit()
        self._current_label = ""
        self._offset = 0
        self._limit = 500
        self._render_pager()

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
        self._current_label = str(label or "")
        self._offset = 0
        self._load_page()

    def _load_page(self) -> None:
        if not self._current_label:
            self.grid.set_items([])
            self._render_pager()
            return
        self._limit = int(self.page_size.value())
        self.controller.load_cleaned(final_label=self._current_label, limit=self._limit, offset=self._offset)

    def _on_cleaned_changed(self) -> None:
        self.grid.set_items(self.store.cleaned_items)
        self._render_pager()

    def _prev_page(self) -> None:
        self._limit = int(self.page_size.value())
        self._offset = max(0, self._offset - self._limit)
        self._load_page()

    def _next_page(self) -> None:
        loaded = len(self.store.cleaned_items)
        total = int(self.store.cleaned_filtered_total or 0)
        if loaded <= 0:
            return
        self._limit = int(self.page_size.value())
        next_offset = self._offset + self._limit
        if total and next_offset >= total:
            return
        self._offset = next_offset
        self._load_page()

    def _reset_page_size(self, _value: int) -> None:
        if not self._current_label or not self.store.cleaned_items:
            self._render_pager()
            return
        self._offset = 0
        self._load_page()

    def _render_pager(self) -> None:
        loaded = len(self.store.cleaned_items)
        total = int(self.store.cleaned_filtered_total or 0)
        offset = int(self.store.cleaned_offset or self._offset)
        limit = int(self.store.cleaned_limit or self.page_size.value())
        if self._current_label and loaded and total:
            start = offset + 1
            end = offset + loaded
            page = (offset // max(1, limit)) + 1
            pages = ((total - 1) // max(1, limit)) + 1
            self.page_chip.setText(f"{self._current_label}: {page}/{pages} ({start}-{end}/{total})")
        elif self._current_label:
            self.page_chip.setText(f"{self._current_label}: 0/0")
        else:
            self.page_chip.setText("Choose label")
        self.prev_page_btn.setEnabled(offset > 0)
        self.next_page_btn.setEnabled(bool(total and offset + loaded < total))

    def _bulk_relabel(self) -> None:
        label = self.relabel.currentText()
        selected = self.grid.selected_items()
        if not selected:
            QtWidgets.QMessageBox.information(self, "Đổi nhãn", "Chưa chọn ảnh nào.")
            return
        for item in selected:
            self.controller.update_cleaned(item, label)
        self._render_commit()

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
