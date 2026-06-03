from __future__ import annotations

from PySide6 import QtWidgets

from ...config.defaults import PROTOTYPE_LABELS
from ...stores.prototype_store import PrototypeStore
from ...stores.run_store import RunStore
from ..widgets.step_log import StepLog
from ..widgets.thumb_grid import ThumbGrid
from ..widgets.ui_kit import Card, Chip, PickedList, Toolbar, primary_button


class PrototypeWorkspace(QtWidgets.QWidget):
    def __init__(self, store: PrototypeStore, run_store: RunStore, controller, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.run_store = run_store
        self.controller = controller
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Thanh công cụ: thiết lập + tải + tóm tắt + chạy ─────────────
        toolbar = Toolbar(self)
        toolbar.add_label("Ngưỡng loại <")
        self.reject_spin = QtWidgets.QDoubleSpinBox(self)
        self.reject_spin.setRange(0.0, 1.0)
        self.reject_spin.setSingleStep(0.05)
        self.reject_spin.setDecimals(2)
        self.reject_spin.setValue(float(self.controller.settings.get("reject_below", 0.5)))
        self.reject_spin.setFixedWidth(78)
        toolbar.add(self.reject_spin)
        toolbar.add_label("Mỗi dải")
        self.band_spin = QtWidgets.QSpinBox(self)
        self.band_spin.setRange(10, 2000)
        self.band_spin.setValue(int(self.controller.settings.get("per_band", 200)))
        self.band_spin.setFixedWidth(78)
        toolbar.add(self.band_spin)
        self.refresh_btn = QtWidgets.QPushButton("Tải ứng viên", self)
        toolbar.add(self.refresh_btn)
        toolbar.add_separator()
        self.pick_chip = Chip("0 đã chọn", self)
        toolbar.add(self.pick_chip)
        toolbar.add_stretch()
        self.step05_btn = QtWidgets.QPushButton("Chỉ tạo bank (step05)", self)
        toolbar.add(self.step05_btn)
        self.chain_btn = primary_button("Tạo bank & chạy 06→08", self)
        toolbar.add(self.chain_btn)
        root.addWidget(toolbar)

        # ── Thân: thư viện ứng viên | đã chọn + bank + log ──────────────
        splitter = QtWidgets.QSplitter(self)
        gallery = Card("Ứng viên prototype", splitter)
        hint = QtWidgets.QLabel("Bấm vào ảnh để chọn / bỏ chọn — ảnh đã chọn có viền xanh và dấu ✓", gallery)
        hint.setObjectName("InfoKey")
        hint.setWordWrap(True)
        gallery.add(hint)
        self.label_tabs = QtWidgets.QTabWidget(gallery)
        self.grids: dict[str, ThumbGrid] = {}
        self._tab_index: dict[str, int] = {}
        for idx, label in enumerate(PROTOTYPE_LABELS):
            grid = ThumbGrid(gallery)
            grid.itemActivated.connect(lambda item, lab=label: self._toggle_pick(item, lab))
            self.grids[label] = grid
            self._tab_index[label] = idx
            self.label_tabs.addTab(grid, f"{label} (0)")
        gallery.add(self.label_tabs, 1)
        splitter.addWidget(gallery)

        right = QtWidgets.QWidget(splitter)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        picked_card = Card("Đã chọn", right)
        self.picked_list = PickedList(picked_card)
        self.picked_empty = QtWidgets.QLabel("Chưa chọn ảnh nào.", picked_card)
        self.picked_empty.setObjectName("InfoKey")
        picked_card.add(self.picked_empty)
        picked_card.add(self.picked_list, 1)
        right_layout.addWidget(picked_card, 2)

        bank_card = Card("Bank mới nhất", right)
        self.latest = QtWidgets.QLabel("Chưa tải ứng viên. Bấm “Tải ứng viên” để bắt đầu.", bank_card)
        self.latest.setWordWrap(True)
        self.latest.setObjectName("InfoVal")
        bank_card.add(self.latest)
        right_layout.addWidget(bank_card)

        log_card = Card("Tiến trình", right)
        self.log = StepLog(log_card)
        log_card.add(self.log, 1)
        right_layout.addWidget(log_card, 2)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        # ── Kết nối (giữ nguyên hợp đồng controller/store) ──────────────
        self.refresh_btn.clicked.connect(self._refresh)
        self.step05_btn.clicked.connect(self.controller.run_step05_only)
        self.chain_btn.clicked.connect(self.controller.run_prototype_chain)
        self.picked_list.removeRequested.connect(self._unpick)
        self.log.stopRequested.connect(
            lambda: getattr(self.controller, "_pipeline", None) and self.controller._pipeline.stop()
        )
        self.store.candidatesChanged.connect(self._render_candidates)
        self.store.picksChanged.connect(self._render_picks)
        self.store.errorRaised.connect(lambda msg: QtWidgets.QMessageBox.warning(self, "Prototype", msg))
        self.run_store.logChanged.connect(lambda: self.log.log.setPlainText(self.run_store.log_text))
        self.run_store.statusChanged.connect(self.log.set_status)
        self._render_picks()

    def _refresh(self) -> None:
        self.controller.settings["reject_below"] = float(self.reject_spin.value())
        self.controller.settings["per_band"] = int(self.band_spin.value())
        self.controller.refresh()

    def _candidate_by_id(self, result_id: int) -> object | None:
        for item in self.store.candidates:
            if int(getattr(item, "result_id", -1)) == int(result_id):
                return item
        return None

    def _render_candidates(self) -> None:
        by_label = {label: [] for label in PROTOTYPE_LABELS}
        for item in self.store.candidates:
            by_label.setdefault(getattr(item, "label", ""), []).append(item)
        for label, grid in self.grids.items():
            items = by_label.get(label, [])
            grid.set_items(items)
            if label in self._tab_index:
                self.label_tabs.setTabText(self._tab_index[label], f"{label} ({len(items)})")
        proto = self.store.latest_prototype
        if proto:
            self.latest.setText(f"Bank #{proto.get('prototype_version_id')} — {proto.get('item_count')} mục")
        else:
            self.latest.setText(f"Đã tải {len(self.store.candidates)} ứng viên. Chưa có bank.")
        self._sync_picked_overlay()

    def _toggle_pick(self, item: object, label: str) -> None:
        item_label = str(getattr(item, "label", label))
        self.store.toggle_pick(int(getattr(item, "result_id")), item_label, item_label == "reject")

    def _unpick(self, result_id: int) -> None:
        pick = self.store.picks.get(int(result_id))
        if pick is None:
            return
        # toggle_pick removes it when called with the same value
        self.store.toggle_pick(int(result_id), pick.get("label", "reject"), bool(pick.get("is_reject")))

    def _sync_picked_overlay(self) -> None:
        picked_ids = {int(rid) for rid in self.store.picks}
        for grid in self.grids.values():
            grid.set_picked_ids(picked_ids)

    def _render_picks(self) -> None:
        picks = self.store.picks
        count = len(picks)
        self.pick_chip.setText(f"{count} đã chọn")
        entries: list[tuple[int, str]] = []
        for result_id, pick in picks.items():
            label = str(pick.get("label") or "reject")
            entries.append((int(result_id), f"#{result_id}  ·  {label}"))
        self.picked_list.set_entries(entries)
        self.picked_empty.setVisible(count == 0)
        self.picked_list.setVisible(count > 0)
        self._sync_picked_overlay()
