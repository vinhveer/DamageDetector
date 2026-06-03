from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from ...config.defaults import LABELS
from ...stores.review_store import ReviewStore
from ..widgets.box_image import BoxImage
from ..widgets.thumb_grid import ThumbGrid
from ..widgets.ui_kit import Card, Chip, EmptyState, InfoPanel, Toolbar, primary_button


class ReviewWorkspace(QtWidgets.QWidget):
    def __init__(self, store: ReviewStore, controller, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.controller = controller
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Toolbar ─────────────────────────────────────────────────────
        toolbar = Toolbar(self)
        self.mode = QtWidgets.QComboBox(self)
        self.mode.addItems(["Queue", "Cleaned"])
        toolbar.add(self.mode)
        self.queue_type = QtWidgets.QComboBox(self)
        self.queue_type.addItems(
            ["all", "suspect", "relabel_candidate", "reject", "suspect_broad_box", "suspect_composite_box"]
        )
        toolbar.add(self.queue_type)
        toolbar.add_label("Lấy mẫu %")
        self.sample = QtWidgets.QSpinBox(self)
        self.sample.setRange(0, 100)
        self.sample.setValue(10)
        self.sample.setSuffix(" %")
        self.sample.setFixedWidth(74)
        toolbar.add(self.sample)
        self.load_btn = QtWidgets.QPushButton("Tải", self)
        toolbar.add(self.load_btn)
        toolbar.add_separator()
        self.count_chip = Chip("0 mục", self)
        toolbar.add(self.count_chip)
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
        toolbar.add_separator()
        self.show_others = QtWidgets.QCheckBox("Hiện box khác", self)
        self.show_others.setChecked(True)
        toolbar.add(self.show_others)
        toolbar.add_stretch()
        self.commit_btn = primary_button("Lưu (0)", self)
        toolbar.add(self.commit_btn)
        root.addWidget(toolbar)

        # ── Body: grid | detail ─────────────────────────────────────────
        splitter = QtWidgets.QSplitter(self)
        grid_card = Card("Danh sách", splitter)
        self.grid = ThumbGrid(grid_card)
        self.grid.itemActivated.connect(self._select_item)
        self.empty = EmptyState("Chưa có dữ liệu. Chọn chế độ rồi bấm “Tải”.", grid_card)
        self._grid_stack = QtWidgets.QStackedWidget(grid_card)
        self._grid_stack.addWidget(self.empty)
        self._grid_stack.addWidget(self.grid)
        grid_card.add(self._grid_stack, 1)
        splitter.addWidget(grid_card)

        detail = QtWidgets.QWidget(splitter)
        detail_layout = QtWidgets.QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(12)

        image_card = Card("Ảnh", detail)
        title_row = QtWidgets.QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        self.position = QtWidgets.QLabel("—", image_card)
        self.position.setObjectName("InfoKey")
        title_row.addStretch(1)
        title_row.addWidget(self.position)
        image_card.body().addLayout(title_row)
        self.image = BoxImage(image_card)
        image_card.add(self.image, 1)
        detail_layout.addWidget(image_card, 3)

        info_card = Card("Thông tin", detail)
        self.info = InfoPanel(info_card)
        info_card.add(self.info)
        detail_layout.addWidget(info_card, 1)

        action_card = Card("Gán nhãn", detail)
        label_bar = QtWidgets.QHBoxLayout()
        label_bar.setSpacing(6)
        for idx, label in enumerate(LABELS, start=1):
            btn = QtWidgets.QPushButton(f"{label}  [{idx}]", action_card)
            btn.setMinimumHeight(34)
            btn.clicked.connect(lambda _=False, lab=label: self._label_current(lab))
            label_bar.addWidget(btn)
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(str(idx)), self)
            shortcut.activated.connect(lambda lab=label: self._label_current(lab))
        action_card.body().addLayout(label_bar)
        hint = QtWidgets.QLabel("Enter: nhận gợi ý · Space: tiếp · Backspace: lùi", action_card)
        hint.setObjectName("InfoKey")
        action_card.add(hint)
        detail_layout.addWidget(action_card)

        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        # ── Wiring ──────────────────────────────────────────────────────
        QtGui.QShortcut(QtGui.QKeySequence("Return"), self).activated.connect(self.controller.accept_suggestion)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(self.controller.next_item)
        QtGui.QShortcut(QtGui.QKeySequence("Backspace"), self).activated.connect(self.controller.prev_item)
        self.load_btn.clicked.connect(self._load)
        self.prev_page_btn.clicked.connect(self._prev_cleaned_page)
        self.next_page_btn.clicked.connect(self._next_cleaned_page)
        self.page_size.valueChanged.connect(self._reset_cleaned_page_size)
        self.commit_btn.clicked.connect(self._commit)
        self.store.queueChanged.connect(self._on_queue_changed)
        self.store.cleanedChanged.connect(self._on_cleaned_changed)
        self.store.selectionChanged.connect(self._render_current)
        self.store.pendingChanged.connect(self._render_commit)
        self.store.errorRaised.connect(lambda msg: QtWidgets.QMessageBox.warning(self, "Review", msg))
        self.mode.currentTextChanged.connect(self._on_mode_changed)
        self.show_others.toggled.connect(self.image.set_show_others)
        self._on_mode_changed(self.mode.currentText())
        self._render_commit()

        # result_id of the item whose context boxes are currently shown, plus a
        # per-image cache so re-selecting an item doesn't re-hit the DB.
        self._boxes_for: object = None
        self._boxes_cache: dict[str, list] = {}
        self._cleaned_offset = 0
        self._cleaned_limit = 500
        self._render_pager()

    def _on_mode_changed(self, text: str) -> None:
        is_queue = text == "Queue"
        self.queue_type.setEnabled(is_queue)
        self.sample.setEnabled(is_queue)
        self._render_pager()

    def _on_queue_changed(self) -> None:
        self.grid.set_items(self.store.queue_items)
        self._update_count(len(self.store.queue_items))

    def _on_cleaned_changed(self) -> None:
        self.grid.set_items(self.store.cleaned_items)
        self._update_cleaned_count()
        self._render_pager()

    def _update_count(self, count: int) -> None:
        self.count_chip.setText(f"{count} mục")
        self._grid_stack.setCurrentWidget(self.grid if count else self.empty)

    def _update_cleaned_count(self) -> None:
        loaded = len(self.store.cleaned_items)
        total = int(self.store.cleaned_filtered_total or loaded)
        if loaded and total:
            start = int(self.store.cleaned_offset) + 1
            end = int(self.store.cleaned_offset) + loaded
            self.count_chip.setText(f"{start}-{end} / {total}")
        else:
            self.count_chip.setText("0 / 0")
        self._grid_stack.setCurrentWidget(self.grid if loaded else self.empty)

    def _load(self) -> None:
        if self.mode.currentText() == "Queue":
            qt = self.queue_type.currentText()
            self.controller.load_queue("" if qt == "all" else qt, self.sample.value())
        else:
            self._cleaned_offset = 0
            self._load_cleaned_page()

    def _load_cleaned_page(self) -> None:
        self._cleaned_limit = int(self.page_size.value())
        self.controller.load_cleaned(limit=self._cleaned_limit, offset=self._cleaned_offset)

    def _prev_cleaned_page(self) -> None:
        self._cleaned_limit = int(self.page_size.value())
        self._cleaned_offset = max(0, self._cleaned_offset - self._cleaned_limit)
        self._load_cleaned_page()

    def _next_cleaned_page(self) -> None:
        loaded = len(self.store.cleaned_items)
        total = int(self.store.cleaned_filtered_total or 0)
        if loaded <= 0:
            return
        self._cleaned_limit = int(self.page_size.value())
        next_offset = self._cleaned_offset + self._cleaned_limit
        if total and next_offset >= total:
            return
        self._cleaned_offset = next_offset
        self._load_cleaned_page()

    def _reset_cleaned_page_size(self, _value: int) -> None:
        if self.mode.currentText() != "Cleaned" or not self.store.cleaned_items:
            self._render_pager()
            return
        self._cleaned_offset = 0
        self._load_cleaned_page()

    def _render_pager(self) -> None:
        if self.mode.currentText() != "Cleaned":
            self.page_chip.setText("0/0")
            self.prev_page_btn.setEnabled(False)
            self.next_page_btn.setEnabled(False)
            self.page_size.setEnabled(False)
            return
        loaded = len(self.store.cleaned_items)
        total = int(self.store.cleaned_filtered_total or loaded)
        offset = int(self.store.cleaned_offset or self._cleaned_offset)
        limit = int(self.store.cleaned_limit or self.page_size.value())
        if loaded and total:
            page = (offset // max(1, limit)) + 1
            pages = ((total - 1) // max(1, limit)) + 1
            self.page_chip.setText(f"{page}/{pages}")
        else:
            self.page_chip.setText("0/0")
        self.prev_page_btn.setEnabled(offset > 0)
        self.next_page_btn.setEnabled(bool(total and offset + loaded < total))
        self.page_size.setEnabled(True)

    def _select_item(self, item: object) -> None:
        items = self.store.queue_items if self.mode.currentText() == "Queue" else self.store.cleaned_items
        if item in items:
            self.store.set_index(items.index(item), "queue" if self.mode.currentText() == "Queue" else "cleaned")

    def _render_current(self) -> None:
        item = self.store.current_item()
        items = self.store.queue_items if self.store.mode == "queue" else self.store.cleaned_items
        if item is None:
            self.image.clear()
            self.info.clear()
            self.position.setText("—")
            self._boxes_for = None
            return
        self.position.setText(f"{self.store.current_index + 1} / {len(items)}")
        self.image.set_item(item, prefer_full_image=True)
        self._load_other_boxes(item)
        rows = [
            ("result_id", str(getattr(item, "result_id", ""))),
            ("image", str(getattr(item, "image_rel_path", ""))),
            ("label", str(getattr(item, "suggested_label", getattr(item, "final_label", "")))),
            ("reliability", f"{getattr(item, 'reliability_score', 0):.4f}"),
            ("prediction", f"{getattr(item, 'pred_label', '')} {getattr(item, 'pred_prob', 0):.3f}".strip()),
            ("defer", ", ".join(getattr(item, "defer_reasons", ())) or "—"),
            ("reasons", ", ".join(getattr(item, "reasons", ())) or "—"),
        ]
        self.info.set_rows(rows)

    def _load_other_boxes(self, item: object) -> None:
        rel = str(getattr(item, "image_rel_path", "") or "")
        result_id = int(getattr(item, "result_id", 0) or 0)
        self._boxes_for = result_id
        if not rel:
            self.image.set_other_boxes([], result_id)
            return
        cached = self._boxes_cache.get(rel)
        if cached is not None:
            self.image.set_other_boxes(cached, result_id)
            return

        def on_done(boxes, rel=rel, result_id=result_id) -> None:
            self._boxes_cache[rel] = boxes
            # Only apply if the user is still on this same item.
            if self._boxes_for == result_id:
                self.image.set_other_boxes(boxes, result_id)

        self.controller.fetch_image_boxes(rel, on_done)

    def _label_current(self, label: str) -> None:
        item = self.store.current_item()
        if item is None:
            return
        if self.store.mode == "cleaned":
            self.controller.update_cleaned(item, label)
            self.controller.next_item()
        else:
            self.controller.decide_current(label)
            self.controller.next_item()

    def _commit(self) -> None:
        payload = (
            self.controller.commit_pending_corrections()
            if self.store.mode == "cleaned"
            else self.controller.commit_pending_decisions()
        )
        if payload.get("error"):
            QtWidgets.QMessageBox.warning(self, "Lưu", payload["error"])
        else:
            QtWidgets.QMessageBox.information(self, "Lưu", f"Đã lưu {payload.get('decisionCount', 0)} quyết định")

    def _render_commit(self) -> None:
        count = len(self.store.pending_corrections if self.store.mode == "cleaned" else self.store.pending_decisions)
        self.commit_btn.setText(f"Lưu ({count})")
        self.commit_btn.setEnabled(count > 0)
