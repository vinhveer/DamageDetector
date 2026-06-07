from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..config.defaults import DEFAULT_SETTINGS, LABELS
from ..services import db_service
from ..services.handoff_service import write_handoff_json
from ..services.image_service import ImageService
from ..services.settings_service import SettingsService
from .widgets.box_image import BoxImage
from .widgets.ui_kit import (
    Card,
    Chip,
    DecisionBar,
    InfoPanel,
    LABEL_BUTTON_STYLE,
    PercentBar,
    PickedList,
    Toolbar,
    danger_button,
    primary_button,
)


def _button(text: str, *, primary: bool = False, danger: bool = False) -> QtWidgets.QPushButton:
    if primary:
        return primary_button(text)
    if danger:
        return danger_button(text)
    b = QtWidgets.QPushButton(text)
    b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    b.setMinimumHeight(32)
    return b


class PayloadListModel(QtCore.QAbstractListModel):
    PayloadRole = QtCore.Qt.ItemDataRole.UserRole + 1
    ThumbnailKeyRole = QtCore.Qt.ItemDataRole.UserRole + 2

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._payloads: list[Any] = []
        self._title_fn = None
        self._thumb_key_fn = None
        self._thumbs: dict[object, QtGui.QPixmap] = {}

    def set_payloads(self, payloads: list[Any], title_fn, thumb_key_fn=None) -> None:
        self.beginResetModel()
        self._payloads = list(payloads)
        self._title_fn = title_fn
        self._thumb_key_fn = thumb_key_fn
        self.endResetModel()

    def thumb_key_at(self, row: int) -> object | None:
        payload = self.payload_at(row)
        if payload is None or self._thumb_key_fn is None:
            return None
        return self._thumb_key_fn(payload)

    def set_thumbnail(self, key: object, pixmap: QtGui.QPixmap) -> None:
        if key is None or pixmap.isNull():
            return
        self._thumbs[key] = pixmap
        for row, _payload in enumerate(self._payloads):
            if self.thumb_key_at(row) == key:
                idx = self.index(row, 0)
                self.dataChanged.emit(idx, idx, [QtCore.Qt.ItemDataRole.DecorationRole])
                break

    def payload_at(self, row: int) -> Any | None:
        if 0 <= row < len(self._payloads):
            return self._payloads[row]
        return None

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._payloads)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        payload = self.payload_at(index.row())
        if payload is None:
            return None
        if role == self.PayloadRole:
            return payload
        if role == self.ThumbnailKeyRole:
            return self.thumb_key_at(index.row())
        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            key = self.thumb_key_at(index.row())
            return self._thumbs.get(key) if key is not None else None
        if role == QtCore.Qt.ItemDataRole.SizeHintRole:
            return QtCore.QSize(280, 104)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if self._title_fn is None:
                return str(payload)
            return self._title_fn(payload)
        if role == QtCore.Qt.ItemDataRole.FontRole and self._title_fn is not None:
            title = str(self._title_fn(payload))
            if "PICK ->" in title or "EXCLUDED" in title:
                font = QtGui.QFont()
                font.setBold(True)
                return font
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and self._title_fn is not None:
            title = str(self._title_fn(payload))
            if "EXCLUDED" in title:
                return QtGui.QBrush(QtGui.QColor("#b00020"))
            if "PICK ->" in title:
                return QtGui.QBrush(QtGui.QColor("#0057b8"))
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return str(getattr(payload, "image_rel_path", "") or getattr(payload, "result_id", ""))
        return None


class PayloadList(QtWidgets.QListView):
    currentPayloadChanged = QtCore.Signal(object)
    visibleRowsChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = PayloadListModel(self)
        self.setModel(self._model)
        self.setAlternatingRowColors(True)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setIconSize(QtCore.QSize(96, 96))
        self.setSpacing(2)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.selectionModel().currentChanged.connect(self._emit_current)
        self.verticalScrollBar().valueChanged.connect(self._emit_visible_rows_later)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._emit_visible_rows_later()

    def set_payloads(self, payloads: list[Any], title_fn, thumb_key_fn=None) -> None:
        self._model.set_payloads(payloads, title_fn, thumb_key_fn)
        if self.count() > 0:
            self.setCurrentRow(0)
        self._emit_visible_rows_later()

    def set_thumbnail(self, key: object, pixmap: QtGui.QPixmap) -> None:
        self._model.set_thumbnail(key, pixmap)

    def visible_rows(self, preload: int = 18) -> list[int]:
        total = self.count()
        if total <= 0:
            return []
        first_index = self.indexAt(QtCore.QPoint(2, 2))
        last_index = self.indexAt(QtCore.QPoint(2, max(2, self.viewport().height() - 2)))
        first = first_index.row() if first_index.isValid() else max(0, self.currentRow())
        last = last_index.row() if last_index.isValid() else min(total - 1, first + preload)
        first = max(0, first - 4)
        last = min(total - 1, last + preload)
        return list(range(first, last + 1))

    def current_payload(self) -> Any | None:
        return self._model.payload_at(self.currentRow())

    def currentRow(self) -> int:  # noqa: N802 - keep QListWidget-compatible API
        idx = self.currentIndex()
        return idx.row() if idx.isValid() else -1

    def setCurrentRow(self, row: int) -> None:  # noqa: N802 - keep QListWidget-compatible API
        idx = self._model.index(int(row), 0)
        if idx.isValid():
            self.setCurrentIndex(idx)
            self.scrollTo(idx, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
            self._emit_visible_rows_later()

    def count(self) -> int:
        return self._model.rowCount()

    def _emit_visible_rows_later(self) -> None:
        QtCore.QTimer.singleShot(0, lambda: self.visibleRowsChanged.emit(self.visible_rows()))

    @QtCore.Slot(object, object)
    def _emit_current(self, current: QtCore.QModelIndex, _previous: QtCore.QModelIndex) -> None:
        payload = self._model.payload_at(current.row()) if current.isValid() else None
        if payload is not None:
            self.currentPayloadChanged.emit(payload)


class PayloadGrid(PayloadList):
    """Thumbnail-first grid for prototype triage."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.setMovement(QtWidgets.QListView.Movement.Static)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.setWrapping(True)
        self.setWordWrap(True)
        self.setUniformItemSizes(True)
        self.setIconSize(QtCore.QSize(148, 148))
        self.setGridSize(QtCore.QSize(190, 210))
        self.setSpacing(8)
        self.setSelectionRectVisible(False)

    def visible_rows(self, preload: int = 60) -> list[int]:
        total = self.count()
        if total <= 0:
            return []
        indexes: set[int] = set()
        step_x = max(40, self.gridSize().width() // 2)
        step_y = max(40, self.gridSize().height() // 2)
        for y in range(2, max(3, self.viewport().height()), step_y):
            for x in range(2, max(3, self.viewport().width()), step_x):
                idx = self.indexAt(QtCore.QPoint(x, y))
                if idx.isValid():
                    indexes.add(idx.row())
        if not indexes:
            current = self.currentRow()
            start = max(0, current if current >= 0 else 0)
            return list(range(start, min(total, start + preload)))
        first = max(0, min(indexes) - 12)
        last = min(total - 1, max(indexes) + preload)
        return list(range(first, last + 1))


class ReviewPage(QtWidgets.QWidget):
    def __init__(self, window: "MainWindow", *, cleaned: bool = False) -> None:
        super().__init__(window)
        self.window = window
        self.cleaned = cleaned
        self.items: list[Any] = []
        self.pending: dict[int, dict[str, Any]] = {}
        self._boxes: dict[str, list[dict[str, Any]]] = {}
        self._image_service = window.image_service
        self._current_image_key: tuple[str, str] | None = None
        self._loaded_once = False
        self._show_context_boxes = not cleaned
        self._thumb_size = 96 if not cleaned else 88
        self._full_prefetch_count = 2 if not cleaned else 0
        self._decision_history: list[tuple[int, dict[str, Any] | None]] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = Toolbar(self)
        self.filter = QtWidgets.QComboBox(self)
        self.filter.addItems(["all", *LABELS] if cleaned else ["all"])
        self.limit = QtWidgets.QSpinBox(self)
        self.limit.setRange(50, 5000)
        self.limit.setValue(1000 if cleaned else 500)
        self.progress_chip = Chip("0 / 0", self)
        self.top_progress = PercentBar(0, "0 / 0", self)
        self.top_progress.setMinimumWidth(180)
        top.add_label("Filter")
        top.add(self.filter)
        top.add_label("Limit")
        top.add(self.limit)
        top.add_stretch()
        top.add(self.top_progress)
        top.add(self.progress_chip)
        root.addWidget(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.list = PayloadList(split)
        center = QtWidgets.QWidget(split)
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)
        self.image = BoxImage(center)
        decision_actions = [
            (label, LABEL_BUTTON_STYLE.get(label, ("#7f8c8d", label.title()))[1], LABEL_BUTTON_STYLE.get(label, ("#7f8c8d", ""))[0])
            for label in LABELS
        ]
        self.decision_bar = DecisionBar(decision_actions, center)
        center_layout.addWidget(self.image, 1)
        center_layout.addWidget(self.decision_bar, 0)

        detail = QtWidgets.QWidget(split)
        detail_layout = QtWidgets.QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(10)
        item_card = Card("Item Info", detail)
        self.meta = InfoPanel(item_card)
        item_card.add(self.meta)
        decision_card = Card("Decision", detail)
        self.decision_info = InfoPanel(decision_card)
        decision_card.add(self.decision_info)
        progress_card = Card("Progress", detail)
        self.side_progress = PercentBar(0, "0 / 0", progress_card)
        progress_card.add(self.side_progress)
        pending_card = Card("Pending", detail)
        self.pending_count = Chip("0 pending", pending_card)
        self.pending_list = PickedList(pending_card)
        self.pending_list.setMaximumHeight(160)
        self.save_btn = _button("Write JSON (0)", primary=True)
        pending_card.add(self.pending_count)
        pending_card.add(self.pending_list)
        pending_card.add(self.save_btn)
        detail_layout.addWidget(item_card)
        detail_layout.addWidget(decision_card)
        detail_layout.addWidget(progress_card)
        detail_layout.addWidget(pending_card)
        detail_layout.addStretch(1)
        split.addWidget(self.list)
        split.addWidget(center)
        split.addWidget(detail)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 2)
        root.addWidget(split, 1)

        self.save_btn.clicked.connect(self.write_json)
        self.pending_list.removeRequested.connect(self.remove_pending)
        self.decision_bar.decided.connect(self.decide)
        self.image.clicked.connect(self.next_item)
        self.filter.currentTextChanged.connect(lambda _value: self.load())
        self.limit.valueChanged.connect(lambda _value: self.load())
        self.list.currentPayloadChanged.connect(self.show_item)
        self.list.visibleRowsChanged.connect(self.load_visible_thumbnails)
        self._image_service.imageLoaded.connect(self.on_image_loaded)
        self._image_service.imageFailed.connect(self.on_image_failed)
        self._install_shortcuts()

    def ensure_loaded(self) -> None:
        if not self._loaded_once:
            self.load()

    @QtCore.Slot()
    def load(self) -> None:
        try:
            if self.cleaned:
                label = self.filter.currentText()
                payload = db_service.list_cleaned(
                    self.window.db_path(), self.window.run_id(), self.window.image_root(),
                    final_label="" if label == "all" else label,
                    limit=self.limit.value(),
                )
                self.items = list(payload.get("items") or [])
            else:
                payload = db_service.list_queue(
                    self.window.db_path(), self.window.run_id(), self.window.image_root(),
                    queue_type="", sample_ratio=0.0, limit=self.limit.value(),
                )
                self.items = list(payload.get("items") or [])
            self.list.set_payloads(self.items, self.title, self.thumb_key)
            self.load_visible_thumbnails(self.list.visible_rows())
            self._loaded_once = True
            self.render_pending(refresh_list=False)
            self.window.status(f"Loaded {len(self.items)} items")
        except Exception as exc:
            self.window.error(str(exc))

    def title(self, item: Any) -> str:
        rid = int(getattr(item, "result_id", 0))
        label = str(getattr(item, "final_label", getattr(item, "suggested_label", "")) or "")
        score = float(getattr(item, "reliability_score", 0) or 0)
        if rid in self.pending:
            picked = str(self.pending[rid].get("newLabel") or "")
            return f"#{rid}  {label}  {score:.3f}   PICK -> {picked}"
        return f"#{rid}  {label}  {score:.3f}"

    def thumb_key(self, item: Any) -> tuple[str, int]:
        return ("thumb", int(getattr(item, "result_id", 0) or 0))

    def _item_image_path(self, item: Any) -> str:
        return BoxImage.item_path(item, prefer_full_image=True)

    @QtCore.Slot(object)
    def load_visible_thumbnails(self, rows: object) -> None:
        for row in list(rows or []):
            if not isinstance(row, int) or not (0 <= row < len(self.items)):
                continue
            item = self.items[row]
            key = self.thumb_key(item)
            cached = self._image_service.cached(key)
            if cached is not None:
                self.list.set_thumbnail(key, cached)
                continue
            path = self._item_image_path(item)
            if path:
                self._image_service.load_item_thumbnail(key, path, getattr(item, "box", None), size=self._thumb_size)

    @QtCore.Slot(object)
    def show_item(self, item: Any) -> None:
        image_path = self._item_image_path(item)
        if not image_path:
            self._current_image_key = None
            self.image.clear()
            self.window.status(
                "Không tìm thấy ảnh. Kiểm tra Image folder phải trỏ tới thư mục chứa file gốc trong DB."
            )
            return
        key = ("full", image_path)
        self._current_image_key = key
        cached = self._image_service.cached(key)
        if cached is not None:
            self.image.set_loading_item(item, prefer_full_image=True)
            self.image.set_pixmap(cached)
        else:
            self.image.set_loading_item(item, prefer_full_image=True)
            self._image_service.load_image(key, image_path, size=0)
        rel = str(getattr(item, "image_rel_path", "") or "")
        rid = int(getattr(item, "result_id", 0) or 0)
        if rel and self._show_context_boxes:
            try:
                boxes = self._boxes.get(rel)
                if boxes is None:
                    boxes = db_service.list_image_boxes(self.window.db_path(), self.window.run_id(), rel)
                    self._boxes[rel] = boxes
                self.image.set_other_boxes(boxes, rid)
            except Exception:
                self.image.set_other_boxes([], rid)
        else:
            self.image.set_other_boxes([], rid)
        label = str(getattr(item, "final_label", getattr(item, "suggested_label", "")) or "")
        reasons = ", ".join(getattr(item, "reasons", ()) or ()) or "-"
        pending = self.pending.get(rid)
        pending_label = str(pending.get("newLabel") or "") if pending else ""
        decision_text = (
            f"Selected decision: {pending_label}"
            if pending_label
            else f"No manual decision yet. Suggested/current: {label}"
        )
        self.decision_bar.set_current(pending_label)
        self.decision_bar.set_caption(f"#{rid}  {label}  score {float(getattr(item, 'reliability_score', 0) or 0):.4f}")
        self.decision_info.set_rows([
            ("Selected", pending_label or "-"),
            ("Current", label or "-"),
            ("Status", decision_text),
        ])
        self.image.set_caption(f"#{rid}  {pending_label or label}")
        self.image.set_decision_indicator(pending_label or "")
        self.meta.set_rows([
            ("ID", str(rid)),
            ("Image", rel or "-"),
            ("Label", label or "-"),
            ("Score", f"{float(getattr(item, 'reliability_score', 0) or 0):.4f}"),
            ("Reasons", reasons),
            ("Pending", pending_label or "-"),
        ])
        self.prefetch_neighbors()

    @QtCore.Slot(object, object)
    def on_image_loaded(self, key: object, pixmap: QtGui.QPixmap) -> None:
        if isinstance(key, tuple) and key and key[0] == "thumb":
            self.list.set_thumbnail(key, pixmap)
            return
        if key == self._current_image_key:
            self.image.set_pixmap(pixmap)

    @QtCore.Slot(object, str)
    def on_image_failed(self, key: object, message: str) -> None:
        if key == self._current_image_key:
            self.window.status(f"Image load failed: {message}")

    def prefetch_neighbors(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        for offset in range(1, self._full_prefetch_count + 1):
            if row + offset >= len(self.items):
                break
            item = self.items[row + offset]
            path = self._item_image_path(item)
            if path:
                self._image_service.load_image(("full", path), path, size=0)

    def decide(self, label: str) -> None:
        item = self.list.current_payload()
        if item is None:
            return
        rid = int(getattr(item, "result_id"))
        prev = str(getattr(item, "final_label", getattr(item, "suggested_label", "")) or "")
        previous_pending = dict(self.pending[rid]) if rid in self.pending else None
        self._decision_history.append((rid, previous_pending))
        self.pending[rid] = {
            "resultId": rid,
            "action": "manual_reject" if label == "reject" else ("manual_accept" if label == prev else "manual_relabel"),
            "previousLabel": prev,
            "newLabel": label,
        }
        self.window.status(f"Picked #{rid}: {prev} -> {label}. Pending: {len(self.pending)}")
        self.render_pending()
        row = self.list.currentRow()
        if row + 1 < self.list.count():
            self.list.setCurrentRow(row + 1)
        else:
            self.show_item(item)

    def render_pending(self, refresh_list: bool = True) -> None:
        pending_count = len(self.pending)
        total = len(self.items)
        text = f"{pending_count} / {total}" if total else "0 / 0"
        fraction = pending_count / total if total else 0.0
        self.progress_chip.setText(text)
        self.top_progress.set_value(fraction, text)
        self.side_progress.set_value(fraction, text)
        self.pending_count.setText(f"{pending_count} pending")
        self.save_btn.setText(f"Write JSON ({pending_count})")
        entries: list[tuple[int, str]] = []
        for rid, pending in sorted(self.pending.items()):
            previous = str(pending.get("previousLabel") or "-")
            new = str(pending.get("newLabel") or "-")
            entries.append((rid, f"#{rid}  {previous} -> {new}"))
        self.pending_list.set_entries(entries)
        if not refresh_list:
            return
        current = self.list.currentRow()
        self.list.set_payloads(self.items, self.title, self.thumb_key)
        if 0 <= current < self.list.count():
            self.list.setCurrentRow(current)
        self.load_visible_thumbnails(self.list.visible_rows())

    @QtCore.Slot(int)
    def remove_pending(self, result_id: int) -> None:
        if result_id not in self.pending:
            return
        self._decision_history.append((result_id, dict(self.pending[result_id])))
        self.pending.pop(result_id, None)
        self.window.status(f"Removed pending decision #{result_id}. Pending: {len(self.pending)}")
        self.render_pending()
        current = self.list.current_payload()
        if current is not None:
            self.show_item(current)

    def undo_last(self) -> None:
        if not self._decision_history:
            self.window.status("Nothing to undo")
            return
        rid, previous = self._decision_history.pop()
        if previous is None:
            self.pending.pop(rid, None)
        else:
            self.pending[rid] = previous
        self.window.status(f"Undid decision #{rid}. Pending: {len(self.pending)}")
        self.render_pending()
        for row, item in enumerate(self.items):
            if int(getattr(item, "result_id", 0) or 0) == rid:
                self.list.setCurrentRow(row)
                break
        current = self.list.current_payload()
        if current is not None:
            self.show_item(current)

    def next_item(self) -> None:
        row = self.list.currentRow()
        if row + 1 < self.list.count():
            self.list.setCurrentRow(row + 1)

    def previous_item(self) -> None:
        row = self.list.currentRow()
        if row > 0:
            self.list.setCurrentRow(row - 1)

    def _install_shortcuts(self) -> None:
        for key, label in zip(("1", "2", "3", "4"), LABELS, strict=False):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda value=label: self.decide(value))
        for key in ("Space", "Down"):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self.next_item)
        previous = QtGui.QShortcut(QtGui.QKeySequence("Up"), self)
        previous.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        previous.activated.connect(self.previous_item)
        undo = QtGui.QShortcut(QtGui.QKeySequence("Z"), self)
        undo.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo.activated.connect(self.undo_last)
        save = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self)
        save.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        save.activated.connect(self.write_json)

    @QtCore.Slot()
    def write_json(self) -> None:
        if not self.pending:
            self.window.error("No pending decisions.")
            return
        key = "corrections" if self.cleaned else "decisions"
        payload = {
            "type": "review_request",
            "db": self.window.db_path(),
            "run_id": self.window.run_id(),
            "reviewer": self.window.reviewer(),
            "notes": self.window.notes(),
            key: list(self.pending.values()),
        }
        try:
            path = write_handoff_json(self.window.db_path(), payload, kind="review", run_id=self.window.run_id())
            self.pending.clear()
            self._decision_history.clear()
            self.render_pending()
            self.window.status(f"JSON written: {path}")
            QtWidgets.QMessageBox.information(self, "JSON written", str(path))
        except Exception as exc:
            self.window.error(str(exc))


class PrototypePage(QtWidgets.QWidget):
    """Step 5 prototype/domain representative review.

    The pipeline proposes representative candidates.  The reviewer assigns the
    final prototype label using only four actions: crack, mold, spall, reject.
    """

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.window = window
        self.items: list[Any] = []
        self.visible_items: list[Any] = []
        self.decisions: dict[int, dict[str, Any]] = {}
        self._image_service = window.image_service
        self._current_image_key: tuple[str, str] | None = None
        self._loaded_once = False
        self._thumb_size = 96
        self._prototype_policy = {
            "damage_total_per_label": 200,
            "reject_total": 300,
            "score_triplet_per_domain": 3,
            "score_triplet_bands": ["low", "mid", "high"],
            "score_field": "reliability_score",
            "fallback": "domain-first anchors, then balanced top-up to target count",
        }
        self._shortcuts: list[QtGui.QShortcut] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = Toolbar(self)
        self.label_filter = QtWidgets.QComboBox(self)
        self.label_filter.addItems(["all", *LABELS])
        self.summary = Chip("Reviewed: 0 / 0", self)
        self.save_btn = _button("Write JSON", primary=True)
        top.add_label("Show")
        top.add(self.label_filter)
        top.add_stretch()
        top.add(self.summary)
        top.add(self.save_btn)
        root.addWidget(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.list = PayloadList(split)

        right = QtWidgets.QWidget(split)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        self.image = BoxImage(right)
        right_layout.addWidget(self.image, 1)

        self.button_row = QtWidgets.QHBoxLayout()
        self.button_row.setContentsMargins(0, 0, 0, 0)
        self.button_row.setSpacing(8)
        for index, label in enumerate(LABELS, start=1):
            button = _button(f"{index}. {label}", danger=(label == "reject"))
            button.clicked.connect(lambda _checked=False, value=label: self.decide(value))
            self.button_row.addWidget(button)
        self.button_row.addStretch(1)
        right_layout.addLayout(self.button_row)

        info = QtWidgets.QWidget(split)
        info_layout = QtWidgets.QVBoxLayout(info)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(10)
        candidate_card = Card("Candidate", info)
        self.meta = InfoPanel(candidate_card)
        candidate_card.add(self.meta)
        progress_card = Card("Progress", info)
        self.progress = InfoPanel(progress_card)
        progress_card.add(self.progress)
        hint = QtWidgets.QLabel(
            "Step 5: review domain representatives. Each domain includes 3 score anchors: Low / Mid / High, "
            "then balanced Extra candidates are added until the target count is reached. "
            "Shortcuts: Enter = accept shown label, 1/2/3/4 = crack/mold/spall/reject, Space/Down = next, Up = previous, Ctrl+S = write JSON.",
            info,
        )
        hint.setWordWrap(True)
        info_layout.addWidget(candidate_card)
        info_layout.addWidget(progress_card)
        info_layout.addWidget(hint)
        info_layout.addStretch(1)

        split.addWidget(self.list)
        split.addWidget(right)
        split.addWidget(info)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 2)
        root.addWidget(split, 1)

        self.save_btn.clicked.connect(self.write_json)
        self.label_filter.currentTextChanged.connect(self.refresh)
        self.list.currentPayloadChanged.connect(self.show_item)
        self.list.visibleRowsChanged.connect(self.load_visible_thumbnails)
        self._image_service.imageLoaded.connect(self.on_image_loaded)
        self._image_service.imageFailed.connect(self.on_image_failed)
        self._install_shortcuts()

    def ensure_loaded(self) -> None:
        if not self._loaded_once:
            self.load()

    @QtCore.Slot()
    def load(self) -> None:
        try:
            payload = db_service.list_prototype_candidates(
                self.window.db_path(), self.window.run_id(), self.window.image_root(), reject_below=0.5, per_band=80
            )
            candidates = list(payload.get("items") or [])
            selected = self._representatives_by_policy(candidates)
            self.items = [item for label in ("crack", "mold", "spall", "reject") for item in selected.get(label, [])]
            self._loaded_once = True
            self.refresh()
            self.window.status(
                f"Loaded {len(self.items)} prototype representatives from {len(candidates)} candidates "
                "(low/mid/high anchors per domain + balanced top-up)"
            )
        except Exception as exc:
            self.window.error(str(exc))

    @QtCore.Slot(str)
    def refresh(self, _label: str = "", keep_result_id: int | None = None) -> None:
        label = self.label_filter.currentText()
        rows = self.items if label == "all" else [i for i in self.items if str(getattr(i, "label", "")) == label]
        self.visible_items = self._diverse_order(list(rows))
        self.list.set_payloads(self.visible_items, self.title, self.thumb_key)
        if keep_result_id is not None:
            for row, item in enumerate(self.visible_items):
                if int(getattr(item, "result_id", 0) or 0) == int(keep_result_id):
                    self.list.setCurrentRow(row)
                    break
        self.render_summary()
        self.load_visible_thumbnails(self.list.visible_rows())

    def _quality_key(self, item: Any) -> tuple[float, float, int]:
        centroid = getattr(item, "centroid_similarity", None)
        centroid_value = float(centroid) if centroid is not None else 0.0
        return (centroid_value, self._score_key(item), -int(getattr(item, "result_id", 0) or 0))

    def _score_key(self, item: Any) -> float:
        return float(getattr(item, "reliability_score", 0) or 0)

    def _score_band(self, item: Any) -> str:
        return str(getattr(item, "score_band", "") or "")

    def _set_score_band(self, item: Any, band: str) -> Any:
        try:
            object.__setattr__(item, "score_band", band)
        except Exception:
            try:
                setattr(item, "score_band", band)
            except Exception:
                pass
        return item

    def _append_unique(self, selected: list[Any], selected_ids: set[int], item: Any, band: str) -> bool:
        rid = int(getattr(item, "result_id", 0) or 0)
        if rid in selected_ids:
            existing = self._score_band(item)
            if band and band not in existing.split("+"):
                merged = "+".join(part for part in (existing, band) if part)
                self._set_score_band(item, merged)
            return False
        selected_ids.add(rid)
        selected.append(self._set_score_band(item, band))
        return True

    def _pick_score_triplet(self, rows: list[Any]) -> list[Any]:
        ordered = sorted(rows, key=lambda item: (self._score_key(item), int(getattr(item, "result_id", 0) or 0)))
        if not ordered:
            return []
        picks = [
            ("low", ordered[0]),
            ("mid", ordered[(len(ordered) - 1) // 2]),
            ("high", ordered[-1]),
        ]
        selected: list[Any] = []
        seen: set[int] = set()
        for band, item in picks:
            self._append_unique(selected, seen, item, band)
        return selected

    def _domain_key(self, item: Any) -> tuple[int, str]:
        idx = getattr(item, "domain_index", None)
        if idx is None:
            idx = 9999
        return (int(idx), str(getattr(item, "cluster_id", "") or "no_cluster"))

    def _select_score_triplets_by_domain(self, rows: list[Any], *, target_total: int) -> list[Any]:
        by_domain: dict[tuple[int, str], list[Any]] = defaultdict(list)
        for candidate in rows:
            by_domain[self._domain_key(candidate)].append(candidate)
        domain_keys = sorted(
            by_domain,
            key=lambda key: (
                key[0],
                -max(int(getattr(i, "cluster_size", 0) or 0) for i in by_domain[key]),
                key[1],
            ),
        )
        selected: list[Any] = []
        selected_ids: set[int] = set()
        for key in domain_keys:
            for item in self._pick_score_triplet(by_domain[key]):
                self._append_unique(selected, selected_ids, item, self._score_band(item) or "anchor")
        if len(selected) >= target_total:
            return selected

        domain_counts: dict[tuple[int, str], int] = defaultdict(int)
        image_counts: dict[str, int] = defaultdict(int)
        for item in selected:
            domain_counts[self._domain_key(item)] += 1
            image_counts[str(getattr(item, "image_rel_path", "") or "")] += 1

        topup_candidates = sorted(
            rows,
            key=lambda item: (
                domain_counts[self._domain_key(item)],
                image_counts[str(getattr(item, "image_rel_path", "") or "")],
                -self._quality_key(item)[0],
                -self._score_key(item),
                int(getattr(item, "result_id", 0) or 0),
            ),
        )
        while len(selected) < target_total:
            picked_any = False
            topup_candidates.sort(
                key=lambda item: (
                    domain_counts[self._domain_key(item)],
                    image_counts[str(getattr(item, "image_rel_path", "") or "")],
                    -self._quality_key(item)[0],
                    -self._score_key(item),
                    int(getattr(item, "result_id", 0) or 0),
                )
            )
            for item in topup_candidates:
                rid = int(getattr(item, "result_id", 0) or 0)
                if rid in selected_ids:
                    continue
                if self._append_unique(selected, selected_ids, item, "extra"):
                    domain_counts[self._domain_key(item)] += 1
                    image_counts[str(getattr(item, "image_rel_path", "") or "")] += 1
                    picked_any = True
                    break
            if not picked_any:
                break
        return selected

    def _representatives_by_policy(self, candidates: list[Any]) -> dict[str, list[Any]]:
        by_label: dict[str, list[Any]] = defaultdict(list)
        for item in candidates:
            by_label[str(getattr(item, "label", "") or "")].append(item)
        selected: dict[str, list[Any]] = {}
        for label in ("crack", "mold", "spall"):
            selected[label] = self._select_score_triplets_by_domain(
                by_label.get(label, []),
                target_total=int(self._prototype_policy["damage_total_per_label"]),
            )
        selected["reject"] = self._select_score_triplets_by_domain(
            by_label.get("reject", []),
            target_total=int(self._prototype_policy["reject_total"]),
        )
        return selected

    def _image_diverse_order(self, rows: list[Any]) -> list[Any]:
        by_image: dict[str, list[Any]] = defaultdict(list)
        for item in rows:
            by_image[str(getattr(item, "image_rel_path", "") or "")].append(item)
        buckets: list[deque[Any]] = []
        for _image, items in sorted(by_image.items(), key=lambda pair: max(self._quality_key(i) for i in pair[1]), reverse=True):
            items.sort(key=self._quality_key, reverse=True)
            buckets.append(deque(items))
        ordered: list[Any] = []
        while buckets:
            next_buckets: list[deque[Any]] = []
            for bucket in buckets:
                if bucket:
                    ordered.append(bucket.popleft())
                if bucket:
                    next_buckets.append(bucket)
            buckets = next_buckets
        return ordered

    def _diverse_order(self, rows: list[Any]) -> list[Any]:
        by_domain: dict[tuple[int, str], list[Any]] = defaultdict(list)
        for item in rows:
            by_domain[self._domain_key(item)].append(item)
        domain_keys = sorted(by_domain, key=lambda key: (key[0], -max(int(getattr(i, "cluster_size", 0) or 0) for i in by_domain[key]), key[1]))
        buckets: dict[tuple[int, str], deque[Any]] = {key: deque(self._image_diverse_order(by_domain[key])) for key in domain_keys}
        ordered: list[Any] = []
        last_image = ""
        while buckets:
            progressed = False
            for key in list(domain_keys):
                bucket = buckets.get(key)
                if not bucket:
                    buckets.pop(key, None)
                    continue
                rotations = len(bucket)
                while rotations > 1 and str(getattr(bucket[0], "image_rel_path", "") or "") == last_image:
                    bucket.rotate(-1)
                    rotations -= 1
                item = bucket.popleft()
                ordered.append(item)
                last_image = str(getattr(item, "image_rel_path", "") or "")
                progressed = True
                if not bucket:
                    buckets.pop(key, None)
            if not progressed:
                break
        return ordered

    def title(self, item: Any) -> str:
        rid = int(getattr(item, "result_id", 0))
        original = str(getattr(item, "label", "") or "")
        chosen = str(self.decisions.get(rid, {}).get("label") or "")
        domain = getattr(item, "domain_index", None)
        domain_text = "D?" if domain is None else f"D{int(domain)}"
        score = float(getattr(item, "reliability_score", 0) or 0)
        band = self._score_band(item).upper() or "SCORE"
        state = f"  -> {chosen}" if chosen else ""
        return f"#{rid}  {original}  {domain_text}  {band}  score {score:.3f}{state}"

    def thumb_key(self, item: Any) -> tuple[str, int]:
        return ("thumb", int(getattr(item, "result_id", 0) or 0))

    def _item_image_path(self, item: Any) -> str:
        return BoxImage.item_path(item, prefer_full_image=True)

    @QtCore.Slot(object)
    def load_visible_thumbnails(self, rows: object) -> None:
        for row in list(rows or []):
            if not isinstance(row, int) or not (0 <= row < len(self.visible_items)):
                continue
            item = self.visible_items[row]
            key = self.thumb_key(item)
            cached = self._image_service.cached(key)
            if cached is not None:
                self.list.set_thumbnail(key, cached)
                continue
            path = self._item_image_path(item)
            if path:
                self._image_service.load_item_thumbnail(key, path, getattr(item, "box", None), size=self._thumb_size)

    @QtCore.Slot(object)
    def show_item(self, item: Any) -> None:
        image_path = self._item_image_path(item)
        if not image_path:
            self._current_image_key = None
            self.image.clear()
            self.window.status("Không tìm thấy ảnh. Kiểm tra Image folder phải trỏ tới thư mục chứa file gốc trong DB.")
            return
        key = ("full", image_path)
        self._current_image_key = key
        cached = self._image_service.cached(key)
        self.image.set_loading_item(item, prefer_full_image=True)
        if cached is not None:
            self.image.set_pixmap(cached)
        else:
            self._image_service.load_image(key, image_path, size=0)
        rid = int(getattr(item, "result_id", 0))
        original = str(getattr(item, "label", "") or "")
        chosen = str(self.decisions.get(rid, {}).get("label") or "")
        domain = getattr(item, "domain_index", None)
        domain_text = "?" if domain is None else str(int(domain))
        state = chosen or "unreviewed"
        band = self._score_band(item).upper() or "SCORE"
        self.image.set_caption(f"#{rid}  {original}  {band}")
        self.image.set_decision_indicator(state)
        self.meta.set_rows([
            ("ID", str(rid)),
            ("Original", original or "-"),
            ("Chosen", chosen or "-"),
            ("Score anchor", band),
            ("Domain", domain_text),
            ("Cluster", str(getattr(item, "cluster_id", "") or "-")),
            ("Cluster size", str(int(getattr(item, "cluster_size", 0) or 0))),
            ("Score", f"{float(getattr(item, 'reliability_score', 0) or 0):.4f}"),
            ("Centroid sim", str(getattr(item, "centroid_similarity", None))),
            ("Image", str(getattr(item, "image_rel_path", "") or "-")),
        ])
        self.prefetch_neighbors()

    @QtCore.Slot(object, object)
    def on_image_loaded(self, key: object, pixmap: QtGui.QPixmap) -> None:
        if isinstance(key, tuple) and key and key[0] == "thumb":
            self.list.set_thumbnail(key, pixmap)
            return
        if key == self._current_image_key:
            self.image.set_pixmap(pixmap)

    @QtCore.Slot(object, str)
    def on_image_failed(self, key: object, message: str) -> None:
        if key == self._current_image_key:
            self.window.status(f"Image load failed: {message}")

    def prefetch_neighbors(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        for offset in range(1, 3):
            if row + offset >= len(self.visible_items):
                break
            item = self.visible_items[row + offset]
            path = self._item_image_path(item)
            if path:
                self._image_service.load_image(("full", path), path, size=0)

    def decide(self, label: str) -> None:
        item = self.list.current_payload()
        if item is None:
            return
        rid = int(getattr(item, "result_id", 0) or 0)
        original = str(getattr(item, "label", "") or "")
        payload = self._candidate_payload(item, label=label)
        payload["previousLabel"] = original
        payload["action"] = "prototype_reject" if label == "reject" else ("prototype_accept" if label == original else "prototype_relabel")
        self.decisions[rid] = payload
        self.window.status(f"Prototype #{rid}: {original} -> {label}. Reviewed {len(self.decisions)}/{len(self.items)}")
        current_row = self.list.currentRow()
        self.render_summary(refresh_list=True, keep_row=current_row)
        if current_row + 1 < self.list.count():
            self.list.setCurrentRow(current_row + 1)
        else:
            self.show_item(item)

    def render_summary(self, *, refresh_list: bool = False, keep_row: int | None = None) -> None:
        reviewed = len(self.decisions)
        total = len(self.items)
        by_label: dict[str, int] = defaultdict(int)
        for decision in self.decisions.values():
            by_label[str(decision.get("label") or "")] += 1
        counts = "  ".join(f"{label}:{by_label.get(label, 0)}" for label in LABELS)
        self.summary.setText(f"Reviewed: {reviewed} / {total}    {counts}")
        self.progress.set_rows([
            ("Reviewed", f"{reviewed} / {total}"),
            ("crack", str(by_label.get("crack", 0))),
            ("mold", str(by_label.get("mold", 0))),
            ("spall", str(by_label.get("spall", 0))),
            ("reject", str(by_label.get("reject", 0))),
        ])
        if refresh_list:
            row = self.list.currentRow() if keep_row is None else int(keep_row)
            self.list.set_payloads(self.visible_items, self.title, self.thumb_key)
            if 0 <= row < self.list.count():
                self.list.setCurrentRow(row)
            self.load_visible_thumbnails(self.list.visible_rows())

    def _candidate_payload(self, item: Any, *, label: str) -> dict[str, Any]:
        domain = getattr(item, "domain_index", None)
        centroid = getattr(item, "centroid_similarity", None)
        return {
            "resultId": int(getattr(item, "result_id", 0) or 0),
            "label": str(label or ""),
            "isReject": str(label or "") == "reject",
            "domainIndex": None if domain is None else int(domain),
            "clusterId": str(getattr(item, "cluster_id", "") or ""),
            "clusterSize": int(getattr(item, "cluster_size", 0) or 0),
            "centroidSimilarity": None if centroid is None else float(centroid),
            "imageRelPath": str(getattr(item, "image_rel_path", "") or ""),
            "candidateLabel": str(getattr(item, "label", "") or ""),
            "predictedLabel": str(getattr(item, "predicted_label", "") or ""),
            "scoreForAuditOnly": float(getattr(item, "reliability_score", 0) or 0),
            "scoreAnchor": self._score_band(item) or "",
        }

    def _install_shortcuts(self) -> None:
        def add_shortcut(key: str | QtGui.QKeySequence.StandardKey, callback) -> None:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

        for key, label in zip(("1", "2", "3", "4"), LABELS, strict=False):
            add_shortcut(key, lambda value=label: self.decide(value))
        add_shortcut("Enter", self.accept_current_label)
        add_shortcut("Return", self.accept_current_label)
        add_shortcut("Space", self.next_item)
        add_shortcut("Down", self.next_item)
        add_shortcut("Up", self.previous_item)
        add_shortcut(QtGui.QKeySequence.StandardKey.Save, self.write_json)

    def accept_current_label(self) -> None:
        item = self.list.current_payload()
        if item is None:
            return
        label = str(getattr(item, "label", "") or "")
        if label not in LABELS:
            return
        self.decide(label)

    def next_item(self) -> None:
        row = self.list.currentRow()
        if row + 1 < self.list.count():
            self.list.setCurrentRow(row + 1)

    def previous_item(self) -> None:
        row = self.list.currentRow()
        if row > 0:
            self.list.setCurrentRow(row - 1)

    @QtCore.Slot()
    def write_json(self) -> None:
        if not self.decisions:
            self.window.error("No prototype decisions yet.")
            return
        prototypes = [p for p in self.decisions.values() if not p.get("isReject") and p.get("label") != "reject"]
        rejects = [p for p in self.decisions.values() if p.get("isReject") or p.get("label") == "reject"]
        unreviewed = [
            int(getattr(item, "result_id", 0) or 0)
            for item in self.items
            if int(getattr(item, "result_id", 0) or 0) not in self.decisions
        ]
        payload = {
            "type": "prototype_request",
            "selection_mode": "representative_relabel",
            "selection_policy": dict(self._prototype_policy),
            "selection_summary": {
                "representative_count": len(self.items),
                "reviewed_count": len(self.decisions),
                "unreviewed_count": len(unreviewed),
                "selected_by_label": {label: sum(1 for p in prototypes if p.get("label") == label) for label in ("crack", "mold", "spall")},
                "reject_count": len(rejects),
            },
            "db": self.window.db_path(),
            "run_id": self.window.run_id(),
            "model_name": self.window.model_name(),
            "view_name": "tight",
            "notes": self.window.notes(),
            "prototypes": prototypes,
            "rejects": rejects,
            "unreviewed": unreviewed,
            "run_seed": True,
            "run_policy": True,
        }
        try:
            path = write_handoff_json(self.window.db_path(), payload, kind="prototype", run_id=self.window.run_id())
            self.window.status(f"JSON written: {path}")
            QtWidgets.QMessageBox.information(self, "JSON written", str(path))
        except Exception as exc:
            self.window.error(str(exc))


class ImageOverviewPage(QtWidgets.QWidget):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.window = window
        self.items: list[dict[str, Any]] = []
        self._image_service = window.image_service
        self._current_image_key: tuple[str, str] | None = None
        self._loaded_once = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = Toolbar(self)
        self.source = QtWidgets.QComboBox(self)
        self.source.addItems(["all", "cleaned", "review"])
        self.limit = QtWidgets.QSpinBox(self)
        self.limit.setRange(20, 5000)
        self.limit.setValue(1000)
        self.summary = Chip("Images: -", self)
        top.add_label("Show")
        top.add(self.source)
        top.add_label("Limit")
        top.add(self.limit)
        top.add_stretch()
        top.add(self.summary)
        root.addWidget(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.list = PayloadList(split)
        self.image = BoxImage(split)
        detail = QtWidgets.QWidget(split)
        detail_layout = QtWidgets.QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(10)
        image_card = Card("Image", detail)
        self.meta = InfoPanel(image_card)
        image_card.add(self.meta)
        legend_card = Card("Legend", detail)
        self.legend = InfoPanel(legend_card)
        self.legend.set_rows([
            ("Crack", "blue"),
            ("Mold", "green"),
            ("Spall", "orange"),
            ("Reject", "gray"),
        ])
        legend_card.add(self.legend)
        detail_layout.addWidget(image_card)
        detail_layout.addWidget(legend_card)
        detail_layout.addStretch(1)
        split.addWidget(self.list)
        split.addWidget(self.image)
        split.addWidget(detail)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 2)
        root.addWidget(split, 1)

        self.source.currentTextChanged.connect(lambda _value: self.load())
        self.limit.valueChanged.connect(lambda _value: self.load())
        self.list.currentPayloadChanged.connect(self.show_item)
        self._image_service.imageLoaded.connect(self.on_image_loaded)
        self._image_service.imageFailed.connect(self.on_image_failed)

    def ensure_loaded(self) -> None:
        if not self._loaded_once:
            self.load()

    @QtCore.Slot()
    def load(self) -> None:
        try:
            payload = db_service.list_image_overview(
                self.window.db_path(),
                self.window.run_id(),
                self.window.image_root(),
                source=self.source.currentText(),
                limit=self.limit.value(),
            )
            self.items = list(payload.get("items") or [])
            self.list.set_payloads(self.items, self.title, self.thumb_key)
            self._loaded_once = True
            self.summary.setText(f"Images: {len(self.items)} / {int(payload.get('total') or 0)}")
            self.window.status(f"Loaded {len(self.items)} images")
        except Exception as exc:
            self.window.error(str(exc))

    def title(self, item: Any) -> str:
        rel = str(item.get("image_rel_path") if isinstance(item, dict) else "")
        box_count = int(item.get("box_count", 0) if isinstance(item, dict) else 0)
        cleaned = int(item.get("cleaned_count", 0) if isinstance(item, dict) else 0)
        review = int(item.get("review_count", 0) if isinstance(item, dict) else 0)
        return f"{rel}\nboxes={box_count}  cleaned={cleaned}  review={review}"

    def thumb_key(self, item: Any) -> tuple[str, str]:
        rel = str(item.get("image_rel_path") if isinstance(item, dict) else "")
        return ("image_overview_thumb", rel)

    @QtCore.Slot(object)
    def show_item(self, item: Any) -> None:
        if not isinstance(item, dict):
            return
        image_path = str(item.get("image_path") or "")
        rel = str(item.get("image_rel_path") or "")
        boxes = list(item.get("boxes") or [])
        caption = f"{rel}  boxes={len(boxes)}"
        if not image_path:
            self._current_image_key = None
            self.image.clear()
            self.window.status("Không tìm thấy ảnh. Kiểm tra Image folder phải trỏ tới thư mục ảnh gốc.")
            return
        key = ("image_overview_full", image_path)
        self._current_image_key = key
        self.image.set_overlay_loading(boxes=boxes, caption=caption)
        cached = self._image_service.cached(key)
        if cached is not None:
            self.image.set_pixmap(cached)
            self.image.set_overlay_boxes(boxes=boxes, caption=caption)
        else:
            self._image_service.load_image(key, image_path, size=0)
        by_label: dict[str, int] = defaultdict(int)
        by_source: dict[str, int] = defaultdict(int)
        for box in boxes:
            by_label[str(box.get("label") or "")] += 1
            by_source[str(box.get("source") or "")] += 1
        label_text = ", ".join(f"{k}:{v}" for k, v in sorted(by_label.items()) if k) or "-"
        source_text = ", ".join(f"{k}:{v}" for k, v in sorted(by_source.items()) if k) or "-"
        self.meta.set_rows([
            ("Image", rel or "-"),
            ("Path", image_path or "-"),
            ("Boxes", str(len(boxes))),
            ("By label", label_text),
            ("By source", source_text),
        ])

    @QtCore.Slot(object, object)
    def on_image_loaded(self, key: object, pixmap: QtGui.QPixmap) -> None:
        if key == self._current_image_key:
            item = self.list.current_payload()
            boxes = list(item.get("boxes") or []) if isinstance(item, dict) else []
            rel = str(item.get("image_rel_path") or "") if isinstance(item, dict) else ""
            self.image.set_pixmap(pixmap)
            self.image.set_overlay_boxes(boxes=boxes, caption=f"{rel}  boxes={len(boxes)}")

    @QtCore.Slot(object, str)
    def on_image_failed(self, key: object, message: str) -> None:
        if key == self._current_image_key:
            self.window.status(f"Image load failed: {message}")


class ConnectDialog(QtWidgets.QDialog):
    def __init__(self, settings: dict[str, Any], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect semi-labeling data")
        self.setModal(True)
        self.resize(760, 260)
        root = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Connect DB and image folder", self)
        subtitle = QtWidgets.QLabel("The app is read-only: it opens pipeline.sqlite3, reads images, and writes JSON handoff files.", self)
        root.addWidget(title)
        root.addWidget(subtitle)

        form = QtWidgets.QGridLayout()
        self.db = QtWidgets.QLineEdit(str(settings.get("db_path") or ""), self)
        self.images = QtWidgets.QLineEdit(str(settings.get("image_root") or ""), self)
        self.run = QtWidgets.QLineEdit(str(settings.get("run_id") or "myrun"), self)
        self.reviewer = QtWidgets.QLineEdit(str(settings.get("reviewer") or ""), self)
        self.notes = QtWidgets.QLineEdit(str(settings.get("notes") or ""), self)
        self.db.setPlaceholderText(".../pipeline.sqlite3")
        self.images.setPlaceholderText("source image folder")
        self.run.setPlaceholderText("myrun")
        self.reviewer.setPlaceholderText("optional")
        self.notes.setPlaceholderText("optional")
        db_btn = _button("Browse...")
        img_btn = _button("Browse...")
        runs_btn = _button("Load runs")
        form.addWidget(QtWidgets.QLabel("SQLite DB"), 0, 0)
        form.addWidget(self.db, 0, 1)
        form.addWidget(db_btn, 0, 2)
        form.addWidget(QtWidgets.QLabel("Image folder"), 1, 0)
        form.addWidget(self.images, 1, 1)
        form.addWidget(img_btn, 1, 2)
        form.addWidget(QtWidgets.QLabel("Run ID"), 2, 0)
        form.addWidget(self.run, 2, 1)
        form.addWidget(runs_btn, 2, 2)
        form.addWidget(QtWidgets.QLabel("Reviewer"), 3, 0)
        form.addWidget(self.reviewer, 3, 1, 1, 2)
        form.addWidget(QtWidgets.QLabel("Notes"), 4, 0)
        form.addWidget(self.notes, 4, 1, 1, 2)
        root.addLayout(form)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        cancel = _button("Cancel")
        connect = _button("Connect", primary=True)
        connect.setDefault(True)
        actions.addWidget(cancel)
        actions.addWidget(connect)
        root.addLayout(actions)

        db_btn.clicked.connect(self.browse_db)
        img_btn.clicked.connect(self.browse_images)
        runs_btn.clicked.connect(self.load_runs)
        cancel.clicked.connect(self.reject)
        connect.clicked.connect(self.try_accept)

    def payload(self) -> dict[str, Any]:
        return {
            "db_path": self.db.text().strip(),
            "image_root": self.images.text().strip(),
            "run_id": self.run.text().strip() or "myrun",
            "reviewer": self.reviewer.text().strip(),
            "notes": self.notes.text().strip(),
        }

    @QtCore.Slot()
    def browse_db(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select pipeline.sqlite3", self.db.text().strip(), "SQLite (*.sqlite3 *.db);;All files (*)")
        if path:
            self.db.setText(path)

    @QtCore.Slot()
    def browse_images(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select image folder", self.images.text().strip())
        if path:
            self.images.setText(path)

    @QtCore.Slot()
    def load_runs(self) -> None:
        try:
            runs = db_service.list_runs(self.db.text().strip()).get("runs") or []
            if not runs:
                QtWidgets.QMessageBox.information(self, "Runs", "No runs found in this DB.")
                return
            labels = [str(row.get("run_id") or "") for row in runs if row.get("run_id")]
            selected, ok = QtWidgets.QInputDialog.getItem(self, "Select run", "Run ID", labels, 0, False)
            if ok and selected:
                self.run.setText(selected)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot load runs", str(exc))

    @QtCore.Slot()
    def try_accept(self) -> None:
        db = Path(self.db.text().strip()).expanduser()
        images = Path(self.images.text().strip()).expanduser()
        if not db.is_file():
            QtWidgets.QMessageBox.warning(self, "Missing DB", "Please select an existing pipeline.sqlite3 file.")
            return
        if not images.is_dir():
            QtWidgets.QMessageBox.warning(self, "Missing image folder", "Please select the source image folder.")
            return
        if not self.run.text().strip():
            QtWidgets.QMessageBox.warning(self, "Missing run", "Please enter or load a run_id.")
            return
        self.accept()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings_service: SettingsService, settings: dict[str, Any]) -> None:
        super().__init__()
        self._settings_service = settings_service
        self._settings = settings
        self._db_path = str(settings.get("db_path") or "")
        self._image_root = str(settings.get("image_root") or "")
        self._run_id = str(settings.get("run_id") or "myrun")
        self._reviewer = str(settings.get("reviewer") or "")
        self._notes = str(settings.get("notes") or "")
        self._model_name = str(settings.get("model_name") or "facebook/dinov2-giant")
        self.image_service = ImageService(max_items=1000, max_full_items=8, max_thumb_items=900, parent=self)
        self.setWindowTitle("Semi-labeling Review")
        self.resize(1420, 860)

        central = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.connection_bar())

        self.tabs = QtWidgets.QTabWidget(central)
        self.review_page = ReviewPage(self, cleaned=False)
        self.qa_page = ReviewPage(self, cleaned=True)
        self.images_page = ImageOverviewPage(self)
        self.prototype_page = PrototypePage(self)
        self.tabs.addTab(self.review_page, "Review")
        self.tabs.addTab(self.qa_page, "QA cleaned")
        self.tabs.addTab(self.images_page, "Images")
        self.tabs.addTab(self.prototype_page, "Prototype")
        self.tabs.currentChanged.connect(self.ensure_current_tab_loaded)
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready — app only reads DB and writes JSON")
        self._install_global_shortcuts()
        QtCore.QTimer.singleShot(0, self.ensure_current_tab_loaded)

    def connection_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame(self)
        bar.setObjectName("ConnectionBar")
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)
        self.connection_label = QtWidgets.QLabel(bar)
        self.connection_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._refresh_connection_label()
        change_btn = _button("Change...")
        layout.addWidget(self.connection_label, 1)
        layout.addWidget(change_btn)
        change_btn.clicked.connect(self.change_connection)
        return bar

    def _install_global_shortcuts(self) -> None:
        for tab_index, key in enumerate(("Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+4")):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda index=tab_index: self.tabs.setCurrentIndex(index))

    def ensure_current_tab_loaded(self, *_args: object) -> None:
        page = self.tabs.currentWidget()
        if hasattr(page, "ensure_loaded"):
            page.ensure_loaded()

    def _refresh_connection_label(self) -> None:
        if hasattr(self, "connection_label"):
            self.connection_label.setText(f"[data] DB: {self._db_path} | Images: {self._image_root} | Run: {self._run_id}")

    def db_path(self) -> str:
        return self._db_path

    def image_root(self) -> str:
        return self._image_root

    def run_id(self) -> str:
        return self._run_id or "myrun"

    def reviewer(self) -> str:
        return self._reviewer

    def notes(self) -> str:
        return self._notes

    def model_name(self) -> str:
        return self._model_name or "facebook/dinov2-giant"

    @QtCore.Slot()
    def change_connection(self) -> None:
        dialog = ConnectDialog(self._settings, self)
        dialog.db.setText(self._db_path)
        dialog.images.setText(self._image_root)
        dialog.run.setText(self._run_id)
        dialog.reviewer.setText(self._reviewer)
        dialog.notes.setText(self._notes)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.apply_connection(dialog.payload())

    def apply_connection(self, payload: dict[str, Any]) -> None:
        self._db_path = str(payload.get("db_path") or "")
        self._image_root = str(payload.get("image_root") or "")
        self._run_id = str(payload.get("run_id") or "myrun")
        self._reviewer = str(payload.get("reviewer") or "")
        self._notes = str(payload.get("notes") or "")
        self._settings.update(payload)
        self._refresh_connection_label()
        self.save_settings()
        for page in (self.review_page, self.qa_page, self.images_page, self.prototype_page):
            page._loaded_once = False
            page.items = []
            if hasattr(page, "visible_items"):
                page.visible_items = []
            if hasattr(page, "excluded"):
                page.excluded.clear()
            page.list.set_payloads([], page.title, page.thumb_key)
            page.image.clear()
        self.ensure_current_tab_loaded()

    def save_settings(self) -> None:
        settings = dict(DEFAULT_SETTINGS)
        settings.update({
            "db_path": self.db_path(),
            "image_root": self.image_root(),
            "run_id": self.run_id(),
            "reviewer": self.reviewer(),
            "notes": self.notes(),
            "model_name": self.model_name(),
        })
        self._settings_service.save({"settings": settings})
        self.status("Settings saved")

    def status(self, text: str) -> None:
        self.statusBar().showMessage(text, 5000)

    def error(self, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Semi-labeling", text)
        self.status(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.save_settings()
        super().closeEvent(event)
