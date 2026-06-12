"""Review / QA page.

One class drives two tabs:

* ``cleaned=False`` — the uncertain review_queue (manual relabel / reject)
* ``cleaned=True``  — QA over already-cleaned labels (corrections)

All DB access goes through ``self.db`` (DbExecutor) so the GUI thread never
blocks on SQLite.  Decisions are kept in-memory and exported as a JSON handoff.
"""
from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ...config.defaults import LABELS
from ...services import db_service
from ...services.handoff_service import write_handoff_json
from ..options_dialog import Option, OptionsDialog
from ..widgets.box_image import BoxImage
from ..widgets.payload_list import PayloadList
from ..widgets.ui_kit import (
    Card,
    Chip,
    DecisionBar,
    InfoPanel,
    LABEL_BUTTON_STYLE,
    PercentBar,
    PickedList,
    primary_button,
)
from .base_page import BasePage


class ReviewPage(BasePage):
    #: Tab title used by the window for the options menu label.
    title_text = "Review"

    def __init__(self, window: "Any", *, cleaned: bool = False) -> None:
        super().__init__(window)
        self.cleaned = cleaned
        self.title_text = "QA cleaned" if cleaned else "Review"
        self.items: list[Any] = []
        self.pending: dict[int, dict[str, Any]] = {}
        self._boxes: dict[str, list[dict[str, Any]]] = {}
        self._show_context_boxes = not cleaned
        self._thumb_size = 96 if not cleaned else 88
        self._full_prefetch_count = 2 if not cleaned else 0
        self._decision_history: list[tuple[int, dict[str, Any] | None]] = []
        # Options previously shown as toolbar controls; now edited via dialog.
        self._filter_value = "all"
        self._limit_value = 1000 if cleaned else 500

        self._build_ui()
        self._wire()
        self._install_shortcuts()

    # -- construction ------------------------------------------------------
    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.list = PayloadList(split)

        center = QtWidgets.QWidget(split)
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(8, 0, 8, 0)
        center_layout.setSpacing(8)
        self.image = BoxImage(center)
        decision_actions = [
            (
                label,
                LABEL_BUTTON_STYLE.get(label, ("#7f8c8d", label.title()))[1],
                LABEL_BUTTON_STYLE.get(label, ("#7f8c8d", ""))[0],
            )
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
        progress_card = Card("Progress", detail)
        self.side_progress = PercentBar(0, "0 / 0", progress_card)
        progress_card.add(self.side_progress)
        pending_card = Card("Pending", detail)
        self.pending_count = Chip("0 pending", pending_card)
        self.pending_list = PickedList(pending_card)
        self.pending_list.setMaximumHeight(200)
        self.save_btn = primary_button("Write JSON (0)")
        pending_card.add(self.pending_count)
        pending_card.add(self.pending_list)
        pending_card.add(self.save_btn)
        detail_layout.addWidget(item_card)
        detail_layout.addWidget(progress_card)
        detail_layout.addWidget(pending_card, 1)

        split.addWidget(self.list)
        split.addWidget(center)
        split.addWidget(detail)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 2)
        root.addWidget(split, 1)

    def _wire(self) -> None:
        self.save_btn.clicked.connect(self.write_json)
        self.pending_list.removeRequested.connect(self.remove_pending)
        self.decision_bar.decided.connect(self.decide)
        self.image.clicked.connect(self.next_item)
        self.list.currentPayloadChanged.connect(self.show_item)
        self.list.visibleRowsChanged.connect(self.load_visible_thumbnails)
        self._image_service.imageLoaded.connect(self.on_image_loaded)
        self._image_service.imageFailed.connect(self.on_image_failed)
        self.db.subscribe("items", self._on_items_loaded, self.window.error)
        self.db.subscribe("boxes", self._on_boxes_loaded)

    # -- options (moved out of the toolbar into a dialog) ------------------
    def options_spec(self) -> list[Option]:
        spec = [
            Option("limit", "Limit", "int", self._limit_value, minimum=50, maximum=5000, step=50),
        ]
        if self.cleaned:
            spec.insert(0, Option("filter", "Filter label", "choice", self._filter_value, choices=["all", *LABELS]))
        return spec

    def apply_options(self, values: dict[str, Any]) -> None:
        if "filter" in values:
            self._filter_value = str(values["filter"])
        if "limit" in values:
            self._limit_value = int(values["limit"])
        self.load()

    # -- data loading (async) ---------------------------------------------
    @QtCore.Slot()
    def load(self) -> None:
        if self.cleaned:
            label = self._filter_value
            self.db.submit(
                "items",
                db_service.list_cleaned,
                self.window.db_path(), self.window.run_id(), self.window.image_root(),
                final_label="" if label == "all" else label,
                limit=self._limit_value,
            )
        else:
            self.db.submit(
                "items",
                db_service.list_queue,
                self.window.db_path(), self.window.run_id(), self.window.image_root(),
                queue_type="", sample_ratio=0.0, limit=self._limit_value,
            )

    def _on_items_loaded(self, payload: Any) -> None:
        self.items = list(payload.get("items") or [])
        self.list.set_payloads(self.items, self.title, self.thumb_key)
        self.load_visible_thumbnails(self.list.visible_rows())
        self._loaded_once = True
        self.render_pending(refresh_list=False)
        self.window.status(f"Loaded {len(self.items)} items")

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
        rid = int(getattr(item, "result_id", 0) or 0)
        self.image.set_loading_item(item, prefer_full_image=True)
        if not self._show_full_image(("full", image_path), image_path, self.image):
            self.image.clear()
            self.window.status(
                "Không tìm thấy ảnh. Kiểm tra Image folder phải trỏ tới thư mục chứa file gốc trong DB."
            )
            return
        rel = str(getattr(item, "image_rel_path", "") or "")
        if rel and self._show_context_boxes:
            boxes = self._boxes.get(rel)
            if boxes is None:
                self.db.submit(
                    "boxes", self._fetch_boxes, self.window.db_path(), self.window.run_id(), rel, rid
                )
            else:
                self.image.set_other_boxes(boxes, rid)
        else:
            self.image.set_other_boxes([], rid)
        self._render_detail(item, rid, rel)
        self.prefetch_neighbors()

    @staticmethod
    def _fetch_boxes(db: str, run_id: str, rel: str, rid: int) -> dict[str, Any]:
        boxes = db_service.list_image_boxes(db, run_id, rel)
        return {"rel": rel, "rid": rid, "boxes": boxes}

    def _on_boxes_loaded(self, payload: dict[str, Any]) -> None:
        rel = str(payload.get("rel") or "")
        boxes = list(payload.get("boxes") or [])
        self._boxes[rel] = boxes
        current = self.list.current_payload()
        if current is not None and str(getattr(current, "image_rel_path", "") or "") == rel:
            self.image.set_other_boxes(boxes, int(payload.get("rid") or 0))

    def _render_detail(self, item: Any, rid: int, rel: str) -> None:
        label = str(getattr(item, "final_label", getattr(item, "suggested_label", "")) or "")
        reasons = ", ".join(getattr(item, "reasons", ()) or ()) or "-"
        pending = self.pending.get(rid)
        pending_label = str(pending.get("newLabel") or "") if pending else ""
        score = float(getattr(item, "reliability_score", 0) or 0)
        self.decision_bar.set_current(pending_label)
        self.decision_bar.set_caption(f"#{rid}  {label}  score {score:.4f}")
        self.image.set_caption(f"#{rid}  {pending_label or label}")
        self.image.set_decision_indicator(pending_label or "")
        self.meta.set_rows([
            ("ID", str(rid)),
            ("Image", rel or "-"),
            ("Label", label or "-"),
            ("Score", f"{score:.4f}"),
            ("Reasons", reasons),
            ("Pending", pending_label or "-"),
        ])

    @QtCore.Slot(object, object)
    def on_image_loaded(self, key: object, pixmap: QtGui.QPixmap) -> None:
        if isinstance(key, tuple) and key and key[0] == "thumb":
            self.list.set_thumbnail(key, pixmap)
            return
        if key == self._current_image_key:
            self.image.set_pixmap(pixmap)

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

    # -- decisions ---------------------------------------------------------
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
        def add(key, callback) -> None:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)

        for key, label in zip(("1", "2", "3", "4"), LABELS, strict=False):
            add(key, lambda value=label: self.decide(value))
        add("Space", self.next_item)
        add("Down", self.next_item)
        add("Up", self.previous_item)
        add("Z", self.undo_last)
        add(QtGui.QKeySequence.StandardKey.Save, self.write_json)

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
        except Exception as exc:  # noqa: BLE001
            self.window.error(str(exc))
