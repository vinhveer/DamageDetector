"""Image overview page: browse images with all their boxes overlaid."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ...services import db_service
from ..options_dialog import Option
from ..widgets.box_image import BoxImage
from ..widgets.payload_list import PayloadList
from ..widgets.ui_kit import Card, Chip, InfoPanel, Toolbar
from .base_page import BasePage


class ImageOverviewPage(BasePage):
    title_text = "Before"

    def __init__(self, window: "Any", *, default_source: str = "all", title_text: str = "Before") -> None:
        super().__init__(window)
        self.title_text = title_text
        self.items: list[dict[str, Any]] = []
        self._filtered: list[dict[str, Any]] = []
        self._source_value = default_source
        self._limit_value = 1000
        self._build_ui()
        self._wire()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Top filter toolbar
        top = Toolbar(self)
        self.search = QtWidgets.QLineEdit(self)
        self.search.setPlaceholderText("Search image path…")
        self.search.setClearButtonEnabled(True)
        self.search.setMaximumWidth(260)

        self.label_filter = QtWidgets.QComboBox(self)
        self.label_filter.addItem("All labels", "all")
        for cls in ("crack", "mold", "spall", "reject"):
            self.label_filter.addItem(cls.title(), cls)

        self.min_boxes = QtWidgets.QSpinBox(self)
        self.min_boxes.setRange(0, 999)
        self.min_boxes.setValue(0)
        self.min_boxes.setPrefix("≥ ")
        self.min_boxes.setSuffix(" boxes")
        self.min_boxes.setMaximumWidth(120)

        self.score_min = QtWidgets.QDoubleSpinBox(self)
        self.score_min.setRange(0.0, 1.0)
        self.score_min.setSingleStep(0.05)
        self.score_min.setDecimals(2)
        self.score_min.setValue(0.0)
        self.score_min.setMaximumWidth(80)

        self.sort_combo = QtWidgets.QComboBox(self)
        self.sort_combo.addItem("Most boxes", "boxes_desc")
        self.sort_combo.addItem("Fewest boxes", "boxes_asc")
        self.sort_combo.addItem("Most review", "review_desc")
        self.sort_combo.addItem("Most cleaned", "cleaned_desc")
        self.sort_combo.addItem("Path A→Z", "path_asc")

        self.busy = Chip("Loading…", self)
        self.busy.setObjectName("BusyChip")
        self.busy.setVisible(False)
        self.summary = Chip("0 / 0 images", self)

        top.add_label("Search")
        top.add(self.search)
        top.add_separator()
        top.add_label("Label")
        top.add(self.label_filter)
        top.add(self.min_boxes)
        top.add_label("Score ≥")
        top.add(self.score_min)
        top.add_separator()
        top.add_label("Sort")
        top.add(self.sort_combo)
        top.add_stretch()
        top.add(self.busy)
        top.add(self.summary)
        root.addWidget(top)

        # Body splitter
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

    def _wire(self) -> None:
        self.list.currentPayloadChanged.connect(self.show_item)
        self._image_service.imageLoaded.connect(self.on_image_loaded)
        self._image_service.imageFailed.connect(self.on_image_failed)
        self.db.subscribe("images", self._on_images_loaded, self.window.error)
        self.search.textChanged.connect(self._refresh_filtered)
        self.label_filter.currentIndexChanged.connect(self._refresh_filtered)
        self.min_boxes.valueChanged.connect(self._refresh_filtered)
        self.score_min.valueChanged.connect(self._refresh_filtered)
        self.sort_combo.currentIndexChanged.connect(self._refresh_filtered)

    # -- options -----------------------------------------------------------
    def options_spec(self) -> list[Option]:
        return [
            Option("source", "Show", "choice", self._source_value, choices=["all", "cleaned", "review"]),
            Option("limit", "Limit", "int", self._limit_value, minimum=20, maximum=5000, step=50),
        ]

    def apply_options(self, values: dict[str, Any]) -> None:
        self._source_value = str(values.get("source", self._source_value))
        self._limit_value = int(values.get("limit", self._limit_value))
        self.load()

    @QtCore.Slot()
    def load(self) -> None:
        self.db.submit(
            "images",
            db_service.list_image_overview,
            self.window.db_path(), self.window.run_id(), self.window.image_root(),
            source=self._source_value, limit=self._limit_value,
        )

    def _on_images_loaded(self, payload: Any) -> None:
        self.items = list(payload.get("items") or [])
        self._loaded_once = True
        self._refresh_filtered()
        total = int(payload.get("total") or len(self.items))
        self.window.status(f"Loaded {len(self.items)} images ({total} total)")

    # -- filtering ---------------------------------------------------------
    def _refresh_filtered(self) -> None:
        query = self.search.text().strip().lower()
        label = self.label_filter.currentData() or "all"
        min_boxes = int(self.min_boxes.value())
        score_min = float(self.score_min.value())
        sort_key = self.sort_combo.currentData() or "boxes_desc"

        out: list[dict[str, Any]] = []
        for item in self.items:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("image_rel_path") or "").lower()
            if query and query not in rel:
                continue

            boxes = list(item.get("boxes") or [])
            if score_min > 0:
                boxes = [b for b in boxes if float(b.get("score") or 0) >= score_min]

            if label != "all":
                boxes = [b for b in boxes if str(b.get("label") or "") == label]
                if not boxes:
                    continue

            box_count = len(boxes)
            if box_count < min_boxes:
                continue

            cleaned = sum(1 for b in boxes if str(b.get("source") or "") == "cleaned")
            review = sum(1 for b in boxes if str(b.get("source") or "") == "review")
            entry = dict(item)
            entry["_filtered_boxes"] = boxes
            entry["_filtered_count"] = box_count
            entry["_filtered_cleaned"] = cleaned
            entry["_filtered_review"] = review
            out.append(entry)

        out.sort(key=self._sort_key(sort_key))
        self._filtered = out
        self.list.set_payloads(self._filtered, self.title, self.thumb_key)
        self.summary.setText(f"{len(self._filtered)} / {len(self.items)} images")

    def _sort_key(self, mode: str):
        if mode == "boxes_asc":
            return lambda i: (int(i.get("_filtered_count", 0)), str(i.get("image_rel_path") or ""))
        if mode == "review_desc":
            return lambda i: (-int(i.get("_filtered_review", 0)), str(i.get("image_rel_path") or ""))
        if mode == "cleaned_desc":
            return lambda i: (-int(i.get("_filtered_cleaned", 0)), str(i.get("image_rel_path") or ""))
        if mode == "path_asc":
            return lambda i: str(i.get("image_rel_path") or "")
        return lambda i: (-int(i.get("_filtered_count", 0)), str(i.get("image_rel_path") or ""))

    # -- rendering ---------------------------------------------------------
    def title(self, item: Any) -> str:
        rel = str(item.get("image_rel_path") if isinstance(item, dict) else "")
        box_count = int(item.get("_filtered_count", item.get("box_count", 0)) if isinstance(item, dict) else 0)
        cleaned = int(item.get("_filtered_cleaned", item.get("cleaned_count", 0)) if isinstance(item, dict) else 0)
        review = int(item.get("_filtered_review", item.get("review_count", 0)) if isinstance(item, dict) else 0)
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
        boxes = list(item.get("_filtered_boxes") or item.get("boxes") or [])
        caption = f"{rel}  boxes={len(boxes)}"
        self.image.set_overlay_loading(boxes=boxes, caption=caption)
        if not self._show_full_image(("image_overview_full", image_path), image_path, self.image):
            self.image.clear()
            self.window.status("Không tìm thấy ảnh. Kiểm tra Image folder phải trỏ tới thư mục ảnh gốc.")
            return
        if self._image_service.cached(("image_overview_full", image_path)) is not None:
            self.image.set_overlay_boxes(boxes=boxes, caption=caption)
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
            boxes = list(item.get("_filtered_boxes") or item.get("boxes") or []) if isinstance(item, dict) else []
            rel = str(item.get("image_rel_path") or "") if isinstance(item, dict) else ""
            self.image.set_pixmap(pixmap)
            self.image.set_overlay_boxes(boxes=boxes, caption=f"{rel}  boxes={len(boxes)}")
