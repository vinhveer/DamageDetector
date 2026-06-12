"""Image overview page: browse images with all their boxes overlaid."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ...services import db_service
from ..options_dialog import Option
from ..widgets.box_image import BoxImage
from ..widgets.payload_list import PayloadList
from ..widgets.ui_kit import Card, InfoPanel
from .base_page import BasePage


class ImageOverviewPage(BasePage):
    title_text = "Before"

    def __init__(self, window: "Any") -> None:
        super().__init__(window)
        self.items: list[dict[str, Any]] = []
        self._source_value = "all"
        self._limit_value = 1000
        self._build_ui()
        self._wire()

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)

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
        self.list.set_payloads(self.items, self.title, self.thumb_key)
        self._loaded_once = True
        self.window.status(f"Loaded {len(self.items)} images ({int(payload.get('total') or 0)} total)")

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
            boxes = list(item.get("boxes") or []) if isinstance(item, dict) else []
            rel = str(item.get("image_rel_path") or "") if isinstance(item, dict) else ""
            self.image.set_pixmap(pixmap)
            self.image.set_overlay_boxes(boxes=boxes, caption=f"{rel}  boxes={len(boxes)}")
