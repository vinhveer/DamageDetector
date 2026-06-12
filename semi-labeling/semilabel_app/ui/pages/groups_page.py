"""Groups page (step 3): browse DINOv2 visual domains / core clusters.

Each group is a near-duplicate cluster of crack/mold/spall crops.  The grid
shows representative thumbnails; double-clicking a group lists its members.
"""
from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ...services import db_service
from ..widgets.box_image import BoxImage
from ..widgets.payload_list import PayloadGrid
from ..widgets.ui_kit import Card, Chip, InfoPanel, Toolbar, primary_button
from .base_page import BasePage


class GroupsPage(BasePage):
    list_attr = "grid"

    def __init__(self, window: "Any") -> None:
        super().__init__(window)
        self.groups: list[Any] = []
        self.members: list[Any] = []
        self._mode = "groups"  # "groups" | "members"
        self._active_group: Any | None = None
        self._thumb_size = 132
        self._build_ui()
        self._wire()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = Toolbar(self)
        self.label_filter = QtWidgets.QComboBox(self)
        self.label_filter.addItems(["all", "crack", "mold", "spall"])
        self.back_btn = primary_button("< Back to groups")
        self.back_btn.setVisible(False)
        self.busy = Chip("Loading…", self)
        self.busy.setObjectName("BusyChip")
        self.busy.setVisible(False)
        self.summary = Chip("Groups: 0", self)
        top.add_label("Show")
        top.add(self.label_filter)
        top.add(self.back_btn)
        top.add_stretch()
        top.add(self.busy)
        top.add(self.summary)
        root.addWidget(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.grid = PayloadGrid(split)

        detail = QtWidgets.QWidget(split)
        detail_layout = QtWidgets.QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(8)
        self.image = BoxImage(detail)
        info_card = Card("Group", detail)
        self.meta = InfoPanel(info_card)
        info_card.add(self.meta)
        hint = QtWidgets.QLabel(
            "Step 3: visual domains (core clusters) per label. Double-click a group to view its "
            "member crops, then use Back to return.",
            detail,
        )
        hint.setWordWrap(True)
        detail_layout.addWidget(self.image, 1)
        detail_layout.addWidget(info_card)
        detail_layout.addWidget(hint)

        split.addWidget(self.grid)
        split.addWidget(detail)
        split.setStretchFactor(0, 6)
        split.setStretchFactor(1, 3)
        root.addWidget(split, 1)

    def _wire(self) -> None:
        self.label_filter.currentTextChanged.connect(lambda _v: self.load())
        self.back_btn.clicked.connect(self.show_groups)
        self.grid.currentPayloadChanged.connect(self.show_item)
        self.grid.visibleRowsChanged.connect(self.load_visible_thumbnails)
        self.grid.doubleClicked.connect(self._on_double_clicked)
        self._image_service.imageLoaded.connect(self.on_image_loaded)
        self._image_service.imageFailed.connect(self.on_image_failed)
        self.db.subscribe("groups", self._on_groups_loaded, self.window.error)
        self.db.subscribe("members", self._on_members_loaded, self.window.error)

    @QtCore.Slot()
    def load(self) -> None:
        label = self.label_filter.currentText()
        self.db.submit(
            "groups",
            db_service.list_groups,
            self.window.db_path(), self.window.run_id(), self.window.image_root(),
            label="" if label == "all" else label,
        )

    def _on_groups_loaded(self, payload: Any) -> None:
        self.groups = list(payload.get("items") or [])
        self._loaded_once = True
        self.show_groups()
        counts = payload.get("counts") or {}
        counts_text = "  ".join(f"{k}:{counts[k]}" for k in ("crack", "mold", "spall") if k in counts)
        self.window.status(f"Loaded {len(self.groups)} groups   {counts_text}")

    @QtCore.Slot()
    def show_groups(self) -> None:
        self._mode = "groups"
        self._active_group = None
        self.back_btn.setVisible(False)
        self.grid.set_payloads(self.groups, self.group_title, self.group_thumb_key)
        self.summary.setText(f"Groups: {len(self.groups)}")
        self.load_visible_thumbnails(self.grid.visible_rows())

    def open_group(self, group: Any) -> None:
        cid = str(getattr(group, "core_cluster_id", "") or "")
        if not cid:
            return
        self._active_group = group
        self.db.submit(
            "members",
            db_service.list_group_members,
            self.window.db_path(), self.window.run_id(), self.window.image_root(), cid, limit=600,
        )

    def _on_members_loaded(self, payload: Any) -> None:
        self.members = list(payload.get("items") or [])
        group = self._active_group
        if group is None:
            return
        self._mode = "members"
        self.back_btn.setVisible(True)
        self.grid.set_payloads(self.members, self.member_title, self.member_thumb_key)
        self.summary.setText(
            f"{group.label}  D{int(getattr(group, 'domain_index', 0))}  members: {len(self.members)}"
        )
        self.load_visible_thumbnails(self.grid.visible_rows())

    @QtCore.Slot(QtCore.QModelIndex)
    def _on_double_clicked(self, index: QtCore.QModelIndex) -> None:
        if self._mode != "groups":
            return
        payload = self.grid.payload_at(index.row())
        if payload is not None:
            self.open_group(payload)

    def group_title(self, group: Any) -> str:
        label = str(getattr(group, "label", "") or "")
        domain = int(getattr(group, "domain_index", 0) or 0)
        size = int(getattr(group, "size", 0) or 0)
        return f"{label}  D{domain}\nsize {size}"

    def group_thumb_key(self, group: Any) -> tuple[str, str]:
        return ("group_thumb", str(getattr(group, "core_cluster_id", "") or ""))

    def member_title(self, item: Any) -> str:
        rid = int(getattr(item, "result_id", 0) or 0)
        sim = float(getattr(item, "centroid_similarity", 0) or 0)
        return f"#{rid}\nsim {sim:.3f}"

    def member_thumb_key(self, item: Any) -> tuple[str, int]:
        return ("group_member_thumb", int(getattr(item, "result_id", 0) or 0))

    def _group_image_path(self, group: Any) -> str:
        uri = str(getattr(group, "rep_image_uri", "") or "")
        return QtCore.QUrl(uri).toLocalFile() if uri.startswith("file:") else uri

    def _member_image_path(self, item: Any) -> str:
        return BoxImage.item_path(item, prefer_full_image=True)

    @QtCore.Slot(object)
    def load_visible_thumbnails(self, rows: object) -> None:
        payloads = self.groups if self._mode == "groups" else self.members
        for row in list(rows or []):
            if not isinstance(row, int) or not (0 <= row < len(payloads)):
                continue
            payload = payloads[row]
            if self._mode == "groups":
                key = self.group_thumb_key(payload)
                path = self._group_image_path(payload)
                box = getattr(payload, "rep_box", None)
            else:
                key = self.member_thumb_key(payload)
                path = self._member_image_path(payload)
                box = getattr(payload, "box", None)
            cached = self._image_service.cached(key)
            if cached is not None:
                self.grid.set_thumbnail(key, cached)
                continue
            if path:
                self._image_service.load_item_thumbnail(key, path, box, size=self._thumb_size)

    @QtCore.Slot(object)
    def show_item(self, payload: Any) -> None:
        if self._mode == "groups":
            self._show_group(payload)
        else:
            self._show_member(payload)

    def _show_group(self, group: Any) -> None:
        image_path = self._group_image_path(group)
        rep_box = getattr(group, "rep_box", None)
        rep_rid = getattr(group, "rep_result_id", None)
        caption = f"{group.label}  D{int(getattr(group, 'domain_index', 0))}  rep #{rep_rid}"
        boxes = [{"result_id": int(rep_rid or 0), "label": str(group.label or ""), "box": rep_box}] if rep_box else []
        self.image.set_overlay_loading(boxes=boxes, caption=caption)
        if self._show_full_image(("group_full", image_path), image_path, self.image):
            if self._image_service.cached(("group_full", image_path)) is not None:
                self.image.set_overlay_boxes(boxes=boxes, caption=caption)
        else:
            self.window.status("Không tìm thấy ảnh đại diện. Kiểm tra Image folder.")
        self.meta.set_rows([
            ("Label", str(group.label or "-")),
            ("Domain", f"D{int(getattr(group, 'domain_index', 0))}"),
            ("Size", str(int(getattr(group, "size", 0) or 0))),
            ("Members", str(int(getattr(group, "member_count", 0) or 0))),
            ("Status", str(getattr(group, "status", "") or "-")),
            ("Cluster", str(getattr(group, "core_cluster_id", "") or "-")),
            ("Rep ID", str(rep_rid if rep_rid is not None else "-")),
            ("Rep sim", f"{float(getattr(group, 'rep_similarity', 0) or 0):.4f}"),
            ("Rep image", str(getattr(group, "rep_image_rel_path", "") or "-")),
        ])

    def _show_member(self, item: Any) -> None:
        image_path = self._member_image_path(item)
        rid = int(getattr(item, "result_id", 0) or 0)
        box = getattr(item, "box", None)
        caption = f"#{rid}  sim {float(getattr(item, 'centroid_similarity', 0) or 0):.3f}"
        boxes = [{"result_id": rid, "label": str(getattr(item, "label", "") or ""), "box": box}] if box else []
        self.image.set_overlay_loading(boxes=boxes, caption=caption)
        if self._show_full_image(("group_full", image_path), image_path, self.image):
            if self._image_service.cached(("group_full", image_path)) is not None:
                self.image.set_overlay_boxes(boxes=boxes, caption=caption)
        else:
            self.window.status("Không tìm thấy ảnh. Kiểm tra Image folder.")
        self.meta.set_rows([
            ("ID", str(rid)),
            ("Label", str(getattr(item, "label", "") or "-")),
            ("Centroid sim", f"{float(getattr(item, 'centroid_similarity', 0) or 0):.4f}"),
            ("Image", str(getattr(item, "image_rel_path", "") or "-")),
        ])

    @QtCore.Slot(object, object)
    def on_image_loaded(self, key: object, pixmap: QtGui.QPixmap) -> None:
        if isinstance(key, tuple) and key and key[0] in ("group_thumb", "group_member_thumb"):
            self.grid.set_thumbnail(key, pixmap)
            return
        if key == self._current_image_key:
            self.image.set_pixmap(pixmap)
            payload = self.grid.current_payload()
            if payload is not None:
                if self._mode == "groups":
                    rep_box = getattr(payload, "rep_box", None)
                    rep_rid = getattr(payload, "rep_result_id", None)
                    boxes = [{"result_id": int(rep_rid or 0), "label": str(payload.label or ""), "box": rep_box}] if rep_box else []
                else:
                    box = getattr(payload, "box", None)
                    boxes = [{"result_id": int(getattr(payload, "result_id", 0) or 0), "label": str(getattr(payload, "label", "") or ""), "box": box}] if box else []
                self.image.set_overlay_boxes(boxes=boxes, caption=self.image._caption)
