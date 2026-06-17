"""Prototype / domain representative review page (step 5).

The pipeline proposes representative candidates per visual domain.  The reviewer
assigns the final prototype label using four actions: crack, mold, spall,
reject.  DB access runs through ``self.db`` (off the GUI thread); the diverse
ordering is cached per filter signature and only recomputed when filters change.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ...config.defaults import LABELS
from ...services import db_service
from ...services.handoff_service import write_handoff_json
from ..options_dialog import Option
from ..widgets.box_image import BoxImage
from ..widgets.payload_list import PayloadList
from ..widgets.ui_kit import Card, Chip, DecisionBar, InfoPanel, LABEL_BUTTON_STYLE, Toolbar, primary_button
from .base_page import BasePage


_BANDS = ("all", "low", "mid", "high", "extra", "anchor")
_REVIEW_STATES = ("all", "unreviewed", "reviewed", "accepted", "relabeled", "rejected")
_SORT_MODES = (
    ("diverse", "Diverse (default)"),
    ("score_desc", "Score ↓"),
    ("score_asc", "Score ↑"),
    ("sim_desc", "Centroid sim ↓"),
    ("domain_asc", "Domain ↑"),
    ("id_asc", "ID ↑"),
)


class PrototypePage(BasePage):
    title_text = "Prototype"

    def __init__(self, window: "Any") -> None:
        super().__init__(window)
        self.items: list[Any] = []
        self.visible_items: list[Any] = []
        self.decisions: dict[int, dict[str, Any]] = {}
        self._thumb_size = 96
        self._order_cache: dict[tuple, list[Any]] = {}
        self._filter_value = "all"  # legacy options_spec compat
        self._prototype_policy = {
            "damage_total_per_label": 200,
            "reject_total": 300,
            "score_triplet_per_domain": 3,
            "score_triplet_bands": ["low", "mid", "high"],
            "score_field": "reliability_score",
            "fallback": "domain-first anchors, then balanced top-up to target count",
        }
        self._build_ui()
        self._wire()
        self._install_shortcuts()

    # -- options (legacy dialog compat) -----------------------------------
    def options_spec(self) -> list[Option]:
        return [Option("filter", "Show label", "choice", self._filter_value, choices=["all", *LABELS])]

    def apply_options(self, values: dict[str, Any]) -> None:
        self._filter_value = str(values.get("filter", self._filter_value))
        if self._filter_value == "all":
            for label in LABELS:
                self._label_chips[label].setChecked(True)
        else:
            for label, chip in self._label_chips.items():
                chip.setChecked(label == self._filter_value)
        self.refresh()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Filter toolbar ──────────────────────────────────────────────
        bar = Toolbar(self)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setPlaceholderText("Search id / image path / cluster…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(220)

        self._label_chips: dict[str, QtWidgets.QCheckBox] = {}
        chip_box = QtWidgets.QWidget(self)
        chip_layout = QtWidgets.QHBoxLayout(chip_box)
        chip_layout.setContentsMargins(0, 0, 0, 0)
        chip_layout.setSpacing(4)
        for label in LABELS:
            cb = QtWidgets.QCheckBox(label.title(), self)
            cb.setChecked(True)
            self._label_chips[label] = cb
            chip_layout.addWidget(cb)

        self.band_combo = QtWidgets.QComboBox(self)
        self.band_combo.addItems(list(_BANDS))

        self.review_combo = QtWidgets.QComboBox(self)
        self.review_combo.addItems(list(_REVIEW_STATES))

        self.domain_combo = QtWidgets.QComboBox(self)
        self.domain_combo.addItem("All domains", -1)

        self.score_min = QtWidgets.QDoubleSpinBox(self)
        self.score_min.setRange(0.0, 1.0)
        self.score_min.setSingleStep(0.05)
        self.score_min.setDecimals(2)
        self.score_min.setValue(0.0)
        self.score_min.setPrefix("≥ ")

        self.score_max = QtWidgets.QDoubleSpinBox(self)
        self.score_max.setRange(0.0, 1.0)
        self.score_max.setSingleStep(0.05)
        self.score_max.setDecimals(2)
        self.score_max.setValue(1.0)
        self.score_max.setPrefix("≤ ")

        self.sim_min = QtWidgets.QDoubleSpinBox(self)
        self.sim_min.setRange(0.0, 1.0)
        self.sim_min.setSingleStep(0.05)
        self.sim_min.setDecimals(2)
        self.sim_min.setValue(0.0)
        self.sim_min.setPrefix("sim ≥ ")

        self.cluster_min = QtWidgets.QSpinBox(self)
        self.cluster_min.setRange(0, 100000)
        self.cluster_min.setSingleStep(1)
        self.cluster_min.setValue(0)
        self.cluster_min.setPrefix("cluster ≥ ")

        self.sort_combo = QtWidgets.QComboBox(self)
        for key, txt in _SORT_MODES:
            self.sort_combo.addItem(txt, key)

        self.reset_btn = primary_button("Reset")
        self.summary = Chip("Items: 0 / 0", self)

        bar.add_label("Search")
        bar.add(self.search_edit)
        bar.add_separator()
        bar.add(chip_box)
        bar.add_separator()
        bar.add_label("Band")
        bar.add(self.band_combo)
        bar.add_label("Review")
        bar.add(self.review_combo)
        bar.add_label("Domain")
        bar.add(self.domain_combo)
        bar.add_separator()
        bar.add(self.score_min)
        bar.add(self.score_max)
        bar.add(self.sim_min)
        bar.add(self.cluster_min)
        bar.add_separator()
        bar.add_label("Sort")
        bar.add(self.sort_combo)
        bar.add_stretch()
        bar.add(self.reset_btn)
        bar.add(self.summary)
        root.addWidget(bar)

        # ── Main 3-pane split ────────────────────────────────────────────
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
            "Enter = accept shown label · 1/2/3/4 = crack/mold/spall/reject · "
            "Space/↓ = next · ↑ = previous · Ctrl+S = write JSON",
            info,
        )
        hint.setWordWrap(True)
        info_layout.addWidget(candidate_card)
        info_layout.addWidget(progress_card)
        info_layout.addWidget(hint)
        info_layout.addStretch(1)

        split.addWidget(self.list)
        split.addWidget(center)
        split.addWidget(info)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 2)
        root.addWidget(split, 1)

    def _wire(self) -> None:
        self.decision_bar.decided.connect(self.decide)
        self.image.clicked.connect(self.next_item)
        self.list.currentPayloadChanged.connect(self.show_item)
        self.list.visibleRowsChanged.connect(self.load_visible_thumbnails)
        self._image_service.imageLoaded.connect(self.on_image_loaded)
        self._image_service.imageFailed.connect(self.on_image_failed)
        self.db.subscribe("candidates", self._on_candidates_loaded, self.window.error)

        # Filter wiring — debounce search, immediate for everything else
        self._search_timer = QtCore.QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(180)
        self._search_timer.timeout.connect(self.refresh)
        self.search_edit.textChanged.connect(lambda _t: self._search_timer.start())

        for chip in self._label_chips.values():
            chip.toggled.connect(lambda _v: self.refresh())
        self.band_combo.currentTextChanged.connect(lambda _v: self.refresh())
        self.review_combo.currentTextChanged.connect(lambda _v: self.refresh())
        self.domain_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        self.score_min.valueChanged.connect(lambda _v: self._on_score_min_changed())
        self.score_max.valueChanged.connect(lambda _v: self._on_score_max_changed())
        self.sim_min.valueChanged.connect(lambda _v: self.refresh())
        self.cluster_min.valueChanged.connect(lambda _v: self.refresh())
        self.sort_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        self.reset_btn.clicked.connect(self._reset_filters)

    def _on_score_min_changed(self) -> None:
        if self.score_min.value() > self.score_max.value():
            self.score_max.setValue(self.score_min.value())
        self.refresh()

    def _on_score_max_changed(self) -> None:
        if self.score_max.value() < self.score_min.value():
            self.score_min.setValue(self.score_max.value())
        self.refresh()

    def _reset_filters(self) -> None:
        self.search_edit.clear()
        for chip in self._label_chips.values():
            chip.setChecked(True)
        self.band_combo.setCurrentIndex(0)
        self.review_combo.setCurrentIndex(0)
        self.domain_combo.setCurrentIndex(0)
        self.score_min.setValue(0.0)
        self.score_max.setValue(1.0)
        self.sim_min.setValue(0.0)
        self.cluster_min.setValue(0)
        self.sort_combo.setCurrentIndex(0)
        self.refresh()

    @QtCore.Slot()
    def load(self) -> None:
        self.db.submit(
            "candidates",
            db_service.list_prototype_candidates,
            self.window.db_path(), self.window.run_id(), self.window.image_root(),
            reject_below=0.5, per_band=80,
        )

    def _on_candidates_loaded(self, payload: Any) -> None:
        candidates = list(payload.get("items") or [])
        selected = self._representatives_by_policy(candidates)
        self.items = [item for label in ("crack", "mold", "spall", "reject") for item in selected.get(label, [])]
        self._order_cache.clear()
        self._populate_domain_combo()
        self._loaded_once = True
        self.refresh()
        self.window.status(
            f"Loaded {len(self.items)} prototype representatives from {len(candidates)} candidates"
        )

    def _populate_domain_combo(self) -> None:
        prev = self.domain_combo.currentData()
        domains = sorted({
            int(getattr(i, "domain_index", -1) or -1)
            for i in self.items
            if getattr(i, "domain_index", None) is not None
        })
        self.domain_combo.blockSignals(True)
        self.domain_combo.clear()
        self.domain_combo.addItem("All domains", -1)
        for d in domains:
            self.domain_combo.addItem(f"D{d}", d)
        idx = self.domain_combo.findData(prev)
        if idx >= 0:
            self.domain_combo.setCurrentIndex(idx)
        self.domain_combo.blockSignals(False)

    # -- filter pipeline ---------------------------------------------------
    def _filter_signature(self) -> tuple:
        labels = tuple(sorted(l for l, c in self._label_chips.items() if c.isChecked()))
        return (
            self.search_edit.text().strip().lower(),
            labels,
            self.band_combo.currentText(),
            self.review_combo.currentText(),
            int(self.domain_combo.currentData() or -1),
            round(self.score_min.value(), 3),
            round(self.score_max.value(), 3),
            round(self.sim_min.value(), 3),
            int(self.cluster_min.value()),
            str(self.sort_combo.currentData() or "diverse"),
        )

    def _apply_filters(self, items: list[Any]) -> list[Any]:
        sig = self._filter_signature()
        query, labels, band, review, domain, smin, smax, sim_min, csz_min, _sort = sig
        out: list[Any] = []
        for item in items:
            label = str(getattr(item, "label", "") or "")
            if labels and label not in labels:
                continue
            if band != "all":
                item_band = self._score_band(item)
                bands = item_band.split("+") if item_band else []
                if band not in bands:
                    continue
            if domain >= 0 and int(getattr(item, "domain_index", -1) or -1) != domain:
                continue
            score = float(getattr(item, "reliability_score", 0) or 0)
            if score < smin or score > smax:
                continue
            sim = getattr(item, "centroid_similarity", None)
            if sim_min > 0 and (sim is None or float(sim) < sim_min):
                continue
            if int(getattr(item, "cluster_size", 0) or 0) < csz_min:
                continue
            rid = int(getattr(item, "result_id", 0) or 0)
            decision = self.decisions.get(rid)
            if review != "all":
                if review == "unreviewed" and decision is not None:
                    continue
                if review == "reviewed" and decision is None:
                    continue
                if review in ("accepted", "relabeled", "rejected"):
                    if decision is None:
                        continue
                    action = str(decision.get("action") or "")
                    suffix = action.split("prototype_", 1)[-1] if "prototype_" in action else ""
                    if suffix != review:
                        continue
            if query:
                hay = " ".join(str(x) for x in (
                    rid,
                    getattr(item, "image_rel_path", ""),
                    getattr(item, "cluster_id", ""),
                    label,
                )).lower()
                if query not in hay:
                    continue
            out.append(item)
        return out

    def _sort_items(self, items: list[Any], mode: str) -> list[Any]:
        if mode == "score_desc":
            return sorted(items, key=lambda i: (-self._score_key(i), int(getattr(i, "result_id", 0) or 0)))
        if mode == "score_asc":
            return sorted(items, key=lambda i: (self._score_key(i), int(getattr(i, "result_id", 0) or 0)))
        if mode == "sim_desc":
            return sorted(items, key=lambda i: (-(float(getattr(i, "centroid_similarity", 0) or 0)), int(getattr(i, "result_id", 0) or 0)))
        if mode == "domain_asc":
            return sorted(items, key=lambda i: (int(getattr(i, "domain_index", 0) or 0), int(getattr(i, "result_id", 0) or 0)))
        if mode == "id_asc":
            return sorted(items, key=lambda i: int(getattr(i, "result_id", 0) or 0))
        return self._diverse_order(list(items))

    @QtCore.Slot()
    def refresh(self, keep_result_id: int | None = None) -> None:
        sig = self._filter_signature()
        if sig not in self._order_cache:
            filtered = self._apply_filters(self.items)
            mode = str(self.sort_combo.currentData() or "diverse")
            self._order_cache[sig] = self._sort_items(filtered, mode)
        self.visible_items = self._order_cache[sig]
        self.list.set_payloads(self.visible_items, self.title, self.thumb_key)
        self.summary.setText(f"Items: {len(self.visible_items)} / {len(self.items)}")
        if keep_result_id is not None:
            for row, item in enumerate(self.visible_items):
                if int(getattr(item, "result_id", 0) or 0) == int(keep_result_id):
                    self.list.setCurrentRow(row)
                    break
        self.render_summary()
        self.load_visible_thumbnails(self.list.visible_rows())

    # -- representative selection / ordering -------------------------------
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

        topup_candidates = list(rows)
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

    # -- rendering ---------------------------------------------------------
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
        self.image.set_loading_item(item, prefer_full_image=True)
        if not self._show_full_image(("full", image_path), image_path, self.image):
            self.image.clear()
            self.window.status("Không tìm thấy ảnh. Kiểm tra Image folder phải trỏ tới thư mục chứa file gốc trong DB.")
            return
        rid = int(getattr(item, "result_id", 0))
        original = str(getattr(item, "label", "") or "")
        chosen = str(self.decisions.get(rid, {}).get("label") or "")
        domain = getattr(item, "domain_index", None)
        domain_text = "?" if domain is None else str(int(domain))
        state = chosen or "unreviewed"
        band = self._score_band(item).upper() or "SCORE"
        self.decision_bar.set_current(chosen)
        self.decision_bar.set_caption(f"#{rid}  {original}  {band}")
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

    # -- decisions ---------------------------------------------------------
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
        # Decisions affect "review" filter results, so invalidate cache
        self._order_cache.clear()
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
        self.progress.set_rows([
            ("Reviewed", f"{reviewed} / {total}"),
            ("Visible", str(len(self.visible_items))),
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

    def _install_shortcuts(self) -> None:
        def add(key, callback) -> None:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)

        for key, label in zip(("1", "2", "3", "4"), LABELS, strict=False):
            add(key, lambda value=label: self.decide(value))
        add("Enter", self.accept_current_label)
        add("Return", self.accept_current_label)
        add("Space", self.next_item)
        add("Down", self.next_item)
        add("Up", self.previous_item)
        add(QtGui.QKeySequence.StandardKey.Save, self.write_json)

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
        except Exception as exc:  # noqa: BLE001
            self.window.error(str(exc))
