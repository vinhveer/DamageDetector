from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..config.defaults import DEFAULT_SETTINGS, LABELS
from ..services import db_service
from ..services.handoff_service import write_handoff_json
from ..services.settings_service import SettingsService
from .widgets.box_image import BoxImage


STYLE = """
QMainWindow { background: #f6f7f8; }
QFrame#TopBar { background: #ffffff; border-bottom: 1px solid #dde1e5; }
QFrame#Panel { background: #ffffff; border: 1px solid #dde1e5; border-radius: 8px; }
QLabel#Title { font-size: 18px; font-weight: 700; color: #202428; }
QLabel#Muted { color: #68707a; }
QPushButton { min-height: 30px; padding: 4px 10px; }
QPushButton#Primary { background: #2563eb; color: white; border-radius: 6px; font-weight: 700; }
QPushButton#Danger { color: #a40000; font-weight: 700; }
QListWidget { background: white; border: 1px solid #dde1e5; border-radius: 6px; }
QTextEdit { background: white; border: 1px solid #dde1e5; border-radius: 6px; }
"""


def _button(text: str, *, primary: bool = False, danger: bool = False) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    if primary:
        b.setObjectName("Primary")
    if danger:
        b.setObjectName("Danger")
    return b


class PayloadList(QtWidgets.QListWidget):
    currentPayloadChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.currentItemChanged.connect(self._emit_current)

    def set_payloads(self, payloads: list[Any], title_fn) -> None:
        self.clear()
        for payload in payloads:
            item = QtWidgets.QListWidgetItem(title_fn(payload))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, payload)
            self.addItem(item)
        if self.count():
            self.setCurrentRow(0)

    def current_payload(self) -> Any | None:
        item = self.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def _emit_current(self, current: QtWidgets.QListWidgetItem | None, _previous) -> None:
        if current:
            self.currentPayloadChanged.emit(current.data(QtCore.Qt.ItemDataRole.UserRole))


class ReviewPage(QtWidgets.QWidget):
    def __init__(self, window: "MainWindow", *, cleaned: bool = False) -> None:
        super().__init__(window)
        self.window = window
        self.cleaned = cleaned
        self.items: list[Any] = []
        self.pending: dict[int, dict[str, Any]] = {}
        self._boxes: dict[str, list[dict[str, Any]]] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        self.filter = QtWidgets.QComboBox(self)
        self.filter.addItems(["all", *LABELS] if cleaned else ["all"])
        self.limit = QtWidgets.QSpinBox(self)
        self.limit.setRange(50, 5000)
        self.limit.setValue(500)
        self.load_btn = _button("Load")
        self.save_btn = _button("Write JSON", primary=True)
        top.addWidget(QtWidgets.QLabel("Filter"))
        top.addWidget(self.filter)
        top.addWidget(QtWidgets.QLabel("Limit"))
        top.addWidget(self.limit)
        top.addWidget(self.load_btn)
        top.addStretch(1)
        self.pending_label = QtWidgets.QLabel("Pending: 0")
        self.pending_label.setObjectName("Muted")
        top.addWidget(self.pending_label)
        top.addWidget(self.save_btn)
        root.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.list = PayloadList(split)
        self.image = BoxImage(split)
        detail = QtWidgets.QFrame(split)
        detail.setObjectName("Panel")
        detail_layout = QtWidgets.QVBoxLayout(detail)
        self.meta = QtWidgets.QLabel("No item")
        self.meta.setWordWrap(True)
        self.meta.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        detail_layout.addWidget(self.meta)
        detail_layout.addSpacing(8)
        for label in LABELS:
            b = _button(label, danger=(label == "reject"))
            b.clicked.connect(lambda _checked=False, value=label: self.decide(value))
            detail_layout.addWidget(b)
        detail_layout.addStretch(1)
        split.addWidget(self.list)
        split.addWidget(self.image)
        split.addWidget(detail)
        split.setSizes([320, 760, 260])
        root.addWidget(split, 1)

        self.load_btn.clicked.connect(self.load)
        self.save_btn.clicked.connect(self.write_json)
        self.list.currentPayloadChanged.connect(self.show_item)

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
                    queue_type="", sample_ratio=0.0,
                )
                self.items = list(payload.get("items") or [])[: self.limit.value()]
            self.list.set_payloads(self.items, self.title)
            self.window.status(f"Loaded {len(self.items)} items")
        except Exception as exc:
            self.window.error(str(exc))

    def title(self, item: Any) -> str:
        rid = int(getattr(item, "result_id", 0))
        label = str(getattr(item, "final_label", getattr(item, "suggested_label", "")) or "")
        score = float(getattr(item, "reliability_score", 0) or 0)
        mark = "  *" if rid in self.pending else ""
        return f"#{rid}  {label}  {score:.3f}{mark}"

    def show_item(self, item: Any) -> None:
        self.image.set_item(item, prefer_full_image=True)
        rel = str(getattr(item, "image_rel_path", "") or "")
        rid = int(getattr(item, "result_id", 0) or 0)
        if rel:
            try:
                boxes = self._boxes.get(rel)
                if boxes is None:
                    boxes = db_service.list_image_boxes(self.window.db_path(), self.window.run_id(), rel)
                    self._boxes[rel] = boxes
                self.image.set_other_boxes(boxes, rid)
            except Exception:
                self.image.set_other_boxes([], rid)
        label = str(getattr(item, "final_label", getattr(item, "suggested_label", "")) or "")
        reasons = ", ".join(getattr(item, "reasons", ()) or ()) or "-"
        pending = self.pending.get(rid)
        self.meta.setText(
            f"ID: {rid}\n"
            f"Image: {rel}\n"
            f"Label: {label}\n"
            f"Score: {float(getattr(item, 'reliability_score', 0) or 0):.4f}\n"
            f"Reasons: {reasons}\n"
            f"Pending: {pending or '-'}"
        )

    def decide(self, label: str) -> None:
        item = self.list.current_payload()
        if item is None:
            return
        rid = int(getattr(item, "result_id"))
        prev = str(getattr(item, "final_label", getattr(item, "suggested_label", "")) or "")
        self.pending[rid] = {
            "resultId": rid,
            "action": "manual_reject" if label == "reject" else ("manual_accept" if label == prev else "manual_relabel"),
            "previousLabel": prev,
            "newLabel": label,
        }
        self.render_pending()
        row = self.list.currentRow()
        if row + 1 < self.list.count():
            self.list.setCurrentRow(row + 1)

    def render_pending(self) -> None:
        self.pending_label.setText(f"Pending: {len(self.pending)}")
        current = self.list.currentRow()
        self.list.set_payloads(self.items, self.title)
        if 0 <= current < self.list.count():
            self.list.setCurrentRow(current)

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
            self.render_pending()
            self.window.status(f"JSON written: {path}")
            QtWidgets.QMessageBox.information(self, "JSON written", str(path))
        except Exception as exc:
            self.window.error(str(exc))


class PrototypePage(QtWidgets.QWidget):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.window = window
        self.items: list[Any] = []
        self.picks: dict[int, dict[str, Any]] = {}
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QComboBox(self)
        self.label.addItems(LABELS)
        self.load_btn = _button("Load")
        self.pick_btn = _button("Pick")
        self.reject_btn = _button("Pick reject", danger=True)
        self.save_btn = _button("Write JSON", primary=True)
        self.count = QtWidgets.QLabel("Picks: 0")
        self.count.setObjectName("Muted")
        for w in (QtWidgets.QLabel("Label"), self.label, self.load_btn, self.pick_btn, self.reject_btn):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.count)
        top.addWidget(self.save_btn)
        root.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.list = PayloadList(split)
        self.image = BoxImage(split)
        self.meta = QtWidgets.QLabel("No candidate", split)
        self.meta.setWordWrap(True)
        self.meta.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        split.addWidget(self.list)
        split.addWidget(self.image)
        split.addWidget(self.meta)
        split.setSizes([320, 760, 260])
        root.addWidget(split, 1)

        self.load_btn.clicked.connect(self.load)
        self.pick_btn.clicked.connect(lambda: self.pick(False))
        self.reject_btn.clicked.connect(lambda: self.pick(True))
        self.save_btn.clicked.connect(self.write_json)
        self.label.currentTextChanged.connect(self.refresh)
        self.list.currentPayloadChanged.connect(self.show_item)

    def load(self) -> None:
        try:
            payload = db_service.list_prototype_candidates(
                self.window.db_path(), self.window.run_id(), self.window.image_root(), reject_below=0.5, per_band=200
            )
            self.items = list(payload.get("items") or [])
            self.refresh()
            self.window.status(f"Loaded {len(self.items)} candidates")
        except Exception as exc:
            self.window.error(str(exc))

    def refresh(self) -> None:
        label = self.label.currentText()
        rows = [i for i in self.items if str(getattr(i, "label", "")) == label]
        self.list.set_payloads(rows, self.title)
        self.count.setText(f"Picks: {len(self.picks)}")

    def title(self, item: Any) -> str:
        rid = int(getattr(item, "result_id", 0))
        mark = "  *" if rid in self.picks else ""
        return f"#{rid}  {getattr(item, 'label', '')}  {float(getattr(item, 'reliability_score', 0)):.3f}{mark}"

    def show_item(self, item: Any) -> None:
        self.image.set_item(item, prefer_full_image=True)
        rid = int(getattr(item, "result_id", 0))
        self.meta.setText(
            f"ID: {rid}\n"
            f"Label: {getattr(item, 'label', '')}\n"
            f"Pred: {getattr(item, 'predicted_label', '')}\n"
            f"Score: {float(getattr(item, 'reliability_score', 0)):.4f}\n"
            f"Cluster: {getattr(item, 'cluster_id', '')}\n"
            f"Picked: {self.picks.get(rid) or '-'}"
        )

    def pick(self, reject: bool) -> None:
        item = self.list.current_payload()
        if item is None:
            return
        rid = int(getattr(item, "result_id"))
        value = {"resultId": rid, "label": "reject" if reject else self.label.currentText(), "isReject": bool(reject)}
        if self.picks.get(rid) == value:
            self.picks.pop(rid, None)
        else:
            self.picks[rid] = value
        self.refresh()

    def write_json(self) -> None:
        if not self.picks:
            self.window.error("No prototype picks.")
            return
        prototypes = [p for p in self.picks.values() if not p.get("isReject") and p.get("label") != "reject"]
        rejects = [p for p in self.picks.values() if p.get("isReject") or p.get("label") == "reject"]
        payload = {
            "type": "prototype_request",
            "db": self.window.db_path(),
            "run_id": self.window.run_id(),
            "model_name": self.window.model_name(),
            "view_name": "tight",
            "notes": self.window.notes(),
            "prototypes": prototypes,
            "rejects": rejects,
            "run_seed": True,
            "run_policy": True,
        }
        try:
            path = write_handoff_json(self.window.db_path(), payload, kind="prototype", run_id=self.window.run_id())
            self.picks.clear()
            self.refresh()
            self.window.status(f"JSON written: {path}")
            QtWidgets.QMessageBox.information(self, "JSON written", str(path))
        except Exception as exc:
            self.window.error(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings_service: SettingsService, settings: dict[str, Any]) -> None:
        super().__init__()
        self._settings_service = settings_service
        self._settings = settings
        self.setWindowTitle("Semi-labeling Review")
        self.resize(1420, 860)
        self.setStyleSheet(STYLE)

        central = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.top_bar())

        self.tabs = QtWidgets.QTabWidget(central)
        self.tabs.addTab(ReviewPage(self, cleaned=False), "Review")
        self.tabs.addTab(ReviewPage(self, cleaned=True), "QA cleaned")
        self.tabs.addTab(PrototypePage(self), "Prototype")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready — app only reads DB and writes JSON")

    def top_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame(self)
        bar.setObjectName("TopBar")
        layout = QtWidgets.QGridLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        self.db = QtWidgets.QLineEdit(str(self._settings.get("db_path") or ""), bar)
        self.img = QtWidgets.QLineEdit(str(self._settings.get("image_root") or ""), bar)
        self.run = QtWidgets.QLineEdit(str(self._settings.get("run_id") or "myrun"), bar)
        self.reviewer_edit = QtWidgets.QLineEdit(str(self._settings.get("reviewer") or ""), bar)
        self.notes_edit = QtWidgets.QLineEdit(str(self._settings.get("notes") or ""), bar)
        self.model = QtWidgets.QLineEdit(str(self._settings.get("model_name") or "facebook/dinov2-giant"), bar)
        self.model.setVisible(False)
        db_btn = _button("DB")
        img_btn = _button("Images")
        save_btn = _button("Save")
        layout.addWidget(QtWidgets.QLabel("DB"), 0, 0)
        layout.addWidget(self.db, 0, 1)
        layout.addWidget(db_btn, 0, 2)
        layout.addWidget(QtWidgets.QLabel("Images"), 0, 3)
        layout.addWidget(self.img, 0, 4)
        layout.addWidget(img_btn, 0, 5)
        layout.addWidget(QtWidgets.QLabel("Run"), 1, 0)
        layout.addWidget(self.run, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Reviewer"), 1, 2)
        layout.addWidget(self.reviewer_edit, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Notes"), 1, 4)
        layout.addWidget(self.notes_edit, 1, 5)
        layout.addWidget(save_btn, 1, 6)
        db_btn.clicked.connect(self.browse_db)
        img_btn.clicked.connect(self.browse_img)
        save_btn.clicked.connect(self.save_settings)
        return bar

    def db_path(self) -> str:
        return self.db.text().strip()

    def image_root(self) -> str:
        return self.img.text().strip()

    def run_id(self) -> str:
        return self.run.text().strip() or "myrun"

    def reviewer(self) -> str:
        return self.reviewer_edit.text().strip()

    def notes(self) -> str:
        return self.notes_edit.text().strip()

    def model_name(self) -> str:
        return self.model.text().strip() or "facebook/dinov2-giant"

    def browse_db(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select pipeline.sqlite3", self.db_path(), "SQLite (*.sqlite3 *.db);;All files (*)")
        if path:
            self.db.setText(path)

    def browse_img(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select image root", self.image_root())
        if path:
            self.img.setText(path)

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