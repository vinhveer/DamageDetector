"""Connection dialog: pick the SQLite DB, image folder, run id, and metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

from ..services import db_service
from .widgets.ui_kit import primary_button


def _button(text: str) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(text)
    button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    button.setMinimumHeight(32)
    return button


class ConnectDialog(QtWidgets.QDialog):
    def __init__(self, settings: dict[str, Any], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect semi-labeling data")
        self.setModal(True)
        self.resize(760, 260)
        root = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Connect DB and image folder", self)
        title.setObjectName("DialogTitle")
        subtitle = QtWidgets.QLabel(
            "The app is read-only: it opens pipeline.sqlite3, reads images, and writes JSON handoff files.",
            self,
        )
        subtitle.setObjectName("DialogSubtitle")
        subtitle.setWordWrap(True)
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
        connect = primary_button("Connect")
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
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select pipeline.sqlite3", self.db.text().strip(), "SQLite (*.sqlite3 *.db);;All files (*)"
        )
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
        except Exception as exc:  # noqa: BLE001
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
