from __future__ import annotations

from PySide6 import QtWidgets


class IsolateDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, labels: str = "", crop: bool = False, white: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Isolate Settings")
        self.resize(420, 220)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(QtWidgets.QLabel("Target labels (comma-separated):", self))
        self.labels_edit = QtWidgets.QLineEdit(self)
        self.labels_edit.setPlaceholderText("e.g. crack, spall")
        self.labels_edit.setText(labels)
        layout.addWidget(self.labels_edit)

        self.crop_check = QtWidgets.QCheckBox("Crop to bounding box", self)
        self.crop_check.setChecked(crop)
        layout.addWidget(self.crop_check)

        self.white_check = QtWidgets.QCheckBox("Outside area white (255)", self)
        self.white_check.setChecked(white)
        layout.addWidget(self.white_check)

        layout.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> dict:
        return {
            "labels": self.labels_edit.text().strip(),
            "crop": self.crop_check.isChecked(),
            "white": self.white_check.isChecked(),
        }
