from PySide6 import QtWidgets, QtCore

class IsolateDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, labels: str = "", crop: bool = False, white: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Isolate Object Settings")
        self.resize(400, 200)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Labels
        layout.addWidget(QtWidgets.QLabel("Target labels (comma-separated):"))
        self.labels_edit = QtWidgets.QLineEdit(self)
        self.labels_edit.setPlaceholderText("e.g. crack, spall")
        self.labels_edit.setText(labels)
        layout.addWidget(self.labels_edit)
        
        # Options
        self.crop_check = QtWidgets.QCheckBox("Crop to bounding box", self)
        self.crop_check.setChecked(crop)
        layout.addWidget(self.crop_check)
        
        self.white_check = QtWidgets.QCheckBox("Outside area white (255)", self)
        self.white_check.setChecked(white)
        layout.addWidget(self.white_check)
        
        layout.addStretch()
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_values(self) -> dict:
        return {
            "labels": self.labels_edit.text().strip(),
            "crop": self.crop_check.isChecked(),
            "white": self.white_check.isChecked(),
        }
