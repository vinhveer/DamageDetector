from pathlib import Path
from PySide6 import QtCore, QtWidgets

class CompareDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare Ground Truth")
        self.setMinimumWidth(400)

        layout = QtWidgets.QVBoxLayout(self)

        # Folder select
        folder_layout = QtWidgets.QHBoxLayout()
        self._folder_input = QtWidgets.QLineEdit()
        self._folder_input.setPlaceholderText("Select ground truth folder...")
        self._folder_btn = QtWidgets.QPushButton("Browse...")
        self._folder_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(self._folder_input)
        folder_layout.addWidget(self._folder_btn)
        layout.addLayout(folder_layout)

        # Prefix
        prefix_layout = QtWidgets.QHBoxLayout()
        self._prefix_input = QtWidgets.QLineEdit()
        self._prefix_input.setPlaceholderText("e.g. mask_ (optional)")
        prefix_layout.addWidget(QtWidgets.QLabel("Mask suffix/prefix:"))
        prefix_layout.addWidget(self._prefix_input)
        layout.addLayout(prefix_layout)
        
        # Hint label
        hint = QtWidgets.QLabel("If image is 'hello.jpg' and mask is 'hellomask.jpg', enter 'mask' here.")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select GT Folder")
        if path:
            self._folder_input.setText(path)

    def get_result(self) -> tuple[str, str]:
        return self._folder_input.text().strip(), self._prefix_input.text().strip()
