from __future__ import annotations

from PySide6 import QtWidgets


class IsolateDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        prompt: str = "",
        mode: str = "dino_sam",
        action: str = "keep",
        profile: str = "QUALITY",
        crop: bool = False,
        white: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Isolate Settings")
        self.resize(460, 320)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(QtWidgets.QLabel("Prompt:", self))
        self.prompt_edit = QtWidgets.QLineEdit(self)
        self.prompt_edit.setPlaceholderText("e.g. crack, spall, mold")
        self.prompt_edit.setText(prompt)
        layout.addWidget(self.prompt_edit)

        layout.addWidget(QtWidgets.QLabel("Method:", self))
        self.mode_combo = QtWidgets.QComboBox(self)
        self.mode_combo.addItem("DINO + SAM", "dino_sam")
        self.mode_combo.addItem("SAM Only", "sam_only")
        mode_index = max(0, self.mode_combo.findData(str(mode or "dino_sam").strip().lower()))
        self.mode_combo.setCurrentIndex(mode_index)
        layout.addWidget(self.mode_combo)

        layout.addWidget(QtWidgets.QLabel("Output:", self))
        self.action_combo = QtWidgets.QComboBox(self)
        self.action_combo.addItem("Keep object", "keep")
        self.action_combo.addItem("Erase object", "erase")
        action_index = max(0, self.action_combo.findData(str(action or "keep").strip().lower()))
        self.action_combo.setCurrentIndex(action_index)
        layout.addWidget(self.action_combo)

        layout.addWidget(QtWidgets.QLabel("Profile:", self))
        self.profile_combo = QtWidgets.QComboBox(self)
        self.profile_combo.addItem("FAST", "FAST")
        self.profile_combo.addItem("QUALITY", "QUALITY")
        self.profile_combo.addItem("ULTRA", "ULTRA")
        chosen = str(profile or "QUALITY").strip().upper()
        index = max(0, self.profile_combo.findData(chosen))
        self.profile_combo.setCurrentIndex(index)
        self.profile_combo.setEnabled(False)
        layout.addWidget(self.profile_combo)

        self.crop_check = QtWidgets.QCheckBox("Crop to bounding box", self)
        self.crop_check.setChecked(crop)
        layout.addWidget(self.crop_check)

        self.white_check = QtWidgets.QCheckBox("Fill masked area with white (255)", self)
        self.white_check.setChecked(white)
        layout.addWidget(self.white_check)

        self.action_combo.currentIndexChanged.connect(self._sync_state)
        self._sync_state()

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
            "prompt": self.prompt_edit.text().strip(),
            "mode": str(self.mode_combo.currentData() or "dino_sam"),
            "action": str(self.action_combo.currentData() or "keep"),
            "profile": "QUALITY",
            "crop": self.crop_check.isChecked(),
            "white": self.white_check.isChecked(),
        }

    def _sync_state(self) -> None:
        is_erase = str(self.action_combo.currentData() or "keep") == "erase"
        if is_erase:
            self.crop_check.setChecked(False)
        else:
            self.crop_check.setChecked(True)
        self.crop_check.setEnabled(False)
