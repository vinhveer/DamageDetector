from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from inference_api.editor_bridge import settings_pages_for_mode

from editor_app.ui.components.prediction_forms import DinoSettingsForm, SamSettingsForm, UnetSettingsForm


class ProcessingDialog(QtWidgets.QDialog):
    stopRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None, title: str) -> None:
        super().__init__(parent)
        self._allow_close = False
        self.setWindowTitle(title)
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self.resize(720, 420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(title, self))
        self._log = QtWidgets.QPlainTextEdit(self)
        self._log.setReadOnly(True)
        layout.addWidget(self._log, 1)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self._stop_btn = QtWidgets.QPushButton("Stop", self)
        self._stop_btn.clicked.connect(self.stopRequested.emit)
        buttons.addWidget(self._stop_btn)
        layout.addLayout(buttons)

    def log_widget(self) -> QtWidgets.QPlainTextEdit:
        return self._log

    def stop_button(self) -> QtWidgets.QPushButton:
        return self._stop_btn

    def allow_close(self) -> None:
        self._allow_close = True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
        else:
            event.ignore()


class PredictRunDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, *, has_image: bool, has_folder: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Prediction")
        self.setModal(True)
        self.resize(420, 340)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self._mode_buttons: dict[str, QtWidgets.QRadioButton] = {}
        model_group = QtWidgets.QGroupBox("Workflow", self)
        model_layout = QtWidgets.QVBoxLayout(model_group)
        for mode, label in (
            ("sam_dino", "SAM + DINO"),
            ("sam_dino_ft", "SAM + DINO + Finetune"),
            ("sam_only", "SAM Only"),
            ("sam_only_ft", "SAM Only + Finetune"),
            ("sam_tiled", "SAM + DINO Tiled"),
            ("unet_only", "UNet Only"),
            ("unet_dino", "UNet + DINO"),
        ):
            button = QtWidgets.QRadioButton(label, model_group)
            self._mode_buttons[mode] = button
            model_layout.addWidget(button)
        self._mode_buttons["sam_dino"].setChecked(True)
        layout.addWidget(model_group)

        self._scope_current = QtWidgets.QRadioButton("Current Image", self)
        self._scope_folder = QtWidgets.QRadioButton("Whole Folder", self)
        scope_group = QtWidgets.QGroupBox("Scope", self)
        scope_layout = QtWidgets.QVBoxLayout(scope_group)
        self._scope_current.setEnabled(has_image)
        self._scope_folder.setEnabled(has_folder)
        if has_image:
            self._scope_current.setChecked(True)
        else:
            self._scope_folder.setChecked(True)
        scope_layout.addWidget(self._scope_current)
        scope_layout.addWidget(self._scope_folder)
        layout.addWidget(scope_group)

        layout.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Run")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_result(self) -> tuple[str, str]:
        mode = next((key for key, button in self._mode_buttons.items() if button.isChecked()), "sam_dino")
        return mode, ("folder" if self._scope_folder.isChecked() else "current")


class PredictDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        title: str,
        mode: str,
        settings: dict,
        has_image: bool,
        has_folder: bool,
        show_scope: bool = True,
        pages: list[str] | None = None,
        ok_text: str = "Run",
        initial_page: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(860, 720)
        self._show_scope = bool(show_scope)
        self._widgets: dict[str, QtWidgets.QWidget] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        root.addWidget(QtWidgets.QLabel(title, self))

        self._scope_current: QtWidgets.QRadioButton | None = None
        self._scope_folder: QtWidgets.QRadioButton | None = None
        if self._show_scope:
            scope_group = QtWidgets.QGroupBox("Scope", self)
            scope_layout = QtWidgets.QHBoxLayout(scope_group)
            self._scope_current = QtWidgets.QRadioButton("Current image", scope_group)
            self._scope_folder = QtWidgets.QRadioButton("Whole folder", scope_group)
            self._scope_current.setEnabled(bool(has_image))
            self._scope_folder.setEnabled(bool(has_folder))
            if has_image:
                self._scope_current.setChecked(True)
            else:
                self._scope_folder.setChecked(True)
            scope_layout.addWidget(self._scope_current)
            scope_layout.addWidget(self._scope_folder)
            scope_layout.addStretch(1)
            root.addWidget(scope_group)

        tabs = QtWidgets.QTabWidget(self)
        tabs.setDocumentMode(True)
        root.addWidget(tabs, 1)

        for page in (pages or settings_pages_for_mode(str(mode).strip().lower())):
            key = str(page or "").strip().lower()
            if key == "sam":
                tabs.addTab(SamSettingsForm(settings, self._widgets, tabs), "sam")
            elif key == "dino":
                tabs.addTab(DinoSettingsForm(settings, self._widgets, tabs), "dino")
            elif key == "unet":
                tabs.addTab(UnetSettingsForm(settings, self._widgets, tabs), "unet")

        if initial_page:
            want = str(initial_page).strip().lower()
            for index in range(tabs.count()):
                if tabs.tabText(index).strip().lower() == want:
                    tabs.setCurrentIndex(index)
                    break

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(str(ok_text or "Run"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def selected_scope(self) -> str:
        if not self._show_scope or self._scope_folder is None:
            return "current"
        return "folder" if self._scope_folder.isChecked() else "current"

    def settings_dict(self) -> dict:
        out: dict = {}
        for key, widget in self._widgets.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                out[key] = widget.text().strip()
            elif isinstance(widget, QtWidgets.QCheckBox):
                out[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QSpinBox):
                out[key] = int(widget.value())
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                out[key] = float(widget.value())
            elif isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentData()
                out[key] = value if value is not None else widget.currentText()
        return out
