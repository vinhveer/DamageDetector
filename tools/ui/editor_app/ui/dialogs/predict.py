from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from inference_api.prediction_models import (
    DETECTION_DINO,
    DETECTION_LABELS,
    DETECTION_NONE,
    SEGMENTATION_LABELS,
    SEGMENTATION_SAM,
    SEGMENTATION_SAM_LORA,
    SEGMENTATION_UNET,
    SCOPE_CURRENT,
    SCOPE_FOLDER,
    TASK_GROUP_CRACK_ONLY,
    TASK_GROUP_LABELS,
    TASK_GROUP_MORE_DAMAGE,
    PredictionConfig,
)


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
    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        has_image: bool,
        has_folder: bool,
        default_config: PredictionConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Prediction")
        self.setModal(True)
        self.resize(520, 520)
        self._default = (default_config or PredictionConfig(
            task_group=TASK_GROUP_CRACK_ONLY,
            segmentation_model=SEGMENTATION_SAM,
            detection_model=DETECTION_DINO,
            scope=SCOPE_CURRENT if has_image else SCOPE_FOLDER,
        )).normalized()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        self._task_buttons = self._build_radio_group(
            layout,
            title="Prediction Type",
            entries=[
                (TASK_GROUP_CRACK_ONLY, TASK_GROUP_LABELS[TASK_GROUP_CRACK_ONLY]),
                (TASK_GROUP_MORE_DAMAGE, TASK_GROUP_LABELS[TASK_GROUP_MORE_DAMAGE]),
            ],
        )
        self._seg_buttons = self._build_radio_group(
            layout,
            title="Choose Segmentation Model",
            entries=[
                (SEGMENTATION_SAM, SEGMENTATION_LABELS[SEGMENTATION_SAM]),
                (SEGMENTATION_SAM_LORA, SEGMENTATION_LABELS[SEGMENTATION_SAM_LORA]),
                (SEGMENTATION_UNET, SEGMENTATION_LABELS[SEGMENTATION_UNET]),
            ],
        )
        self._detect_buttons = self._build_radio_group(
            layout,
            title="Choose Object Detection Model",
            entries=[
                (DETECTION_DINO, DETECTION_LABELS[DETECTION_DINO]),
                (DETECTION_NONE, DETECTION_LABELS[DETECTION_NONE]),
            ],
        )

        self._scope_current = QtWidgets.QRadioButton("Current image", self)
        self._scope_folder = QtWidgets.QRadioButton("Whole folder", self)
        scope_group = QtWidgets.QGroupBox("Scope", self)
        scope_layout = QtWidgets.QVBoxLayout(scope_group)
        self._scope_current.setEnabled(has_image)
        self._scope_folder.setEnabled(has_folder)
        scope_layout.addWidget(self._scope_current)
        scope_layout.addWidget(self._scope_folder)
        layout.addWidget(scope_group)

        self._hint = QtWidgets.QLabel(self)
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        layout.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Run")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._set_checked(self._task_buttons, self._default.task_group)
        self._set_checked(self._seg_buttons, self._default.segmentation_model)
        self._set_checked(self._detect_buttons, self._default.detection_model)
        if self._default.scope == SCOPE_FOLDER and has_folder:
            self._scope_folder.setChecked(True)
        elif has_image:
            self._scope_current.setChecked(True)
        else:
            self._scope_folder.setChecked(True)

        for button in [*self._task_buttons.values(), *self._seg_buttons.values(), *self._detect_buttons.values()]:
            button.toggled.connect(self._sync_state)
        self._sync_state()

    def _build_radio_group(
        self,
        layout: QtWidgets.QVBoxLayout,
        *,
        title: str,
        entries: list[tuple[str, str]],
    ) -> dict[str, QtWidgets.QRadioButton]:
        group = QtWidgets.QGroupBox(title, self)
        group_layout = QtWidgets.QVBoxLayout(group)
        buttons: dict[str, QtWidgets.QRadioButton] = {}
        for key, label in entries:
            button = QtWidgets.QRadioButton(label, group)
            buttons[key] = button
            group_layout.addWidget(button)
        layout.addWidget(group)
        return buttons

    def _set_checked(self, buttons: dict[str, QtWidgets.QRadioButton], key: str) -> None:
        if key in buttons:
            buttons[key].setChecked(True)

    def _checked_key(self, buttons: dict[str, QtWidgets.QRadioButton], fallback: str) -> str:
        for key, button in buttons.items():
            if button.isChecked():
                return key
        return fallback

    def _sync_state(self) -> None:
        task_group = self._checked_key(self._task_buttons, TASK_GROUP_CRACK_ONLY)
        more_damage = task_group == TASK_GROUP_MORE_DAMAGE
        self._seg_buttons[SEGMENTATION_SAM_LORA].setEnabled(not more_damage)
        self._seg_buttons[SEGMENTATION_UNET].setEnabled(not more_damage)
        if more_damage and (
            self._seg_buttons[SEGMENTATION_SAM_LORA].isChecked()
            or self._seg_buttons[SEGMENTATION_UNET].isChecked()
        ):
            self._seg_buttons[SEGMENTATION_SAM].setChecked(True)
        if more_damage:
            self._hint.setText("More damage uses SAM with optional DINO detection. Crack boxes can use UNet or SAM Finetune from Settings > Tasks.")
        else:
            self._hint.setText("Crack-only supports SAM, SAM Finetune with LoRA, or UNet, with or without DINO.")

    def get_result(self) -> PredictionConfig:
        scope = SCOPE_FOLDER if self._scope_folder.isChecked() else SCOPE_CURRENT
        return PredictionConfig(
            task_group=self._checked_key(self._task_buttons, TASK_GROUP_CRACK_ONLY),
            segmentation_model=self._checked_key(self._seg_buttons, SEGMENTATION_SAM),
            detection_model=self._checked_key(self._detect_buttons, DETECTION_DINO),
            scope=scope,
        )
