from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from canvas import ImageCanvas
from predict_unet import UnetParams

from .dialogs import ProcessingDialog
from .features.explorer import ExplorerPanel
from .features.image_tools import ImageToolsPanel
from .features.predict.workers import WorkerBase
from .main_window_actions import MainWindowActionsMixin
from .main_window_io import MainWindowIOMixin
from .main_window_layout import MainWindowLayoutMixin
from .main_window_predict import MainWindowPredictMixin
from .main_window_settings import MainWindowSettingsMixin
from .state import LoadedState
from .toolbar import ToolbarController


class MainWindow(
    QtWidgets.QMainWindow,
    MainWindowLayoutMixin,
    MainWindowActionsMixin,
    MainWindowIOMixin,
    MainWindowSettingsMixin,
    MainWindowPredictMixin,
):
    def __init__(self) -> None:
        super().__init__()

        self.resize(1400, 860)
        self._thread: QtCore.QThread | None = None
        self._worker: WorkerBase | None = None
        self._active_stop_btn: QtWidgets.QPushButton | None = None
        self._active_log_widget: QtWidgets.QPlainTextEdit | None = None
        self._progress_dialog: ProcessingDialog | None = None
        self._post_run_action: dict | None = None

        self._state: LoadedState | None = None
        self._mask_path: Path | None = None
        self._pending_unet: tuple[str, UnetParams] | None = None
        self._current_run_id: str | None = None
        self._current_run_scope: str | None = None
        self._current_run_started_at: str | None = None

        self._overlay_canvas = ImageCanvas(self, render_mode="overlay", editable=True)
        self._image_canvas = ImageCanvas(self, render_mode="image", editable=False)
        self._mask_canvas = ImageCanvas(self, render_mode="mask", editable=False)

        self._image_detections: dict[str, list[dict]] = {}
        self._image_detections_all: dict[str, list[dict]] = {}
        self._history_view_detections: dict[str, list[dict]] = {}
        self._folder_images: list[str] = []
        self._explorer_panel = ExplorerPanel(self)
        self._explorer_panel.imageClicked.connect(lambda p: self.load_image(p, switch_tab=False))
        self._explorer_panel.imageActivated.connect(lambda p: self.load_image(p, switch_tab=True))

        self._image_tools_panel = ImageToolsPanel(self._overlay_canvas, self)
        self._brush_slider = self._image_tools_panel.brush_slider()
        self._brush_spin = self._image_tools_panel.brush_spin()
        self._brush_value = self._image_tools_panel.brush_value_label()
        self._overlay_slider = self._image_tools_panel.overlay_slider()
        self._overlay_spin = self._image_tools_panel.overlay_spin()

        self._roi_selecting = False
        self._toolbar: ToolbarController | None = None

        self._overlay_canvas.cursorInfo.connect(self._on_cursor_info)
        self._overlay_canvas.brushRadiusChanged.connect(self._sync_brush_slider)
        self._overlay_canvas.maskChanged.connect(self._sync_mask_views)
        self._overlay_canvas.roiSelected.connect(self._on_roi_selected)
        self._overlay_canvas.roiCanceled.connect(self._on_roi_canceled)

        self._status_label = QtWidgets.QLabel("")
        self._image_label = QtWidgets.QLabel("")
        self._mask_label = QtWidgets.QLabel("")
        self._image_label.setMinimumWidth(240)
        self._mask_label.setMinimumWidth(200)
        self.statusBar().addPermanentWidget(self._status_label)
        self.statusBar().addPermanentWidget(self._image_label, 1)
        self.statusBar().addPermanentWidget(self._mask_label)

        self._build_actions()
        self._build_toolbar()
        self._build_layout()
        self._init_settings_persistence()
        self._init_workspace()  # Sets up output dir and title
        self._refresh_ui_state()
