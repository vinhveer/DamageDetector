from __future__ import annotations

from PySide6 import QtGui

from .toolbar import ToolbarActions, ToolbarController


class MainWindowActionsMixin:
    def _build_actions(self) -> None:
        self._act_open_workspace = QtGui.QAction("Open Workspace...", self)
        self._act_open_workspace.triggered.connect(self.open_workspace_dialog)

        self._act_add_image = QtGui.QAction("Add Image...", self)
        self._act_add_image.setShortcut(QtGui.QKeySequence.Open)
        self._act_add_image.triggered.connect(self.add_image_dialog)

        self._act_add_folder = QtGui.QAction("Add Image Folder...", self)
        self._act_add_folder.setShortcut(QtGui.QKeySequence("Ctrl+F"))
        self._act_add_folder.triggered.connect(self.add_folder_dialog)

        self._act_open_mask = QtGui.QAction("Open Mask...", self)
        self._act_open_mask.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        self._act_open_mask.triggered.connect(self.open_mask_dialog)

        self._act_save_mask = QtGui.QAction("Save Mask As...", self)
        self._act_save_mask.setShortcut(QtGui.QKeySequence.Save)
        self._act_save_mask.triggered.connect(self.save_mask_dialog)

        self._act_exit = QtGui.QAction("Exit", self)
        self._act_exit.setShortcut(QtGui.QKeySequence.Quit)
        self._act_exit.triggered.connect(self.close)

        self._act_exit.triggered.connect(self.close)

        self._act_predict_with = QtGui.QAction("Predict with...", self)
        self._act_predict_with.triggered.connect(self._open_predict_mode_dialog)

        self._act_predict_folder = QtGui.QAction("Predict Folder...", self)
        self._act_predict_folder.triggered.connect(self._predict_folder_dialog)

        self._act_isolate_object = QtGui.QAction("Isolate Object", self)
        self._act_isolate_object.triggered.connect(self._isolate_object)
        
        self._act_predict_sam_dino = QtGui.QAction("Predict SAM + DINO", self)
        self._act_predict_sam_dino.triggered.connect(lambda: self._predict_with_scope("sam_dino"))

        self._act_predict_sam_dino_ft = QtGui.QAction("Predict SAM + DINO + Finetune", self)
        self._act_predict_sam_dino_ft.triggered.connect(lambda: self._predict_with_scope("sam_dino_ft"))

        self._act_predict_unet_dino = QtGui.QAction("Predict UNet + DINO", self)
        self._act_predict_unet_dino.triggered.connect(lambda: self._predict_with_scope("unet"))

        self._act_prev_image = QtGui.QAction("Previous Image", self)
        self._act_prev_image.setShortcut(QtGui.QKeySequence("PgUp"))
        self._act_prev_image.triggered.connect(lambda: self._navigate_folder(-1))

        self._act_next_image = QtGui.QAction("Next Image", self)
        self._act_next_image.setShortcut(QtGui.QKeySequence("PgDown"))
        self._act_next_image.triggered.connect(lambda: self._navigate_folder(1))

        self._act_focus_folder_filter = QtGui.QAction("Focus Folder Filter", self)
        self._act_focus_folder_filter.setShortcut(QtGui.QKeySequence("Ctrl+K"))
        self._act_focus_folder_filter.triggered.connect(self._focus_folder_filter)

        self._act_stop = QtGui.QAction("Stop", self)
        self._act_stop.setShortcut(QtGui.QKeySequence("Ctrl+."))
        self._act_stop.setEnabled(False)
        self._act_stop.triggered.connect(self._stop_current)

        self._act_view_folder = QtGui.QAction("View: Folder", self)
        self._act_view_folder.setShortcut(QtGui.QKeySequence("Ctrl+4"))
        self._act_view_folder.triggered.connect(self._focus_folder_filter)

        self._act_view_overlay = QtGui.QAction("View: Overlay", self)
        self._act_view_overlay.setShortcut(QtGui.QKeySequence("Ctrl+1"))
        self._act_view_overlay.triggered.connect(lambda: self._view_tabs.setCurrentIndex(0))

        self._act_view_image = QtGui.QAction("View: Image", self)
        self._act_view_image.setShortcut(QtGui.QKeySequence("Ctrl+2"))
        self._act_view_image.triggered.connect(lambda: self._view_tabs.setCurrentIndex(1))

        self._act_view_mask = QtGui.QAction("View: Mask", self)
        self._act_view_mask.setShortcut(QtGui.QKeySequence("Ctrl+3"))
        self._act_view_mask.triggered.connect(lambda: self._view_tabs.setCurrentIndex(2))

        self._act_view_history_folder = QtGui.QAction("Folder History", self)
        self._act_view_history_folder.triggered.connect(self._open_folder_history_dialog)

        self._act_view_history_image = QtGui.QAction("Image History", self)
        self._act_view_history_image.triggered.connect(self._open_image_history_dialog)

        self._act_model_settings = QtGui.QAction("Model Settings...", self)
        self._act_model_settings.setShortcut(QtGui.QKeySequence("Ctrl+,"))
        self._act_model_settings.triggered.connect(self._open_model_settings_dialog)

        self._act_settings_sam = QtGui.QAction("Settings: SAM", self)
        self._act_settings_sam.setShortcut(QtGui.QKeySequence("Alt+1"))
        self._act_settings_sam.triggered.connect(lambda: self._open_model_settings_dialog("SAM"))

        self._act_settings_dino = QtGui.QAction("Settings: DINO", self)
        self._act_settings_dino.setShortcut(QtGui.QKeySequence("Alt+2"))
        self._act_settings_dino.triggered.connect(lambda: self._open_model_settings_dialog("DINO"))

        self._act_settings_unet = QtGui.QAction("Settings: UNet", self)
        self._act_settings_unet.setShortcut(QtGui.QKeySequence("Alt+3"))
        self._act_settings_unet.triggered.connect(lambda: self._open_model_settings_dialog("UNet"))

        menu = self.menuBar().addMenu("File")
        menu.addAction(self._act_open_workspace)
        menu.addSeparator()
        menu.addAction(self._act_add_image)
        menu.addAction(self._act_add_folder)
        menu.addSeparator()
        menu.addAction(self._act_open_mask)
        menu.addSeparator()
        menu.addAction(self._act_save_mask)
        menu.addSeparator()
        menu.addAction(self._act_exit)

        nav = self.menuBar().addMenu("Navigate")
        nav.addAction(self._act_prev_image)
        nav.addAction(self._act_next_image)
        nav.addSeparator()
        nav.addAction(self._act_focus_folder_filter)

        view = self.menuBar().addMenu("View")
        view.addAction(self._act_view_overlay)
        view.addAction(self._act_view_image)
        view.addAction(self._act_view_mask)
        view.addAction(self._act_view_folder)

        settings_menu = self.menuBar().addMenu("Settings")
        settings_menu.addAction(self._act_model_settings)
        settings_menu.addSeparator()
        settings_menu.addAction(self._act_settings_sam)
        settings_menu.addAction(self._act_settings_dino)
        settings_menu.addAction(self._act_settings_unet)

        run = self.menuBar().addMenu("Run")
        run.addAction(self._act_predict_with)  # Added here too
        run.addAction(self._act_predict_folder)
        run.addSeparator()
        run.addAction(self._act_predict_sam_dino)
        run.addAction(self._act_predict_sam_dino_ft)
        run.addAction(self._act_predict_unet_dino)
        run.addSeparator()
        run.addAction(self._act_stop)

    def _build_toolbar(self) -> None:
        self._toolbar = ToolbarController(self)
        self._toolbar.set_actions(
            ToolbarActions(
                predict_with=self._act_predict_with,
                isolate_object=self._act_isolate_object,
                model_settings=self._act_model_settings,
                open_folder=self._act_add_folder,
                open_image=self._act_add_image,
                open_mask=self._act_open_mask,
                save_mask=self._act_save_mask,
                prev_image=self._act_prev_image,
                next_image=self._act_next_image,
                stop=self._act_stop,
                folder_history=self._act_view_history_folder,
            )
        )

    def _visible_folder_items(self) -> int:
        return self._explorer_panel.visible_count()

    def _refresh_ui_state(self) -> None:
        running = self._thread is not None
        has_image = self._state is not None
        has_folder = bool(self._folder_images)
        has_mask = bool(self._overlay_canvas.canvas_state().mask_loaded)
        roi = bool(self._roi_selecting)

        can_interact = (not running) and (not roi)

        self._act_open_workspace.setEnabled(can_interact)
        self._act_add_image.setEnabled(can_interact)
        self._act_add_folder.setEnabled(can_interact)
        self._act_model_settings.setEnabled(not running)

        self._act_open_mask.setEnabled(can_interact and has_image)
        self._act_save_mask.setEnabled(can_interact and has_image and has_mask)

        visible_items = self._visible_folder_items()
        can_nav = can_interact and has_folder and visible_items > 1
        self._act_prev_image.setEnabled(can_nav)
        self._act_next_image.setEnabled(can_nav)

        can_predict = can_interact and (has_image or has_folder)
        self._act_predict_sam_dino.setEnabled(can_predict)
        self._act_predict_sam_dino_ft.setEnabled(can_predict)
        self._act_predict_unet_dino.setEnabled(can_predict)
        self._act_view_history_folder.setEnabled(can_interact)
        self._act_view_history_folder.setEnabled(can_interact)
        self._act_view_history_image.setEnabled(can_interact and has_image)
        self._act_isolate_object.setEnabled(can_interact and has_image)
        self._act_predict_folder.setEnabled(can_interact)

        if self._toolbar is not None:
            self._toolbar.render()

    def _set_running(self, running: bool) -> None:
        self._unet_run_btn.setEnabled(not running)
        self._unet_batch_btn.setEnabled(not running)
        self._sd_run_btn.setEnabled(not running)
        self._sd_batch_btn.setEnabled(not running)
        if hasattr(self, "_sd_isolate_btn"):
            self._sd_isolate_btn.setEnabled(not running)
        if self._active_stop_btn is not None:
            self._active_stop_btn.setEnabled(running)
        self._act_stop.setEnabled(running)
        self._refresh_ui_state()

    def _set_roi_selecting(self, selecting: bool) -> None:
        self._roi_selecting = bool(selecting)
        if selecting:
            self._unet_run_btn.setEnabled(False)
            self._sd_run_btn.setEnabled(False)
            if hasattr(self, "_sd_isolate_btn"):
                self._sd_isolate_btn.setEnabled(False)
            self.statusBar().showMessage("Select ROI: drag on Overlay tab. Esc = cancel. Click = full image.")
        else:
            self.statusBar().clearMessage()
            if self._thread is None:
                self._unet_run_btn.setEnabled(True)
                self._sd_run_btn.setEnabled(True)
                if hasattr(self, "_sd_isolate_btn"):
                    self._sd_isolate_btn.setEnabled(True)
        self._refresh_ui_state()
