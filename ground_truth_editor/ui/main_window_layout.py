from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .utils import browse_dir, browse_file


class MainWindowLayoutMixin:
    def _get_detections_for_image(self, image_path: str) -> list[dict]:
        if not image_path:
            return []
        if hasattr(self, "_history_view_detections"):
            dets = self._history_view_detections.get(image_path)
            if dets is not None:
                return dets
        if hasattr(self, "_image_detections"):
            return self._image_detections.get(image_path, [])
        return []

    def _build_layout(self) -> None:
        central = QtWidgets.QWidget(self)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_view_tabs())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 980])
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self.addAction(self._act_prev_image)
        self.addAction(self._act_next_image)
        self.addAction(self._act_focus_folder_filter)
        self.addAction(self._act_view_folder)
        self.addAction(self._act_view_overlay)
        self.addAction(self._act_view_image)
        self.addAction(self._act_view_mask)
        self.addAction(self._act_model_settings)
        self.addAction(self._act_settings_sam)
        self.addAction(self._act_settings_dino)
        self.addAction(self._act_settings_unet)
        self.addAction(self._act_stop)

    def _build_sidebar(self) -> QtWidgets.QWidget:
        sidebar = QtWidgets.QWidget(self)
        sidebar.setMinimumWidth(360)
        layout = QtWidgets.QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self._left_tabs = QtWidgets.QTabWidget(sidebar)
        self._left_tabs.setDocumentMode(True)
        self._left_tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)

        self._explorer_tab = self._build_tab_folder()
        self._history_tab = self._build_tab_history()
        self._image_tools_tab = self._build_tab_editor()
        self._mask_tab = self._build_tab_mask()

        self._left_tabs.addTab(self._explorer_tab, "Explorer")
        self._left_tabs.addTab(self._history_tab, "History")
        self._left_tabs.addTab(self._image_tools_tab, "Image Tools")
        self._left_tabs.addTab(self._mask_tab, "Mask")
        layout.addWidget(self._left_tabs, 1)
        return sidebar

    def _wrap_scroll(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _build_tab_editor(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self._image_tools_panel.setParent(tab)
        layout.addWidget(self._image_tools_panel, 1)
        return tab

    def _focus_history_tab(self) -> None:
        if hasattr(self, "_left_tabs") and hasattr(self, "_history_tab"):
            self._left_tabs.setCurrentWidget(self._history_tab)

    def _focus_mask_tab(self) -> None:
        if hasattr(self, "_left_tabs") and hasattr(self, "_mask_tab"):
            self._left_tabs.setCurrentWidget(self._mask_tab)

    def _build_tab_history(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(QtWidgets.QLabel("Workspace detect history:", tab))
        
        self._history_current_label = QtWidgets.QLabel("Current Run: None", tab)
        self._history_current_label.setWordWrap(True)
        layout.addWidget(self._history_current_label)

        self._history_list = QtWidgets.QListWidget(tab)
        self._history_list.itemClicked.connect(self._on_history_item_clicked)
        layout.addWidget(self._history_list, 1)

        return tab

    def _build_tab_mask(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Buttons Row
        # Buttons Row removed as requested

        # History List
        layout.addWidget(QtWidgets.QLabel("Image History Lists", tab))
        self._mask_history_tree = QtWidgets.QTreeWidget(tab)
        self._mask_history_tree.setHeaderLabels(["Run ID", "Model", "Score"])
        self._mask_history_tree.setRootIsDecorated(False)
        self._mask_history_tree.setAlternatingRowColors(True)
        self._mask_history_tree.itemClicked.connect(self._on_sidebar_history_clicked)
        layout.addWidget(self._mask_history_tree, 1)

        # Object/Mask List
        layout.addWidget(QtWidgets.QLabel("Mask List", tab))
        self._object_list = QtWidgets.QListWidget(tab)
        self._object_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self._object_list.itemSelectionChanged.connect(self._on_object_selection_changed)
        self._object_list.itemChanged.connect(self._on_object_item_changed)
        layout.addWidget(self._object_list, 2)

        return tab

    # _build_tab_object removed

    def _populate_mask_list(self) -> None:
        self._object_list.clear()
        if not self._state:
            return

        dets = self._get_detections_for_image(self._state.image_path)
        if not dets:
            self._update_composite_mask()
            return

        # Identify the latest run_id for this image to show only the most recent results by default
        latest_run_id = ""
        for det in dets:
            rid = str(det.get("run_id") or "")
            if rid > latest_run_id:
                latest_run_id = rid

        for i, det in enumerate(dets):
            label = det.get("label", "Object")
            score = float(det.get("score", 0.0))
            model_name = det.get("model_name", "Model")
            rid = str(det.get("run_id") or "")

            # Sanitizing label
            label_clean = label.replace(" ", "")

            text = f"{model_name}_{label_clean}_{score:.2f}"

            item = QtWidgets.QListWidgetItem(text)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            
            # Check if this item belongs to the latest run (or if we have no run info at all)
            # This ensures that when we click an image, we see the most recent detection, not a mess of all history.
            is_latest = (rid == latest_run_id) or (not latest_run_id)
            item.setCheckState(QtCore.Qt.CheckState.Checked if is_latest else QtCore.Qt.CheckState.Unchecked)
            
            item.setData(QtCore.Qt.UserRole, i)
            self._object_list.addItem(item)

        self._update_composite_mask()

    def _on_object_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        self._update_composite_mask()

    def _on_object_selection_changed(self) -> None:
        items = self._object_list.selectedItems()
        if not items:
            self._overlay_canvas.set_highlight_box(None)
            self._update_composite_mask()
            return

        # Highlight last selected

        item = items[-1]
        idx = int(item.data(QtCore.Qt.UserRole))
        if self._state:
            dets = self._get_detections_for_image(self._state.image_path)
            if 0 <= idx < len(dets):
                det = dets[idx]
                box = det.get("box")
                if box:
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    self._overlay_canvas.set_highlight_box(QtCore.QRectF(x1, y1, w, h))
                    self._view_tabs.setCurrentIndex(0)

        self._update_composite_mask()

    def _update_composite_mask(self) -> None:
        if not self._state:
            return

        dets = self._get_detections_for_image(self._state.image_path)
        if not dets:
            return
        import numpy as np
        import cv2

        merged = np.zeros((self._state.image_h, self._state.image_w), dtype=np.uint8)

        count = self._object_list.count()
        for i in range(count):
            item = self._object_list.item(i)
            if item is None:
                continue
            # Show masks based on checkbox state, not selection.
            if item.checkState() != QtCore.Qt.CheckState.Checked:
                continue

            idx = int(item.data(QtCore.Qt.UserRole))
            if 0 <= idx < len(dets):
                det = dets[idx]
                mask_b64 = det.get("mask_b64")
                mask = None
                if mask_b64:
                    import base64

                    mask_bytes = base64.b64decode(mask_b64)
                    arr = np.frombuffer(mask_bytes, np.uint8)
                    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                else:
                    mask_path = det.get("mask_path")
                    if mask_path:
                        import os
                        import base64

                        path = str(mask_path)
                        if hasattr(self, "_resolve_service_path"):
                            try:
                                path = self._resolve_service_path(path)
                            except Exception:
                                pass
                        if os.path.isfile(path):
                            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                            if mask is not None:
                                ok, png_bytes = cv2.imencode(".png", mask)
                                if ok:
                                    det["mask_b64"] = base64.b64encode(png_bytes.tobytes()).decode("ascii")

                if mask is not None:
                    if mask.shape != merged.shape:
                        mask = cv2.resize(
                            mask, (self._state.image_w, self._state.image_h), interpolation=cv2.INTER_NEAREST
                        )
                    merged = np.maximum(merged, mask)

        # Update canvas
        qimg = QtGui.QImage(
            merged.data, merged.shape[1], merged.shape[0], merged.strides[0], QtGui.QImage.Format.Format_Grayscale8
        )
        # Copy to avoid lifetime issues
        self._overlay_canvas.set_mask(qimg.copy())
        self._sync_mask_views()

    def _build_view_tabs(self) -> QtWidgets.QWidget:
        self._view_tabs = QtWidgets.QTabWidget(self)
        self._view_tabs.addTab(self._overlay_canvas, "Overlay")
        self._view_tabs.addTab(self._image_canvas, "Image")
        self._view_tabs.addTab(self._mask_canvas, "Mask")
        return self._view_tabs

    def _build_tab_folder(self) -> QtWidgets.QWidget:
        return self._explorer_panel

    def _build_tab_unet(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._unet_model_edit = QtWidgets.QLineEdit(tab)
        self._unet_model_edit.setText(str(Path("PredictTools/Unet/best_model.pth")))
        model_browse = QtWidgets.QPushButton("Browse...", tab)
        model_browse.clicked.connect(lambda: browse_file(self, self._unet_model_edit, "Model (*.pth);;All files (*)"))


        self._unet_threshold = QtWidgets.QDoubleSpinBox(tab)
        self._unet_threshold.setRange(0.0, 1.0)
        self._unet_threshold.setSingleStep(0.01)
        self._unet_threshold.setValue(0.5)

        self._unet_post = QtWidgets.QCheckBox("Apply postprocessing", tab)
        self._unet_post.setChecked(True)

        self._unet_mode = QtWidgets.QComboBox(tab)
        self._unet_mode.addItems(["tile", "letterbox", "resize"])

        self._unet_input_size = QtWidgets.QSpinBox(tab)
        self._unet_input_size.setRange(64, 4096)
        self._unet_input_size.setValue(256)

        self._unet_overlap = QtWidgets.QSpinBox(tab)
        self._unet_overlap.setRange(0, 4096)
        self._unet_overlap.setValue(0)

        self._unet_tile_batch = QtWidgets.QSpinBox(tab)
        self._unet_tile_batch.setRange(1, 64)
        self._unet_tile_batch.setValue(4)

        layout.addWidget(QtWidgets.QLabel("Model (.pth)", tab))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._unet_model_edit, 1)
        row.addWidget(model_browse)
        layout.addLayout(row)


        layout.addWidget(QtWidgets.QLabel("Threshold", tab))
        layout.addWidget(self._unet_threshold)

        layout.addWidget(QtWidgets.QLabel("Mode", tab))
        layout.addWidget(self._unet_mode)

        layout.addWidget(QtWidgets.QLabel("Input size", tab))
        layout.addWidget(self._unet_input_size)

        layout.addWidget(QtWidgets.QLabel("Tile overlap", tab))
        layout.addWidget(self._unet_overlap)

        layout.addWidget(QtWidgets.QLabel("Tile batch size", tab))
        layout.addWidget(self._unet_tile_batch)

        layout.addWidget(self._unet_post)

        btn_row = QtWidgets.QHBoxLayout()
        self._unet_run_btn = QtWidgets.QPushButton("Run UNet", tab)
        self._unet_run_btn.clicked.connect(self._run_unet)
        self._unet_batch_btn = QtWidgets.QPushButton("Batch (Folder)", tab)
        self._unet_batch_btn.clicked.connect(self._run_batch_unet)
        btn_row.addWidget(self._unet_run_btn)
        btn_row.addWidget(self._unet_batch_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)
        return tab

    def _build_tab_sam(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- SAM ---
        self._sd_sam_ckpt = QtWidgets.QLineEdit(tab)
        self._sd_sam_ckpt.setText(str(Path("PredictTools/SAM/sam_vit_h_4b8939.pth")))
        ckpt_browse = QtWidgets.QPushButton("Browse...", tab)
        ckpt_browse.clicked.connect(
            lambda: browse_file(self, self._sd_sam_ckpt, "SAM checkpoint (*.pth);;All files (*)")
        )

        self._sd_sam_type = QtWidgets.QComboBox(tab)
        self._sd_sam_type.addItems(["auto", "vit_b", "vit_l", "vit_h"])

        sam_group = QtWidgets.QGroupBox("SAM", tab)
        sam_layout = QtWidgets.QVBoxLayout(sam_group)
        sam_layout.setContentsMargins(10, 10, 10, 10)
        sam_layout.setSpacing(6)
        sam_layout.addWidget(QtWidgets.QLabel("SAM checkpoint", sam_group))
        sam_ckpt_row = QtWidgets.QHBoxLayout()
        sam_ckpt_row.addWidget(self._sd_sam_ckpt, 1)
        sam_ckpt_row.addWidget(ckpt_browse)
        sam_layout.addLayout(sam_ckpt_row)
        sam_layout.addWidget(QtWidgets.QLabel("SAM type", sam_group))
        sam_layout.addWidget(self._sd_sam_type)
        layout.addWidget(sam_group)

        # --- Delta fine-tune (Auto-detect from path) ---
        self._sd_delta_group = QtWidgets.QGroupBox("Fine-tune (delta) - Set path to enable", tab)
        self._sd_delta_group.setVisible(True)
        dg = QtWidgets.QVBoxLayout(self._sd_delta_group)
        dg.setContentsMargins(10, 10, 10, 10)
        dg.setSpacing(6)

        self._sd_delta_type = QtWidgets.QComboBox(self._sd_delta_group)
        self._sd_delta_type.addItems(["adapter", "lora", "both"])
        self._sd_delta_type.setCurrentText("lora")

        self._sd_delta_ckpt = QtWidgets.QLineEdit(self._sd_delta_group)
        self._sd_delta_ckpt.setPlaceholderText("auto or path/to/delta.pth")
        self._sd_delta_ckpt.setText("auto")
        delta_browse = QtWidgets.QPushButton("Browse...", self._sd_delta_group)
        delta_browse.clicked.connect(
            lambda: browse_file(self, self._sd_delta_ckpt, "Delta checkpoint (*.pth);;All files (*)")
        )

        self._sd_middle_dim = QtWidgets.QSpinBox(self._sd_delta_group)
        self._sd_middle_dim.setRange(1, 4096)
        self._sd_middle_dim.setValue(32)

        self._sd_scaling_factor = QtWidgets.QDoubleSpinBox(self._sd_delta_group)
        self._sd_scaling_factor.setRange(0.0, 10.0)
        self._sd_scaling_factor.setDecimals(4)
        self._sd_scaling_factor.setSingleStep(0.05)
        self._sd_scaling_factor.setValue(0.2)

        self._sd_rank = QtWidgets.QSpinBox(self._sd_delta_group)
        self._sd_rank.setRange(1, 1024)
        self._sd_rank.setValue(4)

        dg.addWidget(QtWidgets.QLabel("Delta type", self._sd_delta_group))
        dg.addWidget(self._sd_delta_type)

        dg.addWidget(QtWidgets.QLabel("Delta checkpoint", self._sd_delta_group))
        delta_row = QtWidgets.QHBoxLayout()
        delta_row.addWidget(self._sd_delta_ckpt, 1)
        delta_row.addWidget(delta_browse)
        dg.addLayout(delta_row)

        dg.addWidget(QtWidgets.QLabel("Adapter middle_dim", self._sd_delta_group))
        dg.addWidget(self._sd_middle_dim)

        dg.addWidget(QtWidgets.QLabel("Adapter scaling_factor", self._sd_delta_group))
        dg.addWidget(self._sd_scaling_factor)

        dg.addWidget(QtWidgets.QLabel("LoRA rank", self._sd_delta_group))
        dg.addWidget(self._sd_rank)

        layout.addWidget(self._sd_delta_group)

        # --- Mask postprocess ---
        self._sd_invert = QtWidgets.QCheckBox("Invert mask", tab)
        self._sd_invert.setChecked(False)

        self._sd_min_area = QtWidgets.QSpinBox(tab)
        self._sd_min_area.setRange(0, 10_000_000)
        self._sd_min_area.setValue(0)

        self._sd_dilate = QtWidgets.QSpinBox(tab)
        self._sd_dilate.setRange(0, 50)
        self._sd_dilate.setValue(0)

        mask_group = QtWidgets.QGroupBox("Mask postprocess", tab)
        mask_layout = QtWidgets.QVBoxLayout(mask_group)
        mask_layout.setContentsMargins(10, 10, 10, 10)
        mask_layout.setSpacing(6)
        mask_layout.addWidget(self._sd_invert)
        mask_layout.addWidget(QtWidgets.QLabel("Min component area", mask_group))
        mask_layout.addWidget(self._sd_min_area)
        mask_layout.addWidget(QtWidgets.QLabel("Dilate iters", mask_group))
        mask_layout.addWidget(self._sd_dilate)
        layout.addWidget(mask_group)

        self._sd_delta_type.currentTextChanged.connect(self._sync_delta_controls)
        self._sync_delta_controls(self._sd_delta_type.currentText())

        layout.addStretch(1)
        return tab

    def _build_tab_dino(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self._sd_gdino_ckpt = QtWidgets.QLineEdit(tab)
        self._sd_gdino_ckpt.setPlaceholderText("path/to/groundingdino.pth or HF model id")
        self._sd_gdino_ckpt.setText("IDEA-Research/grounding-dino-base")
        gdino_browse = QtWidgets.QPushButton("Browse...", tab)
        gdino_browse.clicked.connect(
            lambda: browse_file(self, self._sd_gdino_ckpt, "GroundingDINO checkpoint (*.pth);;All files (*)")
        )

        self._sd_gdino_cfg = QtWidgets.QComboBox(tab)
        self._sd_gdino_cfg.addItem("Auto (infer from filename)", "auto")
        self._sd_gdino_cfg.addItem("grounding-dino-base", "IDEA-Research/grounding-dino-base")
        self._sd_gdino_cfg.addItem("grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny")

        gdino_group = QtWidgets.QGroupBox("GroundingDINO", tab)
        gdino_layout = QtWidgets.QVBoxLayout(gdino_group)
        gdino_layout.setContentsMargins(10, 10, 10, 10)
        gdino_layout.setSpacing(6)
        gdino_layout.addWidget(QtWidgets.QLabel("Checkpoint / model id", gdino_group))
        gdino_row = QtWidgets.QHBoxLayout()
        gdino_row.addWidget(self._sd_gdino_ckpt, 1)
        gdino_row.addWidget(gdino_browse)
        gdino_layout.addLayout(gdino_row)
        gdino_layout.addWidget(QtWidgets.QLabel("Config", gdino_group))
        gdino_layout.addWidget(self._sd_gdino_cfg)

        self._sd_device = QtWidgets.QComboBox(tab)
        self._sd_device.addItems(["cpu", "auto", "cuda"])
        self._sd_device.setCurrentText("cpu")
        gdino_layout.addWidget(QtWidgets.QLabel("Device (SAM + DINO)", gdino_group))
        gdino_layout.addWidget(self._sd_device)
        layout.addWidget(gdino_group)

        self._sd_queries = QtWidgets.QLineEdit(tab)
        self._sd_queries.setText("crack,mold,stain,spall,damage,column")
        layout.addWidget(QtWidgets.QLabel("Text queries (comma)", tab))
        layout.addWidget(self._sd_queries)


        self._sd_box_thr = QtWidgets.QDoubleSpinBox(tab)
        self._sd_box_thr.setRange(0.0, 1.0)
        self._sd_box_thr.setSingleStep(0.01)
        self._sd_box_thr.setValue(0.25)

        self._sd_text_thr = QtWidgets.QDoubleSpinBox(tab)
        self._sd_text_thr.setRange(0.0, 1.0)
        self._sd_text_thr.setSingleStep(0.01)
        self._sd_text_thr.setValue(0.25)

        self._sd_max_dets = QtWidgets.QSpinBox(tab)
        self._sd_max_dets.setRange(0, 999)
        self._sd_max_dets.setValue(20)

        thr_group = QtWidgets.QGroupBox("Thresholds", tab)
        thr_layout = QtWidgets.QVBoxLayout(thr_group)
        thr_layout.setContentsMargins(10, 10, 10, 10)
        thr_layout.setSpacing(6)
        thr_layout.addWidget(QtWidgets.QLabel("Box thr", thr_group))
        thr_layout.addWidget(self._sd_box_thr)
        thr_layout.addWidget(QtWidgets.QLabel("Text thr", thr_group))
        thr_layout.addWidget(self._sd_text_thr)
        thr_layout.addWidget(QtWidgets.QLabel("Max dets", thr_group))
        thr_layout.addWidget(self._sd_max_dets)
        layout.addWidget(thr_group)


        btn_row = QtWidgets.QHBoxLayout()
        self._sd_run_btn = QtWidgets.QPushButton("Run SAM + DINO", tab)
        self._sd_run_btn.clicked.connect(self._run_sam_dino)
        self._sd_batch_btn = QtWidgets.QPushButton("Batch (Folder)", tab)
        self._sd_batch_btn.clicked.connect(self._run_batch_sam_dino)
        self._sd_isolate_btn = QtWidgets.QPushButton("TÃ¡ch váº­t thá»ƒ", tab)
        self._sd_isolate_btn.clicked.connect(self._isolate_object)
        btn_row.addWidget(self._sd_run_btn)
        btn_row.addWidget(self._sd_batch_btn)
        btn_row.addWidget(self._sd_isolate_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)
        return tab

    def _sync_delta_controls(self, delta_type: str) -> None:
        dt = str(delta_type).strip().lower()
        use_adapter = dt in {"adapter", "both"}
        use_lora = dt in {"lora", "both"}
        self._sd_middle_dim.setEnabled(use_adapter)
        self._sd_scaling_factor.setEnabled(use_adapter)
        self._sd_rank.setEnabled(use_lora)

