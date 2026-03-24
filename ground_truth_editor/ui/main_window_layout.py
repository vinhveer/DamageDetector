from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from color_utils import label_color
from .utils import browse_dir, browse_file


class MainWindowLayoutMixin:
    def _label_color(self, label: str) -> QtGui.QColor:
        return label_color(label)

    def _detection_key(self, det: dict) -> str:
        det_id = str(det.get("detection_id") or "").strip()
        if det_id:
            return det_id
        box = det.get("box") or []
        if isinstance(box, (list, tuple)) and len(box) == 4:
            box_text = ",".join(str(float(x)) for x in box)
        else:
            box_text = ""
        return "|".join(
            [
                str(det.get("run_id") or ""),
                str(det.get("model_name") or ""),
                str(det.get("label") or ""),
                str(det.get("score") or ""),
                box_text,
            ]
        )

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
        self._compare_tab = self._build_tab_compare()
        self._isolate_tab = self._build_tab_isolate()

        self._left_tabs.addTab(self._explorer_tab, "Explorer")
        self._left_tabs.addTab(self._history_tab, "History")
        self._left_tabs.addTab(self._image_tools_tab, "Image Tools")
        self._left_tabs.addTab(self._mask_tab, "Mask")
        self._left_tabs.addTab(self._compare_tab, "Compare")
        self._left_tabs.addTab(self._isolate_tab, "Isolate")
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

    def _build_tab_compare(self) -> QtWidgets.QWidget:
        from .features.compare_panel import ComparePanel
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._compare_panel = ComparePanel(tab)
        layout.addWidget(self._compare_panel)
        return tab

    def _build_tab_isolate(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(QtWidgets.QLabel("Isolated Objects:", tab))
        
        self._isolate_list = QtWidgets.QListWidget(tab)
        self._isolate_list.itemClicked.connect(self._on_isolate_item_clicked)
        layout.addWidget(self._isolate_list, 1)

        self._btn_refresh_isolate = QtWidgets.QPushButton("Refresh", tab)
        self._btn_refresh_isolate.clicked.connect(self._populate_isolate_list)
        layout.addWidget(self._btn_refresh_isolate)

        return tab

    def _populate_isolate_list(self) -> None:
        if not hasattr(self, "_isolate_list"):
            return
        self._isolate_list.clear()
        if not hasattr(self, "_iter_run_csv_files") or not hasattr(self, "_history_rows_from_csv"):
            return

        import os

        seen = set()
        isolate_items = []
        try:
            for csv_path in self._iter_run_csv_files():
                rows = self._history_rows_from_csv(csv_path)
                for row in rows:
                    isolate_path = ""
                    if hasattr(self, "_resolve_results_asset_path"):
                        isolate_path = self._resolve_results_asset_path(
                            row.get("isolate_rel") or "",
                            (row.get("run_id") or ""),
                        )
                    if not isolate_path or not os.path.isfile(isolate_path):
                        continue
                    key = (row.get("run_id") or "", isolate_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    isolate_items.append((row.get("run_id") or "-", isolate_path))
        except Exception:
            pass

        for run_id, isolate_path in isolate_items:
            item = QtWidgets.QListWidgetItem(f"{run_id} - {os.path.basename(isolate_path)}")
            item.setData(QtCore.Qt.UserRole, isolate_path)
            self._isolate_list.addItem(item)
            
    def _on_isolate_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return
        path = item.data(QtCore.Qt.UserRole)
        import os
        if path and os.path.isfile(path):
            self.load_image(path, switch_tab=True)

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
        list_header = QtWidgets.QHBoxLayout()
        list_header.addWidget(QtWidgets.QLabel("Mask List", tab))
        list_header.addStretch(1)
        self._btn_mask_tick_all = QtWidgets.QPushButton("Tick All", tab)
        self._btn_mask_tick_all.clicked.connect(self._check_all_mask_items)
        list_header.addWidget(self._btn_mask_tick_all)
        layout.addLayout(list_header)
        self._object_list = QtWidgets.QListWidget(tab)
        self._object_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self._object_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._object_list.itemSelectionChanged.connect(self._on_object_selection_changed)
        self._object_list.itemChanged.connect(self._on_object_item_changed)
        self._object_list.customContextMenuRequested.connect(self._show_object_list_context_menu)
        layout.addWidget(self._object_list, 2)

        return tab

    # _build_tab_object removed

    def _populate_mask_list(self) -> None:
        existing_checked: dict[str, bool] = {}
        if hasattr(self, "_state") and self._state and self._object_list.count() > 0:
            current_dets = self._get_detections_for_image(self._state.image_path)
            for i in range(self._object_list.count()):
                item = self._object_list.item(i)
                if item is None:
                    continue
                idx = int(item.data(QtCore.Qt.UserRole))
                if 0 <= idx < len(current_dets):
                    existing_checked[self._detection_key(current_dets[idx])] = (
                        item.checkState() == QtCore.Qt.CheckState.Checked
                    )

        blocker = QtCore.QSignalBlocker(self._object_list)
        self._object_list.clear()
        if not self._state:
            del blocker
            return

        dets = self._get_detections_for_image(self._state.image_path)
        if not dets:
            del blocker
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
            
            det_key = self._detection_key(det)
            is_latest = (rid == latest_run_id) or (not latest_run_id)
            checked = det.get("_checked")
            if checked is None:
                checked = existing_checked.get(det_key, is_latest)
            item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
            
            item.setData(QtCore.Qt.UserRole, i)
            self._object_list.addItem(item)

        del blocker
        self._update_composite_mask()

    def _on_object_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._state and item is not None:
            dets = self._get_detections_for_image(self._state.image_path)
            idx = int(item.data(QtCore.Qt.UserRole))
            if 0 <= idx < len(dets):
                dets[idx]["_checked"] = item.checkState() == QtCore.Qt.CheckState.Checked
        self._update_composite_mask()

    def _on_object_selection_changed(self) -> None:
        # Just update composite mask to show checked items, 
        # selection highlight is now disabled since we show boxes 
        # based on what is checked.
        pass

    def _check_all_mask_items(self) -> None:
        if not self._state:
            return

        dets = self._get_detections_for_image(self._state.image_path)
        blocker = QtCore.QSignalBlocker(self._object_list)
        for i in range(self._object_list.count()):
            item = self._object_list.item(i)
            if item is None:
                continue
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            idx = int(item.data(QtCore.Qt.UserRole))
            if 0 <= idx < len(dets):
                dets[idx]["_checked"] = True
        del blocker
        self._update_composite_mask()

    def _show_object_list_context_menu(self, pos) -> None:
        if not hasattr(self, "_object_list"):
            return
        item = self._object_list.itemAt(pos)
        if item is not None and not item.isSelected():
            self._object_list.clearSelection()
            item.setSelected(True)

        selected_items = self._object_list.selectedItems()
        if not selected_items:
            return

        menu = QtWidgets.QMenu(self._object_list)
        label = "Delete Selected Mask(s)" if len(selected_items) > 1 else "Delete Mask"
        act_delete = menu.addAction(label)
        chosen = menu.exec(self._object_list.mapToGlobal(pos))
        if chosen == act_delete:
            self._delete_selected_mask_items()

    def _delete_selected_mask_items(self) -> None:
        if not self._state:
            return
        selected_items = list(self._object_list.selectedItems())
        if not selected_items:
            return

        text = "Delete selected masks?" if len(selected_items) > 1 else "Delete selected mask?"
        resp = QtWidgets.QMessageBox.question(
            self,
            "Delete Mask",
            text,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        image_path = self._state.image_path
        current_dets = self._get_detections_for_image(image_path)
        selected_dets = []
        selected_keys = set()
        selected_indices = []
        for item in selected_items:
            idx = int(item.data(QtCore.Qt.UserRole))
            if 0 <= idx < len(current_dets):
                det = current_dets[idx]
                selected_dets.append(det)
                selected_keys.add(self._detection_key(det))
                selected_indices.append(idx)

        if not selected_dets:
            return

        if hasattr(self, "_run_with_loading"):
            def do_delete():
                for det in selected_dets:
                    if hasattr(self, "_delete_detection_persisted"):
                        self._delete_detection_persisted(det)
            self._run_with_loading("Delete Mask", "Deleting selected mask(s)...", do_delete)
        else:
            for det in selected_dets:
                if hasattr(self, "_delete_detection_persisted"):
                    self._delete_detection_persisted(det)

        def _filter_dets(store: dict[str, list[dict]]) -> None:
            dets = list(store.get(image_path, []))
            dets = [det for det in dets if self._detection_key(det) not in selected_keys]
            if dets:
                store[image_path] = dets
            elif image_path in store:
                del store[image_path]

        if hasattr(self, "_image_detections"):
            _filter_dets(self._image_detections)
        if hasattr(self, "_image_detections_all"):
            _filter_dets(self._image_detections_all)
        if hasattr(self, "_history_view_detections"):
            _filter_dets(self._history_view_detections)

        self._populate_mask_list()

    def _update_composite_mask(self) -> None:
        if not self._state:
            return

        dets = self._get_detections_for_image(self._state.image_path)
        if not dets:
            # Clear everything if no detections exist
            qimg = QtGui.QImage(self._state.image_w, self._state.image_h, QtGui.QImage.Format.Format_Grayscale8)
            qimg.fill(0)
            self._overlay_canvas.set_mask(qimg)
            self._overlay_canvas.set_overlay_visual(QtGui.QImage())
            if hasattr(self, "_active_highlight_boxes"):
                self._active_highlight_boxes = []
            self._overlay_canvas.set_highlight_boxes([])
            self._image_canvas.set_highlight_boxes([])
            if hasattr(self, "_mask_canvas") and getattr(self, "_mask_canvas", None) is not None:
                self._mask_canvas.set_highlight_boxes([])
            self._sync_mask_views()
            return
            
        import numpy as np
        import cv2

        merged = np.zeros((self._state.image_h, self._state.image_w), dtype=np.uint8)
        overlay_rgba = np.zeros((self._state.image_h, self._state.image_w, 4), dtype=np.uint8)
        
        active_boxes = []

        count = self._object_list.count()
        for i in range(count):
            item = self._object_list.item(i)
            if item is None:
                continue
            # Show masks based on checkbox state, not selection.
            if item.checkState() != QtCore.Qt.CheckState.Checked:
                continue

            idx = int(item.data(QtCore.Qt.UserRole))
            if idx >= 0 and idx < len(dets):
                det = dets[idx]
                
                # Add to active boxes to show
                box = det.get("box")
                label = det.get("label", "")
                score = det.get("score", 0.0)
                display_text = f"{label} {score:.2f}" if label else ""
                
                if box and len(box) == 4:
                    from PySide6.QtCore import QRectF
                    # Ensure coordinates are proper float
                    hb = QRectF(float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1]))
                    active_boxes.append((hb, display_text))
                    
                mask_b64 = det.get("mask_b64")
                mask = det.get("_mask_cache")
                if mask is None:
                    if mask_b64:
                        import base64

                        mask_bytes = base64.b64decode(mask_b64)
                        arr = np.frombuffer(mask_bytes, np.uint8)
                        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    else:
                        mask_path = det.get("mask_path")
                        if mask_path:
                            import os

                            path = str(mask_path)
                            if hasattr(self, "_resolve_service_path"):
                                try:
                                    path = self._resolve_service_path(path)
                                except Exception:
                                    pass
                            if os.path.isfile(path):
                                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if mask is not None:
                    if mask.shape != merged.shape:
                        mask = cv2.resize(
                            mask, (self._state.image_w, self._state.image_h), interpolation=cv2.INTER_NEAREST
                        )
                    det["_mask_cache"] = mask
                    merged = np.maximum(merged, mask)
                    
                    # Dùng đúng display_text như box để màu khớp
                    color = self._label_color(display_text)
                    mask_on = mask > 0
                    overlay_rgba[mask_on, 0] = color.red()
                    overlay_rgba[mask_on, 1] = color.green()
                    overlay_rgba[mask_on, 2] = color.blue()
                    overlay_rgba[mask_on, 3] = 255

        # Update canvas
        qimg = QtGui.QImage(self._state.image_w, self._state.image_h, QtGui.QImage.Format.Format_Grayscale8)
        qimg.fill(0)
        bpl = qimg.bytesPerLine()
        ptr = qimg.bits()
        # Create a numpy view into the QImage's buffer, taking care of row padding (bpl)
        view = np.frombuffer(ptr, dtype=np.uint8).reshape((self._state.image_h, bpl))
        # Copy our merged array into the image buffer
        view[:, :self._state.image_w] = merged

        overlay_qimg = QtGui.QImage(
            overlay_rgba.data,
            self._state.image_w,
            self._state.image_h,
            overlay_rgba.strides[0],
            QtGui.QImage.Format.Format_RGBA8888,
        ).copy()

        self._overlay_canvas.set_mask(qimg.copy())
        self._overlay_canvas.set_overlay_visual(overlay_qimg)
        
        if hasattr(self, "_active_highlight_boxes"):
            self._active_highlight_boxes = active_boxes
        self._overlay_canvas.set_highlight_boxes(active_boxes)
        self._image_canvas.set_highlight_boxes(active_boxes)
        if hasattr(self, "_mask_canvas") and getattr(self, "_mask_canvas", None) is not None:
            self._mask_canvas.set_highlight_boxes(active_boxes)
        
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
        self._unet_input_size.setValue(512)

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

        workflow_group = QtWidgets.QGroupBox("Inference Workflows", tab)
        workflow_layout = QtWidgets.QGridLayout(workflow_group)
        workflow_layout.setContentsMargins(10, 10, 10, 10)
        workflow_layout.setHorizontalSpacing(8)
        workflow_layout.setVerticalSpacing(8)

        self._unet_run_btn = QtWidgets.QPushButton("Run UNet Only", workflow_group)
        self._unet_run_btn.clicked.connect(self._run_unet)
        self._unet_batch_btn = QtWidgets.QPushButton("Batch UNet Only", workflow_group)
        self._unet_batch_btn.clicked.connect(self._run_batch_unet)
        self._unet_dino_run_btn = QtWidgets.QPushButton("Run UNet + DINO", workflow_group)
        self._unet_dino_run_btn.clicked.connect(self._run_unet_dino)
        self._unet_dino_batch_btn = QtWidgets.QPushButton("Batch UNet + DINO", workflow_group)
        self._unet_dino_batch_btn.clicked.connect(self._run_batch_unet_dino)

        workflow_layout.addWidget(self._unet_run_btn, 0, 0)
        workflow_layout.addWidget(self._unet_batch_btn, 0, 1)
        workflow_layout.addWidget(self._unet_dino_run_btn, 1, 0)
        workflow_layout.addWidget(self._unet_dino_batch_btn, 1, 1)
        layout.addWidget(workflow_group)

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
        self._sd_device.addItems(["auto", "cuda", "mps", "cpu"])
        self._sd_device.setCurrentText("auto")
        gdino_layout.addWidget(QtWidgets.QLabel("Device (default auto: cuda -> mps -> cpu)", gdino_group))
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


        workflow_group = QtWidgets.QGroupBox("Inference Workflows", tab)
        workflow_layout = QtWidgets.QGridLayout(workflow_group)
        workflow_layout.setContentsMargins(10, 10, 10, 10)
        workflow_layout.setHorizontalSpacing(8)
        workflow_layout.setVerticalSpacing(8)

        self._sd_run_btn = QtWidgets.QPushButton("Run SAM + DINO", workflow_group)
        self._sd_run_btn.clicked.connect(self._run_sam_dino)
        self._sd_batch_btn = QtWidgets.QPushButton("Batch SAM + DINO", workflow_group)
        self._sd_batch_btn.clicked.connect(self._run_batch_sam_dino)
        self._sd_ft_run_btn = QtWidgets.QPushButton("Run SAM + DINO + FT", workflow_group)
        self._sd_ft_run_btn.clicked.connect(lambda: self._run_sam_dino(force_use_delta=True))
        self._sd_ft_batch_btn = QtWidgets.QPushButton("Batch SAM + DINO + FT", workflow_group)
        self._sd_ft_batch_btn.clicked.connect(lambda: self._run_batch_sam_dino(force_use_delta=True))
        self._sam_only_run_btn = QtWidgets.QPushButton("Run SAM Only", workflow_group)
        self._sam_only_run_btn.clicked.connect(self._run_sam_only)
        self._sam_only_batch_btn = QtWidgets.QPushButton("Batch SAM Only", workflow_group)
        self._sam_only_batch_btn.clicked.connect(self._run_batch_sam_only)
        self._sam_only_ft_run_btn = QtWidgets.QPushButton("Run SAM Only + FT", workflow_group)
        self._sam_only_ft_run_btn.clicked.connect(lambda: self._run_sam_only(force_use_delta=True))
        self._sam_only_ft_batch_btn = QtWidgets.QPushButton("Batch SAM Only + FT", workflow_group)
        self._sam_only_ft_batch_btn.clicked.connect(lambda: self._run_batch_sam_only(force_use_delta=True))
        self._sam_tiled_run_btn = QtWidgets.QPushButton("Run SAM + DINO Tiled", workflow_group)
        self._sam_tiled_run_btn.clicked.connect(self._run_sam_tiled)
        self._sam_tiled_batch_btn = QtWidgets.QPushButton("Batch SAM + DINO Tiled", workflow_group)
        self._sam_tiled_batch_btn.clicked.connect(self._run_batch_sam_tiled)
        self._sd_isolate_btn = QtWidgets.QPushButton("Isolate Object", workflow_group)
        self._sd_isolate_btn.clicked.connect(self._isolate_object)

        workflow_layout.addWidget(self._sd_run_btn, 0, 0)
        workflow_layout.addWidget(self._sd_batch_btn, 0, 1)
        workflow_layout.addWidget(self._sd_ft_run_btn, 1, 0)
        workflow_layout.addWidget(self._sd_ft_batch_btn, 1, 1)
        workflow_layout.addWidget(self._sam_only_run_btn, 2, 0)
        workflow_layout.addWidget(self._sam_only_batch_btn, 2, 1)
        workflow_layout.addWidget(self._sam_only_ft_run_btn, 3, 0)
        workflow_layout.addWidget(self._sam_only_ft_batch_btn, 3, 1)
        workflow_layout.addWidget(self._sam_tiled_run_btn, 4, 0)
        workflow_layout.addWidget(self._sam_tiled_batch_btn, 4, 1)
        workflow_layout.addWidget(self._sd_isolate_btn, 5, 0, 1, 2)
        layout.addWidget(workflow_group)

        layout.addStretch(1)
        return tab

    def _sync_delta_controls(self, delta_type: str) -> None:
        dt = str(delta_type).strip().lower()
        use_adapter = dt in {"adapter", "both"}
        use_lora = dt in {"lora", "both"}
        self._sd_middle_dim.setEnabled(use_adapter)
        self._sd_scaling_factor.setEnabled(use_adapter)
        self._sd_rank.setEnabled(use_lora)
