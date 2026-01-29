import os
import sys

import numpy as np

from ui.qt import QtCore, QtGui, QtWidgets
from ui.dialogs import GroundTruthEditorDialog, RoiSelectionDialog
from ui.utils import _gray_to_qimage, _qimage_to_gray_np, _rgb_to_qimage
from ui.widgets import CtrlZoomGraphicsView
from ui.workers import PredictWorker, SinglePredictWorker

from predict_lib import _iter_images, diff_rgb, mask_metrics


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crack Detection U-Net - Prediction UI")

        self._thread = None
        self._worker = None
        self._active_mode = None
        self._single_last_details = None
        self._single_last_image_path = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        single_page = QtWidgets.QWidget()
        folder_page = QtWidgets.QWidget()
        compare_page = QtWidgets.QWidget()
        self.tabs.addTab(single_page, "Single image")
        self.tabs.addTab(folder_page, "Folder")
        self.tabs.addTab(compare_page, "Compare")

        # ---- Single image tab ----
        single_root = QtWidgets.QVBoxLayout(single_page)

        single_paths_group = QtWidgets.QGroupBox("Paths")
        single_paths_layout = QtWidgets.QGridLayout(single_paths_group)
        single_root.addWidget(single_paths_group)

        self.single_image_edit = QtWidgets.QLineEdit()
        self.single_gt_mask_edit = QtWidgets.QLineEdit()
        self.single_model_edit = QtWidgets.QLineEdit("output_results/best_model.pth")
        self.single_output_dir_edit = QtWidgets.QLineEdit("results")

        single_image_btn = QtWidgets.QPushButton("Browse...")
        single_gt_btn = QtWidgets.QPushButton("Browse...")
        single_model_btn = QtWidgets.QPushButton("Browse...")
        single_output_btn = QtWidgets.QPushButton("Browse...")

        single_paths_layout.addWidget(QtWidgets.QLabel("Image"), 0, 0)
        single_paths_layout.addWidget(self.single_image_edit, 0, 1)
        single_paths_layout.addWidget(single_image_btn, 0, 2)

        single_paths_layout.addWidget(QtWidgets.QLabel("GT mask (optional)"), 1, 0)
        single_paths_layout.addWidget(self.single_gt_mask_edit, 1, 1)
        single_paths_layout.addWidget(single_gt_btn, 1, 2)

        single_paths_layout.addWidget(QtWidgets.QLabel("Model weights (.pth)"), 2, 0)
        single_paths_layout.addWidget(self.single_model_edit, 2, 1)
        single_paths_layout.addWidget(single_model_btn, 2, 2)

        single_paths_layout.addWidget(QtWidgets.QLabel("Output folder"), 3, 0)
        single_paths_layout.addWidget(self.single_output_dir_edit, 3, 1)
        single_paths_layout.addWidget(single_output_btn, 3, 2)

        single_opts_group = QtWidgets.QGroupBox("Options")
        single_opts_layout = QtWidgets.QGridLayout(single_opts_group)
        single_root.addWidget(single_opts_group)

        self.single_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.single_threshold_spin.setRange(0.0, 1.0)
        self.single_threshold_spin.setSingleStep(0.05)
        self.single_threshold_spin.setDecimals(3)
        self.single_threshold_spin.setValue(0.5)

        self.single_mode_combo = QtWidgets.QComboBox()
        self.single_mode_combo.addItems(["tile", "letterbox", "resize"])
        self.single_mode_combo.setCurrentText("tile")

        self.single_input_size_spin = QtWidgets.QSpinBox()
        self.single_input_size_spin.setRange(32, 4096)
        self.single_input_size_spin.setValue(256)

        self.single_tile_overlap_spin = QtWidgets.QSpinBox()
        self.single_tile_overlap_spin.setRange(0, 4096)
        self.single_tile_overlap_spin.setValue(0)
        self.single_tile_overlap_spin.setToolTip("0 = auto (input_size//2)")

        self.single_tile_batch_spin = QtWidgets.QSpinBox()
        self.single_tile_batch_spin.setRange(1, 128)
        self.single_tile_batch_spin.setValue(4)

        self.single_no_post_check = QtWidgets.QCheckBox("Disable post-processing")

        single_opts_layout.addWidget(QtWidgets.QLabel("Threshold"), 0, 0)
        single_opts_layout.addWidget(self.single_threshold_spin, 0, 1)
        single_opts_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 2)
        single_opts_layout.addWidget(self.single_mode_combo, 0, 3)

        single_opts_layout.addWidget(QtWidgets.QLabel("Input size"), 1, 0)
        single_opts_layout.addWidget(self.single_input_size_spin, 1, 1)
        single_opts_layout.addWidget(QtWidgets.QLabel("Tile overlap"), 1, 2)
        single_opts_layout.addWidget(self.single_tile_overlap_spin, 1, 3)

        single_opts_layout.addWidget(QtWidgets.QLabel("Tile batch size"), 2, 0)
        single_opts_layout.addWidget(self.single_tile_batch_spin, 2, 1)
        single_opts_layout.addWidget(self.single_no_post_check, 2, 3)

        single_run_row = QtWidgets.QHBoxLayout()
        single_root.addLayout(single_run_row)
        self.single_start_btn = QtWidgets.QPushButton("Start")
        self.single_stop_btn = QtWidgets.QPushButton("Stop")
        self.single_create_gt_btn = QtWidgets.QPushButton("Create Ground Truth")
        self.single_create_gt_btn.setEnabled(False)
        self.single_stop_btn.setEnabled(False)
        self.single_progress = QtWidgets.QProgressBar()
        self.single_progress.setRange(0, 1)
        self.single_progress.setValue(0)
        single_run_row.addWidget(self.single_start_btn)
        single_run_row.addWidget(self.single_stop_btn)
        single_run_row.addWidget(self.single_create_gt_btn)
        single_run_row.addWidget(self.single_progress, 1)

        single_result_group = QtWidgets.QGroupBox("Result")
        single_result_layout = QtWidgets.QGridLayout(single_result_group)
        single_root.addWidget(single_result_group)
        self.single_out_path_label = QtWidgets.QLabel("-")
        self.single_dice_label = QtWidgets.QLabel("-")
        single_result_layout.addWidget(QtWidgets.QLabel("Output image"), 0, 0)
        single_result_layout.addWidget(self.single_out_path_label, 0, 1)
        single_result_layout.addWidget(QtWidgets.QLabel("Dice"), 1, 0)
        single_result_layout.addWidget(self.single_dice_label, 1, 1)

        self.single_log_box = QtWidgets.QPlainTextEdit()
        self.single_log_box.setReadOnly(True)
        single_root.addWidget(self.single_log_box, 1)

        # ---- Folder tab (existing UI) ----
        folder_root = QtWidgets.QVBoxLayout(folder_page)

        paths_group = QtWidgets.QGroupBox("Paths")
        paths_layout = QtWidgets.QGridLayout(paths_group)
        folder_root.addWidget(paths_group)

        self.input_dir_edit = QtWidgets.QLineEdit()
        self.gt_dir_edit = QtWidgets.QLineEdit()
        self.model_edit = QtWidgets.QLineEdit("output_results/best_model.pth")
        self.output_dir_edit = QtWidgets.QLineEdit("results")

        input_btn = QtWidgets.QPushButton("Browse...")
        gt_btn = QtWidgets.QPushButton("Browse...")
        model_btn = QtWidgets.QPushButton("Browse...")
        output_btn = QtWidgets.QPushButton("Browse...")

        paths_layout.addWidget(QtWidgets.QLabel("Input images folder"), 0, 0)
        paths_layout.addWidget(self.input_dir_edit, 0, 1)
        paths_layout.addWidget(input_btn, 0, 2)

        paths_layout.addWidget(QtWidgets.QLabel("GT masks folder (optional)"), 1, 0)
        paths_layout.addWidget(self.gt_dir_edit, 1, 1)
        paths_layout.addWidget(gt_btn, 1, 2)

        paths_layout.addWidget(QtWidgets.QLabel("Model weights (.pth)"), 2, 0)
        paths_layout.addWidget(self.model_edit, 2, 1)
        paths_layout.addWidget(model_btn, 2, 2)

        paths_layout.addWidget(QtWidgets.QLabel("Output folder"), 3, 0)
        paths_layout.addWidget(self.output_dir_edit, 3, 1)
        paths_layout.addWidget(output_btn, 3, 2)

        opts_group = QtWidgets.QGroupBox("Options")
        opts_layout = QtWidgets.QGridLayout(opts_group)
        folder_root.addWidget(opts_group)

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setValue(0.5)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["tile", "letterbox", "resize"])
        self.mode_combo.setCurrentText("tile")

        self.input_size_spin = QtWidgets.QSpinBox()
        self.input_size_spin.setRange(32, 4096)
        self.input_size_spin.setValue(256)

        self.tile_overlap_spin = QtWidgets.QSpinBox()
        self.tile_overlap_spin.setRange(0, 4096)
        self.tile_overlap_spin.setValue(0)
        self.tile_overlap_spin.setToolTip("0 = auto (input_size//2)")

        self.tile_batch_spin = QtWidgets.QSpinBox()
        self.tile_batch_spin.setRange(1, 128)
        self.tile_batch_spin.setValue(4)

        self.recursive_check = QtWidgets.QCheckBox("Recursive")
        self.no_post_check = QtWidgets.QCheckBox("Disable post-processing")

        opts_layout.addWidget(QtWidgets.QLabel("Threshold"), 0, 0)
        opts_layout.addWidget(self.threshold_spin, 0, 1)

        opts_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 2)
        opts_layout.addWidget(self.mode_combo, 0, 3)

        opts_layout.addWidget(QtWidgets.QLabel("Input size"), 1, 0)
        opts_layout.addWidget(self.input_size_spin, 1, 1)

        opts_layout.addWidget(QtWidgets.QLabel("Tile overlap"), 1, 2)
        opts_layout.addWidget(self.tile_overlap_spin, 1, 3)

        opts_layout.addWidget(QtWidgets.QLabel("Tile batch size"), 2, 0)
        opts_layout.addWidget(self.tile_batch_spin, 2, 1)

        opts_layout.addWidget(self.recursive_check, 2, 2)
        opts_layout.addWidget(self.no_post_check, 2, 3)

        run_row = QtWidgets.QHBoxLayout()
        folder_root.addLayout(run_row)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        run_row.addWidget(self.start_btn)
        run_row.addWidget(self.stop_btn)
        run_row.addWidget(self.progress, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        folder_root.addWidget(splitter, 1)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "Image", "GT mask", "Dice", "Output", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        splitter.addWidget(self.table)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        splitter.addWidget(self.log_box)
        splitter.setSizes([400, 200])

        # ---- Compare tab ----
        compare_root = QtWidgets.QVBoxLayout(compare_page)

        compare_paths = QtWidgets.QGroupBox("Masks")
        compare_paths_layout = QtWidgets.QGridLayout(compare_paths)
        compare_root.addWidget(compare_paths)

        self.compare_pred_mask_edit = QtWidgets.QLineEdit()
        self.compare_gt_mask_edit = QtWidgets.QLineEdit()
        compare_pred_btn = QtWidgets.QPushButton("Browse...")
        compare_gt_btn = QtWidgets.QPushButton("Browse...")
        self.compare_use_last_btn = QtWidgets.QPushButton("Use last single prediction")
        self.compare_compute_btn = QtWidgets.QPushButton("Compute")

        compare_paths_layout.addWidget(QtWidgets.QLabel("Predicted mask"), 0, 0)
        compare_paths_layout.addWidget(self.compare_pred_mask_edit, 0, 1)
        compare_paths_layout.addWidget(compare_pred_btn, 0, 2)

        compare_paths_layout.addWidget(QtWidgets.QLabel("Ground truth mask"), 1, 0)
        compare_paths_layout.addWidget(self.compare_gt_mask_edit, 1, 1)
        compare_paths_layout.addWidget(compare_gt_btn, 1, 2)

        btn_row = QtWidgets.QHBoxLayout()
        compare_root.addLayout(btn_row)
        btn_row.addWidget(self.compare_use_last_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.compare_compute_btn)

        metrics_group = QtWidgets.QGroupBox("Metrics")
        metrics_layout = QtWidgets.QGridLayout(metrics_group)
        compare_root.addWidget(metrics_group)

        self.compare_info_label = QtWidgets.QLabel("-")
        self.compare_dice_label = QtWidgets.QLabel("-")
        self.compare_iou_label = QtWidgets.QLabel("-")
        self.compare_precision_label = QtWidgets.QLabel("-")
        self.compare_recall_label = QtWidgets.QLabel("-")
        self.compare_accuracy_label = QtWidgets.QLabel("-")
        self.compare_specificity_label = QtWidgets.QLabel("-")
        self.compare_conf_label = QtWidgets.QLabel("-")

        metrics_layout.addWidget(QtWidgets.QLabel("Info"), 0, 0)
        metrics_layout.addWidget(self.compare_info_label, 0, 1, 1, 3)

        metrics_layout.addWidget(QtWidgets.QLabel("Dice"), 1, 0)
        metrics_layout.addWidget(self.compare_dice_label, 1, 1)
        metrics_layout.addWidget(QtWidgets.QLabel("IoU"), 1, 2)
        metrics_layout.addWidget(self.compare_iou_label, 1, 3)

        metrics_layout.addWidget(QtWidgets.QLabel("Precision"), 2, 0)
        metrics_layout.addWidget(self.compare_precision_label, 2, 1)
        metrics_layout.addWidget(QtWidgets.QLabel("Recall"), 2, 2)
        metrics_layout.addWidget(self.compare_recall_label, 2, 3)

        metrics_layout.addWidget(QtWidgets.QLabel("Accuracy"), 3, 0)
        metrics_layout.addWidget(self.compare_accuracy_label, 3, 1)
        metrics_layout.addWidget(QtWidgets.QLabel("Specificity"), 3, 2)
        metrics_layout.addWidget(self.compare_specificity_label, 3, 3)

        metrics_layout.addWidget(QtWidgets.QLabel("TP/FP/FN/TN"), 4, 0)
        metrics_layout.addWidget(self.compare_conf_label, 4, 1, 1, 3)

        preview_group = QtWidgets.QGroupBox("Preview (Ctrl+Wheel = zoom)")
        preview_layout = QtWidgets.QHBoxLayout(preview_group)
        compare_root.addWidget(preview_group, 1)

        def _make_preview(title: str):
            box = QtWidgets.QGroupBox(title)
            v = QtWidgets.QVBoxLayout(box)
            view = CtrlZoomGraphicsView()
            scene = QtWidgets.QGraphicsScene(view)
            view.setScene(scene)
            view.setMinimumHeight(220)
            v.addWidget(view, 1)
            return box, view, scene

        box1, self.compare_pred_view, self.compare_pred_scene = _make_preview("Pred mask")
        box2, self.compare_gt_view, self.compare_gt_scene = _make_preview("GT mask")
        box3, self.compare_diff_view, self.compare_diff_scene = _make_preview("Diff (TP=green FP=red FN=blue)")
        preview_layout.addWidget(box1, 1)
        preview_layout.addWidget(box2, 1)
        preview_layout.addWidget(box3, 1)

        input_btn.clicked.connect(self._browse_input)
        gt_btn.clicked.connect(self._browse_gt)
        model_btn.clicked.connect(self._browse_model)
        output_btn.clicked.connect(self._browse_output)
        self.start_btn.clicked.connect(self._start_folder)
        self.stop_btn.clicked.connect(self._stop)

        single_image_btn.clicked.connect(self._browse_single_image)
        single_gt_btn.clicked.connect(self._browse_single_gt)
        single_model_btn.clicked.connect(self._browse_single_model)
        single_output_btn.clicked.connect(self._browse_single_output)
        self.single_start_btn.clicked.connect(self._start_single)
        self.single_stop_btn.clicked.connect(self._stop)
        self.single_create_gt_btn.clicked.connect(self._open_gt_editor)

        compare_pred_btn.clicked.connect(self._browse_compare_pred)
        compare_gt_btn.clicked.connect(self._browse_compare_gt)
        self.compare_use_last_btn.clicked.connect(self._compare_use_last)
        self.compare_compute_btn.clicked.connect(self._compare_compute)

    def _append_log(self, text: str):
        self.log_box.appendPlainText(text)

    def _append_single_log(self, text: str):
        self.single_log_box.appendPlainText(text)

    def _browse_input(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input images folder")
        if d:
            self.input_dir_edit.setText(d)

    def _browse_gt(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select ground-truth masks folder (optional)")
        if d:
            self.gt_dir_edit.setText(d)

    def _browse_model(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select model weights", filter="Model (*.pth);;All files (*)")
        if f:
            self.model_edit.setText(f)

    def _browse_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.output_dir_edit.setText(d)

    def _browse_single_image(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select image",
            filter="Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All files (*)",
        )
        if f:
            self.single_image_edit.setText(f)

    def _browse_single_gt(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select GT mask (optional)",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)",
        )
        if f:
            self.single_gt_mask_edit.setText(f)

    def _browse_single_model(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model weights", filter="Model (*.pth);;All files (*)"
        )
        if f:
            self.single_model_edit.setText(f)

    def _browse_single_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.single_output_dir_edit.setText(d)

    def _browse_compare_pred(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select predicted mask",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)",
        )
        if f:
            self.compare_pred_mask_edit.setText(f)

    def _browse_compare_gt(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ground truth mask",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)",
        )
        if f:
            self.compare_gt_mask_edit.setText(f)

    def _compare_use_last(self):
        if self._single_last_details:
            mask_path = self._single_last_details.get("mask_path")
            if mask_path:
                self.compare_pred_mask_edit.setText(mask_path)
        gt_path = self.single_gt_mask_edit.text().strip()
        if gt_path:
            self.compare_gt_mask_edit.setText(gt_path)
        self.tabs.setCurrentIndex(2)

    def _compare_compute(self):
        pred_path = self.compare_pred_mask_edit.text().strip()
        gt_path = self.compare_gt_mask_edit.text().strip()

        if not pred_path or not os.path.exists(pred_path):
            QtWidgets.QMessageBox.critical(self, "Compare", f"Predicted mask not found:\n{pred_path}")
            return
        if not gt_path or not os.path.exists(gt_path):
            QtWidgets.QMessageBox.critical(self, "Compare", f"Ground truth mask not found:\n{gt_path}")
            return

        pred_img = QtGui.QImage(pred_path)
        gt_img = QtGui.QImage(gt_path)
        if pred_img.isNull():
            QtWidgets.QMessageBox.critical(self, "Compare", f"Failed to load predicted mask:\n{pred_path}")
            return
        if gt_img.isNull():
            QtWidgets.QMessageBox.critical(self, "Compare", f"Failed to load ground truth mask:\n{gt_path}")
            return

        pred_gray = _qimage_to_gray_np(pred_img)
        gt_gray = _qimage_to_gray_np(gt_img)

        info = []
        if gt_gray.shape != pred_gray.shape:
            info.append(f"Resized GT {gt_gray.shape[1]}x{gt_gray.shape[0]} -> {pred_gray.shape[1]}x{pred_gray.shape[0]}")
            gt_img2 = gt_img.convertToFormat(QtGui.QImage.Format_Grayscale8).scaled(
                pred_img.width(),
                pred_img.height(),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation,
            )
            gt_gray = _qimage_to_gray_np(gt_img2)

        pred_bin = pred_gray > 127
        gt_bin = gt_gray > 127

        m = mask_metrics(pred_bin, gt_bin)
        self.compare_info_label.setText("; ".join(info) if info else f"Size: {pred_gray.shape[1]}x{pred_gray.shape[0]}")
        self.compare_dice_label.setText(f"{m['dice']:.4f}")
        self.compare_iou_label.setText(f"{m['iou']:.4f}")
        self.compare_precision_label.setText(f"{m['precision']:.4f}")
        self.compare_recall_label.setText(f"{m['recall']:.4f}")
        self.compare_accuracy_label.setText(f"{m['accuracy']:.4f}")
        self.compare_specificity_label.setText(f"{m['specificity']:.4f}")
        self.compare_conf_label.setText(
            f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}  (total={m['total']})"
        )

        pred_show = pred_bin.astype(np.uint8) * 255
        gt_show = gt_bin.astype(np.uint8) * 255
        diff = diff_rgb(pred_bin, gt_bin)

        self.compare_pred_scene.clear()
        self.compare_gt_scene.clear()
        self.compare_diff_scene.clear()

        self.compare_pred_scene.addPixmap(QtGui.QPixmap.fromImage(_gray_to_qimage(pred_show)))
        self.compare_gt_scene.addPixmap(QtGui.QPixmap.fromImage(_gray_to_qimage(gt_show)))
        self.compare_diff_scene.addPixmap(QtGui.QPixmap.fromImage(_rgb_to_qimage(diff)))

        self.compare_pred_view.fitInView(self.compare_pred_scene.itemsBoundingRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.compare_gt_view.fitInView(self.compare_gt_scene.itemsBoundingRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.compare_diff_view.fitInView(self.compare_diff_scene.itemsBoundingRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _open_gt_editor(self):
        if not self._single_last_image_path or not self._single_last_details:
            QtWidgets.QMessageBox.information(self, "Ground Truth", "Run single-image prediction first.")
            return

        initial_mask_path = self._single_last_details.get("mask_path")
        dlg = GroundTruthEditorDialog(self._single_last_image_path, initial_mask_path, parent=self)
        dlg.exec()
        if dlg.last_saved_path:
            self.single_gt_mask_edit.setText(dlg.last_saved_path)
            self._append_single_log(f"GT saved: {dlg.last_saved_path}")

    def _set_running(self, running: bool):
        folder_running = running and (self._active_mode == "folder")
        single_running = running and (self._active_mode == "single")

        self.tabs.tabBar().setEnabled(not running)

        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(folder_running)
        for w in [
            self.input_dir_edit,
            self.gt_dir_edit,
            self.model_edit,
            self.output_dir_edit,
            self.threshold_spin,
            self.mode_combo,
            self.input_size_spin,
            self.tile_overlap_spin,
            self.tile_batch_spin,
            self.recursive_check,
            self.no_post_check,
        ]:
            w.setEnabled(not running)

        self.single_start_btn.setEnabled(not running)
        self.single_stop_btn.setEnabled(single_running)
        self.single_create_gt_btn.setEnabled((not running) and (self._single_last_details is not None))
        for w in [
            self.single_image_edit,
            self.single_gt_mask_edit,
            self.single_model_edit,
            self.single_output_dir_edit,
            self.single_threshold_spin,
            self.single_mode_combo,
            self.single_input_size_spin,
            self.single_tile_overlap_spin,
            self.single_tile_batch_spin,
            self.single_no_post_check,
        ]:
            w.setEnabled(not running)

        for w in [
            self.compare_pred_mask_edit,
            self.compare_gt_mask_edit,
            self.compare_use_last_btn,
            self.compare_compute_btn,
        ]:
            w.setEnabled(not running)

    def _start_folder(self):
        input_dir = self.input_dir_edit.text().strip()
        gt_dir = self.gt_dir_edit.text().strip() or None
        model_path = self.model_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not input_dir or not os.path.isdir(input_dir):
            QtWidgets.QMessageBox.critical(self, "Error", f"Input folder not found:\n{input_dir}")
            return
        if gt_dir and not os.path.isdir(gt_dir):
            QtWidgets.QMessageBox.critical(self, "Error", f"GT folder not found:\n{gt_dir}")
            return
        if not model_path or not os.path.exists(model_path):
            QtWidgets.QMessageBox.critical(self, "Error", f"Model file not found:\n{model_path}")
            return
        if not output_dir:
            QtWidgets.QMessageBox.critical(self, "Error", "Output folder is empty.")
            return

        os.makedirs(output_dir, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        image_paths = sorted(_iter_images(input_dir, self.recursive_check.isChecked(), exts))
        if not image_paths:
            QtWidgets.QMessageBox.information(self, "No images", f"No images found in:\n{input_dir}")
            return

        self.table.setRowCount(len(image_paths))
        for i, p in enumerate(image_paths):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(p))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(""))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(""))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(""))
            self.table.setItem(i, 5, QtWidgets.QTableWidgetItem("Pending"))

        self.progress.setRange(0, len(image_paths))
        self.progress.setValue(0)
        self.log_box.clear()
        self._append_log(f"Found {len(image_paths)} image(s).")

        self._active_mode = "folder"
        self._thread = QtCore.QThread(self)
        self._worker = PredictWorker(
            image_paths=image_paths,
            input_dir=input_dir,
            gt_dir=gt_dir,
            model_path=model_path,
            output_dir=output_dir,
            threshold=self.threshold_spin.value(),
            apply_postprocessing=not self.no_post_check.isChecked(),
            recursive=self.recursive_check.isChecked(),
            mode=self.mode_combo.currentText(),
            input_size=self.input_size_spin.value(),
            tile_overlap=self.tile_overlap_spin.value(),
            tile_batch_size=self.tile_batch_spin.value(),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.started.connect(lambda total: self._append_log(f"Starting... total={total}"))
        self._worker.progress.connect(self._on_folder_progress)
        self._worker.result.connect(self._on_folder_result)
        self._worker.failed.connect(self._on_folder_failed)
        self._worker.finished.connect(self._on_finished)

        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._set_running(True)
        self._thread.start()

    def _start_single(self):
        image_path = self.single_image_edit.text().strip()
        gt_mask_path = self.single_gt_mask_edit.text().strip() or None
        model_path = self.single_model_edit.text().strip()
        output_dir = self.single_output_dir_edit.text().strip()

        if not image_path or not os.path.exists(image_path):
            QtWidgets.QMessageBox.critical(self, "Error", f"Image not found:\n{image_path}")
            return
        if gt_mask_path and not os.path.exists(gt_mask_path):
            QtWidgets.QMessageBox.critical(self, "Error", f"GT mask not found:\n{gt_mask_path}")
            return
        if not model_path or not os.path.exists(model_path):
            QtWidgets.QMessageBox.critical(self, "Error", f"Model file not found:\n{model_path}")
            return
        if not output_dir:
            QtWidgets.QMessageBox.critical(self, "Error", "Output folder is empty.")
            return

        os.makedirs(output_dir, exist_ok=True)

        try:
            dlg = RoiSelectionDialog(image_path, self)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        roi_box = dlg.roi_box

        self.single_progress.setRange(0, 1)
        self.single_progress.setValue(0)
        self.single_out_path_label.setText("-")
        self.single_dice_label.setText("-")
        self.single_log_box.clear()
        self._append_single_log(f"Image: {image_path}")
        if roi_box is None:
            self._append_single_log("ROI: full image")
        else:
            l, t, r, b = roi_box
            self._append_single_log(f"ROI: left={l}, top={t}, right={r}, bottom={b}")

        self._single_last_details = None
        self._single_last_image_path = image_path
        self.single_create_gt_btn.setEnabled(False)

        self._active_mode = "single"
        self._thread = QtCore.QThread(self)
        self._worker = SinglePredictWorker(
            image_path=image_path,
            gt_mask_path=gt_mask_path,
            model_path=model_path,
            output_dir=output_dir,
            threshold=self.single_threshold_spin.value(),
            apply_postprocessing=not self.single_no_post_check.isChecked(),
            roi_box=roi_box,
            mode=self.single_mode_combo.currentText(),
            input_size=self.single_input_size_spin.value(),
            tile_overlap=self.single_tile_overlap_spin.value(),
            tile_batch_size=self.single_tile_batch_spin.value(),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_single_log)
        self._worker.started.connect(lambda total: self._append_single_log(f"Starting... total={total}"))
        self._worker.progress.connect(lambda idx, total, p: self._append_single_log(f"[{idx}/{total}] {p}"))
        self._worker.result.connect(self._on_single_result)
        self._worker.failed.connect(self._on_single_failed)
        self._worker.finished.connect(self._on_finished)

        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._set_running(True)
        self._thread.start()

    def _stop(self):
        if self._worker is not None:
            self._worker.stop()

    def _on_folder_progress(self, idx: int, total: int, image_path: str):
        self.progress.setValue(idx - 1)
        self.table.setItem(idx - 1, 5, QtWidgets.QTableWidgetItem("Running"))
        self._append_log(f"[{idx}/{total}] {image_path}")

    def _on_folder_result(self, row: int, details: dict):
        gt_path = details.get("gt_mask_path") or ""
        dice = details.get("dice", None)
        output_path = details.get("output_path") or ""

        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(gt_path))
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem("" if dice is None else f"{dice:.4f}"))
        self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(output_path))
        self.table.setItem(row, 5, QtWidgets.QTableWidgetItem("Done"))

        self.progress.setValue(row + 1)

    def _on_folder_failed(self, row: int, msg: str):
        self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"Failed: {msg}"))
        self.progress.setValue(row + 1)

    def _on_single_result(self, details: dict):
        out_path = details.get("output_path") or "-"
        dice = details.get("dice", None)
        self.single_out_path_label.setText(out_path)
        self.single_dice_label.setText("-" if dice is None else f"{dice:.4f}")
        self.single_progress.setValue(1)
        self._single_last_details = details
        self.single_create_gt_btn.setEnabled(True)

    def _on_single_failed(self, msg: str):
        self._append_single_log(f"Failed: {msg}")
        self.single_progress.setValue(1)
        self._single_last_details = None
        self.single_create_gt_btn.setEnabled(False)

    def _on_finished(self):
        if self._active_mode == "folder":
            self.progress.setValue(self.progress.maximum())
            self._append_log("Finished.")
        elif self._active_mode == "single":
            self.single_progress.setValue(self.single_progress.maximum())
            self._append_single_log("Finished.")
        self._set_running(False)
        self._thread = None
        self._worker = None
        self._active_mode = None


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
