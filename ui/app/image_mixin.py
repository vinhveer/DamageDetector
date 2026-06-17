from __future__ import annotations

from pathlib import Path

from PySide6 import QtGui, QtWidgets


class ImageMixin:
    def _open_image(self) -> None:
        start_dir = str(self._image_path.parent) if self._image_path else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        self._image_path = Path(path)
        self._detect_panel.table.setRowCount(0)
        self._undo.clear()
        self._canvas.set_image(self._image_path)
        self._reset_detection_groups()
        self._inspector.image_tab.path_label.setText(str(self._image_path))
        rect = self._canvas.image_rect()
        self._inspector.image_tab.size_label.setText(f"{int(rect.width())}x{int(rect.height())}")
        self._refresh_results()
        self._canvas.setFocus()
        self._append_log(f"Loaded {self._image_path.name}")

    def _save_results(self) -> None:
        if self._image_path is None or not self._rows:
            return
        out_dir_str = QtWidgets.QFileDialog.getExistingDirectory(self, "Save results to", str(self._image_path.parent))
        if not out_dir_str:
            return

        from datetime import datetime
        import csv

        out_dir = Path(out_dir_str) / f"roi_detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        crops_dir = out_dir / "rois"
        out_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        overlay = self._canvas.render_overlay()
        overlay.save(str(out_dir / f"{self._image_path.stem}_overlay.png"))

        image = QtGui.QImage(str(self._image_path)).convertToFormat(QtGui.QImage.Format.Format_RGB888)
        for roi_index, rect in enumerate(self._canvas.roi_rects(), start=1):
            x1 = max(0, int(rect.left()))
            y1 = max(0, int(rect.top()))
            x2 = min(image.width(), int(rect.right()))
            y2 = min(image.height(), int(rect.bottom()))
            image.copy(x1, y1, max(1, x2 - x1), max(1, y2 - y1)).save(str(crops_dir / f"roi_{roi_index:03d}.png"))

        with (out_dir / "detections.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["roi_index", "detector_name", "group_name", "label", "score", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            for row in self._filtered_rows():
                writer.writerow(row.__dict__)
        self.statusBar().showMessage(f"Saved: {out_dir}")
