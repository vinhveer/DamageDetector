import csv
import os
from pathlib import Path

from PySide6 import QtCore, QtWidgets


class ComparePanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)

        # Top controls
        top_layout = QtWidgets.QHBoxLayout()
        self._export_btn = QtWidgets.QPushButton("Export CSV...")
        self._export_btn.clicked.connect(self._export_csv)
        self._export_btn.setEnabled(False)
        top_layout.addWidget(self._export_btn)
        top_layout.addStretch()

        self._layout.addLayout(top_layout)

        # Table
        self._table = QtWidgets.QTableWidget(self)
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Image", "GT Mask", "Dice", "IoU"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._layout.addWidget(self._table)

        self._results = []

    def set_results(self, results: list[dict]) -> None:
        self._results = results
        self._table.setRowCount(len(results))
        for row, res in enumerate(results):
            img_item = QtWidgets.QTableWidgetItem(res.get("image", ""))
            gt_item = QtWidgets.QTableWidgetItem(res.get("gt_mask", ""))
            
            dice = res.get("dice", 0.0)
            iou = res.get("iou", 0.0)
            
            dice_item = QtWidgets.QTableWidgetItem(f"{dice:.4f}")
            iou_item = QtWidgets.QTableWidgetItem(f"{iou:.4f}")
            
            self._table.setItem(row, 0, img_item)
            self._table.setItem(row, 1, gt_item)
            self._table.setItem(row, 2, dice_item)
            self._table.setItem(row, 3, iou_item)

        self._table.resizeColumnsToContents()
        self._export_btn.setEnabled(len(results) > 0)

    def _export_csv(self) -> None:
        if not self._results:
            return
            
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Compare Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "gt_mask", "dice", "iou"])
                writer.writeheader()
                for res in self._results:
                    writer.writerow({
                        "image": res.get("image", ""),
                        "gt_mask": res.get("gt_mask", ""),
                        "dice": res.get("dice", 0.0),
                        "iou": res.get("iou", 0.0),
                    })
            QtWidgets.QMessageBox.information(self, "Export", "Export successful.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")
