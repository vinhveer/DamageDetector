from __future__ import annotations

import csv

from PySide6 import QtCore, QtWidgets


class CompareResultsPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: list[dict] = []

        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self._summary = QtWidgets.QLabel("No results", self)
        top.addWidget(self._summary)
        top.addStretch(1)
        self._export_btn = QtWidgets.QPushButton("Export CSV...", self)
        self._export_btn.clicked.connect(self._export_csv)
        self._export_btn.setEnabled(False)
        top.addWidget(self._export_btn)
        layout.addLayout(top)

        self._table = QtWidgets.QTableWidget(self)
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Image", "GT Mask", "Dice", "IoU"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        layout.addWidget(self._table, 1)

    def set_results(self, results: list[dict]) -> None:
        self._results = [dict(item) for item in results]
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._results))
        for row, result in enumerate(self._results):
            self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(result.get("image") or "")))
            self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(result.get("gt_mask") or "")))
            self._table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{float(result.get('dice', 0.0)):.4f}"))
            self._table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{float(result.get('iou', 0.0)):.4f}"))
        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()
        self._export_btn.setEnabled(bool(self._results))
        if not self._results:
            self._summary.setText("No results")
            return
        mean_dice = sum(float(item.get("dice", 0.0)) for item in self._results) / len(self._results)
        mean_iou = sum(float(item.get("iou", 0.0)) for item in self._results) / len(self._results)
        self._summary.setText(f"{len(self._results)} items | mean Dice {mean_dice:.4f} | mean IoU {mean_iou:.4f}")

    def _export_csv(self) -> None:
        if not self._results:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Compare Results", "", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "gt_mask", "dice", "iou"])
            writer.writeheader()
            for row in self._results:
                writer.writerow(
                    {
                        "image": row.get("image", ""),
                        "gt_mask": row.get("gt_mask", ""),
                        "dice": row.get("dice", 0.0),
                        "iou": row.get("iou", 0.0),
                    }
                )
        QtWidgets.QMessageBox.information(self, "Compare", "Export successful.")
