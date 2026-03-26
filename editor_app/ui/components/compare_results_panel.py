from __future__ import annotations

import csv

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from PySide6 import QtCharts
except Exception:
    QtCharts = None


CHART_OPTIONS = [
    ("dice_bar", "Dice by image"),
    ("iou_bar", "IoU by image"),
    ("scatter", "Dice vs IoU"),
    ("quality_pie", "Quality breakdown"),
]


class CompareChartDialog(QtWidgets.QDialog):
    def __init__(self, chart_key: str, chart_label: str, results: list[dict], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Compare Chart - {chart_label}")
        self.resize(980, 620)

        layout = QtWidgets.QVBoxLayout(self)
        if QtCharts is None:
            label = QtWidgets.QLabel("QtCharts is unavailable in this environment.", self)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, 1)
            return

        chart = self._build_chart(chart_key, chart_label, results)
        view = QtCharts.QChartView(chart, self)
        view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        layout.addWidget(view, 1)

    def _build_chart(self, chart_key: str, chart_label: str, results: list[dict]):
        if chart_key == "dice_bar":
            return self._build_metric_bar_chart(results, metric="dice", title="Dice score by image")
        if chart_key == "iou_bar":
            return self._build_metric_bar_chart(results, metric="iou", title="IoU score by image")
        if chart_key == "scatter":
            return self._build_scatter_chart(results, title="Dice vs IoU")
        return self._build_quality_pie_chart(results, title=chart_label)

    def _build_metric_bar_chart(self, results: list[dict], *, metric: str, title: str):
        chart = QtCharts.QChart()
        chart.setTitle(title)
        series = QtCharts.QBarSeries(chart)
        bar_set = QtCharts.QBarSet(metric.upper(), series)
        categories: list[str] = []
        for result in results:
            value = float(result.get(metric, 0.0))
            bar_set.append(max(0.0, min(1.0, value)))
            categories.append(self._short_label(str(result.get("image") or "")))
        series.append(bar_set)
        chart.addSeries(series)
        axis_x = QtCharts.QBarCategoryAxis(chart)
        axis_x.append(categories)
        axis_x.setLabelsAngle(-70)
        axis_y = QtCharts.QValueAxis(chart)
        axis_y.setRange(0.0, 1.0)
        axis_y.setTitleText(metric.upper())
        axis_y.setLabelFormat("%.2f")
        chart.addAxis(axis_x, QtCore.Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, QtCore.Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        chart.legend().setVisible(True)
        return chart

    def _build_scatter_chart(self, results: list[dict], *, title: str):
        chart = QtCharts.QChart()
        chart.setTitle(title)
        series = QtCharts.QScatterSeries(chart)
        series.setName("Images")
        series.setMarkerSize(12.0)
        for result in results:
            dice = max(0.0, min(1.0, float(result.get("dice", 0.0))))
            iou = max(0.0, min(1.0, float(result.get("iou", 0.0))))
            series.append(dice, iou)
        chart.addSeries(series)
        axis_x = QtCharts.QValueAxis(chart)
        axis_x.setRange(0.0, 1.0)
        axis_x.setTitleText("Dice")
        axis_x.setLabelFormat("%.2f")
        axis_y = QtCharts.QValueAxis(chart)
        axis_y.setRange(0.0, 1.0)
        axis_y.setTitleText("IoU")
        axis_y.setLabelFormat("%.2f")
        chart.addAxis(axis_x, QtCore.Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, QtCore.Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        chart.legend().setVisible(False)
        return chart

    def _build_quality_pie_chart(self, results: list[dict], *, title: str):
        chart = QtCharts.QChart()
        chart.setTitle(title)
        series = QtCharts.QPieSeries(chart)
        counts = {"Low (<0.50)": 0, "Medium (0.50-0.79)": 0, "High (>=0.80)": 0}
        for result in results:
            dice = float(result.get("dice", 0.0))
            if dice >= 0.8:
                counts["High (>=0.80)"] += 1
            elif dice >= 0.5:
                counts["Medium (0.50-0.79)"] += 1
            else:
                counts["Low (<0.50)"] += 1
        for label, value in counts.items():
            if value <= 0:
                continue
            slice_obj = series.append(label, value)
            slice_obj.setLabelVisible(True)
        chart.addSeries(series)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        return chart

    def _short_label(self, text: str) -> str:
        value = str(text or "").strip()
        if len(value) <= 18:
            return value
        return f"{value[:15]}..."


class CompareResultsPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: list[dict] = []

        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self._summary = QtWidgets.QLabel("No results", self)
        top.addWidget(self._summary)
        top.addStretch(1)

        self._chart_combo = QtWidgets.QComboBox(self)
        for key, label in CHART_OPTIONS:
            self._chart_combo.addItem(label, key)
        top.addWidget(self._chart_combo)

        self._chart_btn = QtWidgets.QPushButton("Show Chart", self)
        self._chart_btn.clicked.connect(self._show_chart)
        self._chart_btn.setEnabled(False)
        if QtCharts is None:
            self._chart_btn.setToolTip("QtCharts is unavailable.")
        top.addWidget(self._chart_btn)

        self._export_btn = QtWidgets.QPushButton("Export CSV...", self)
        self._export_btn.clicked.connect(self._export_csv)
        self._export_btn.setEnabled(False)
        top.addWidget(self._export_btn)
        layout.addLayout(top)

        self._table = QtWidgets.QTableWidget(self)
        self._table.setColumnCount(4)
        self._table.setRowCount(1)
        self._table.setHorizontalHeaderLabels(["Image", "GT Mask", "Dice", "IoU"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(False)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, 1)
        self._populate_empty_table()

    def set_results(self, results: list[dict]) -> None:
        self._results = [dict(item) for item in results]
        self._export_btn.setEnabled(bool(self._results))
        self._chart_btn.setEnabled(bool(self._results) and QtCharts is not None)
        if not self._results:
            self._summary.setText("No results")
            self._populate_empty_table()
            return

        self._table.clearSpans()
        self._table.setSortingEnabled(False)
        self._table.clearContents()
        self._table.setRowCount(len(self._results))
        for row, result in enumerate(self._results):
            self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(result.get("image") or "")))
            self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(result.get("gt_mask") or "")))
            self._table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{float(result.get('dice', 0.0)):.4f}"))
            self._table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{float(result.get('iou', 0.0)):.4f}"))
        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()
        mean_dice = sum(float(item.get("dice", 0.0)) for item in self._results) / len(self._results)
        mean_iou = sum(float(item.get("iou", 0.0)) for item in self._results) / len(self._results)
        self._summary.setText(f"{len(self._results)} items | mean Dice {mean_dice:.4f} | mean IoU {mean_iou:.4f}")

    def _populate_empty_table(self) -> None:
        self._table.clearSpans()
        self._table.setSortingEnabled(False)
        self._table.clearContents()
        self._table.setRowCount(1)
        self._table.setSpan(0, 0, 1, self._table.columnCount())
        item = QtWidgets.QTableWidgetItem("No compare results")
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(0, 0, item)

    def _show_chart(self) -> None:
        if not self._results:
            return
        chart_key = str(self._chart_combo.currentData() or "")
        chart_label = str(self._chart_combo.currentText() or "Chart")
        dialog = CompareChartDialog(chart_key, chart_label, self._results, self)
        dialog.exec()

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
