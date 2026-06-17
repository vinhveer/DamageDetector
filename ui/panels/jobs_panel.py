from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.models.job import JobSpec, JobStatus

_STATUS_COLOR: dict[str, str] = {
    JobStatus.queued.value:    "#888",
    JobStatus.running.value:   "#3796FF",
    JobStatus.completed.value: "#34D399",
    JobStatus.failed.value:    "#F87171",
    JobStatus.cancelled.value: "#F59E0B",
}


class JobsPanel(QtWidgets.QWidget):
    """Bottom dock tab: job list with status badges and progress."""

    cancelRequested = QtCore.Signal(str)

    COLS = ["ID", "Kind", "Status", "Progress", "Message", ""]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.table = QtWidgets.QTableWidget(0, len(self.COLS), self)
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(28)

        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 90)
        hh.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        root.addWidget(self.table, 1)

        self._row_for: dict[str, int] = {}

    def upsert(self, job: JobSpec) -> None:
        if job.id in self._row_for:
            row = self._row_for[job.id]
        else:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._row_for[job.id] = row

            id_item = QtWidgets.QTableWidgetItem(job.id[-6:])
            id_item.setForeground(QtGui.QColor("#888"))
            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(job.kind.value))

            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(12)
            bar.setStyleSheet("""
                QProgressBar { border: none; background: rgba(255,255,255,0.08); border-radius: 3px; }
                QProgressBar::chunk { background: #3796FF; border-radius: 3px; }
            """)
            # Wrap bar in a centered widget so it doesn't stretch full row height
            bar_wrap = QtWidgets.QWidget()
            bar_layout = QtWidgets.QVBoxLayout(bar_wrap)
            bar_layout.setContentsMargins(8, 0, 8, 0)
            bar_layout.addWidget(bar)
            self.table.setCellWidget(row, 3, bar_wrap)

            cancel_btn = QtWidgets.QPushButton("✕")
            cancel_btn.setFixedSize(24, 22)
            cancel_btn.setToolTip("Cancel job")
            cancel_btn.setStyleSheet("""
                QPushButton { border: none; color: #888; font-size: 11px; }
                QPushButton:hover { color: #F87171; }
                QPushButton:disabled { color: #444; }
            """)
            cancel_btn.clicked.connect(lambda _=False, jid=job.id: self.cancelRequested.emit(jid))
            self.table.setCellWidget(row, 5, cancel_btn)

        # Status (colored)
        status_item = QtWidgets.QTableWidgetItem(f"  {job.status.value}")
        color = _STATUS_COLOR.get(job.status.value, "#ccc")
        status_item.setForeground(QtGui.QColor(color))
        self.table.setItem(row, 2, status_item)

        self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(job.message or ""))

        # Update progress bar
        bar_wrap = self.table.cellWidget(row, 3)
        if bar_wrap is not None:
            bar = bar_wrap.findChild(QtWidgets.QProgressBar)
            if bar is not None:
                bar.setValue(int(max(0.0, min(1.0, job.progress)) * 100))

        # Update cancel button
        cancel_btn = self.table.cellWidget(row, 5)
        if isinstance(cancel_btn, QtWidgets.QPushButton):
            running = job.status in (JobStatus.queued, JobStatus.running)
            cancel_btn.setEnabled(running)
