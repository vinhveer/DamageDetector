from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from image_loader import ImageLoader
from models import AssignmentRow, ClusterSummary


# ─────────────────────────────────────────────────────────
#  Cluster Card
# ─────────────────────────────────────────────────────────


class ClusterCard(QFrame):
    opened = Signal(str)

    def __init__(self, cluster: ClusterSummary) -> None:
        super().__init__()
        self.cluster = cluster
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        title = QLabel(cluster.cluster_key)
        title.setFont(QFont("Arial", 13, QFont.Bold))
        layout.addWidget(title)

        layout.addWidget(QLabel(f"Images: {cluster.cluster_size}  |  Major: {cluster.major_label}  |  Purity: {cluster.purity:.3f}"))
        layout.addWidget(QLabel(f"crack {cluster.crack_count}  ·  mold {cluster.mold_count}  ·  spall {cluster.spall_count}"))

        button = QPushButton("Open group")
        button.clicked.connect(lambda: self.opened.emit(cluster.cluster_key))
        layout.addWidget(button)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        self.opened.emit(self.cluster.cluster_key)
        super().mouseDoubleClickEvent(event)


# ─────────────────────────────────────────────────────────
#  Group List Page  –  all cards in a scrollable grid
# ─────────────────────────────────────────────────────────


class GroupListPage(QWidget):
    opened = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.label_scope = "crack"
        self.clusters: list[ClusterSummary] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(6)

        self.summary = QLabel("No groups loaded")
        self.summary.setFont(QFont("Arial", 13, QFont.Bold))
        root.addWidget(self.summary)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setSpacing(8)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)

    def set_groups(self, label_scope: str, clusters: list[ClusterSummary]) -> None:
        self.label_scope = label_scope
        self.clusters = clusters
        self._render()

    def _render(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        total_images = sum(c.cluster_size for c in self.clusters)
        self.summary.setText(f"{self.label_scope}: {len(self.clusters)} groups | {total_images} images")

        cols = 3
        for idx, cluster in enumerate(self.clusters):
            card = ClusterCard(cluster)
            card.opened.connect(lambda key, label=self.label_scope: self.opened.emit(label, key))
            self.grid.addWidget(card, idx // cols, idx % cols)
        self.grid.setRowStretch(max(1, len(self.clusters) // cols + 1), 1)


# ─────────────────────────────────────────────────────────
#  Image Tile
# ─────────────────────────────────────────────────────────


class ImageTile(QFrame):
    def __init__(self, row: AssignmentRow, pixmap, error: str = "") -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        if error:
            image_label.setText(error)
            image_label.setWordWrap(True)
        else:
            image_label.setPixmap(pixmap)
        layout.addWidget(image_label)

        caption = QLabel(
            f"id={row.result_id} | {row.predicted_label} | "
            f"conf={row.predicted_probability_pct:.1f}% | dist={row.distance_to_center:.3f}"
        )
        caption.setWordWrap(True)
        layout.addWidget(caption)


# ─────────────────────────────────────────────────────────
#  Detail Page  –  all images in a scrollable grid + table
# ─────────────────────────────────────────────────────────


class DetailPage(QWidget):
    back_requested = Signal()

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()
        self.image_loader = image_loader
        self.rows: list[AssignmentRow] = []
        self.cluster: ClusterSummary | None = None
        self.image_root: Path | None = None
        self.image_size = 240
        self.padding_ratio = 0.05

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(6)

        # ── Top bar ──────────────────────────────────────
        top = QHBoxLayout()
        self.back_button = QPushButton("Back to group list")
        self.back_button.clicked.connect(self.back_requested.emit)
        self.title = QLabel("No group selected")
        self.title.setFont(QFont("Arial", 15, QFont.Bold))
        top.addWidget(self.back_button)
        top.addWidget(self.title, 1)
        root.addLayout(top)

        # ── Image grid ───────────────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setSpacing(6)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)

        # ── Data table ───────────────────────────────────
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(180)
        root.addWidget(self.table)

    def set_detail(
        self,
        cluster: ClusterSummary,
        rows: list[AssignmentRow],
        *,
        image_root: Path | None,
        image_size: int,
        padding_ratio: float,
    ) -> None:
        self.cluster = cluster
        self.rows = rows
        self.image_root = image_root
        self.image_size = int(image_size)
        self.padding_ratio = float(padding_ratio)
        self._render()

    def update_image_settings(self, *, image_size: int, padding_ratio: float) -> None:
        self.image_size = int(image_size)
        self.padding_ratio = float(padding_ratio)
        self._render()

    def _render(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if self.cluster is None:
            return

        self.title.setText(f"{self.cluster.cluster_key} | {len(self.rows)} images")

        cols = 4
        for idx, row in enumerate(self.rows):
            try:
                pixmap = self.image_loader.crop_pixmap(
                    row,
                    self.image_root,
                    padding_ratio=self.padding_ratio,
                    image_size=self.image_size,
                )
                tile = ImageTile(row, pixmap)
            except Exception as exc:
                tile = ImageTile(row, None, error=str(exc))
            self.grid.addWidget(tile, idx // cols, idx % cols)
        self.grid.setRowStretch(max(1, len(self.rows) // cols + 1), 1)
        self._render_table()

    def _render_table(self) -> None:
        headers = ["result_id", "image", "label", "clip", "det", "distance", "suggested"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self.rows))
        for row_idx, row in enumerate(self.rows):
            values = [
                row.result_id,
                row.image_rel_path,
                row.predicted_label,
                f"{row.predicted_probability_pct:.2f}",
                f"{row.detector_score:.4f}",
                f"{row.distance_to_center:.4f}",
                row.suggested_label,
            ]
            for col_idx, value in enumerate(values):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
