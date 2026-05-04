from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from edit_window import EditWindow
from image_loader import ImageLoader
from models import LABELS, AssignmentRow, ClusterSummary

# ── Constants ────────────────────────────────────────────
CARD_W = 220   # fixed card width  (px)
CARD_H = 270   # fixed card height (px)
COLS   = 5     # columns in card grid


# ─────────────────────────────────────────────────────────
#  Cluster Card  (group-list page)
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

        layout.addWidget(QLabel(
            f"Images: {cluster.cluster_size}  |  Major: {cluster.major_label}  |  Purity: {cluster.purity:.3f}"
        ))
        layout.addWidget(QLabel(
            f"crack {cluster.crack_count}  ·  mold {cluster.mold_count}  ·  spall {cluster.spall_count}"
        ))

        button = QPushButton("Open group")
        button.setFixedSize(120, 36)
        button.clicked.connect(lambda: self.opened.emit(cluster.cluster_key))
        layout.addWidget(button)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        self.opened.emit(self.cluster.cluster_key)
        super().mouseDoubleClickEvent(event)


# ─────────────────────────────────────────────────────────
#  Group List Page
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
        self.summary.setText(
            f"{self.label_scope}: {len(self.clusters)} groups | {total_images} images"
        )

        cols = 3
        for idx, cluster in enumerate(self.clusters):
            card = ClusterCard(cluster)
            card.opened.connect(lambda key, label=self.label_scope: self.opened.emit(label, key))
            self.grid.addWidget(card, idx // cols, idx % cols)
        self.grid.setRowStretch(max(1, len(self.clusters) // cols + 1), 1)


# ─────────────────────────────────────────────────────────
#  Fixed-size Image Card  (card-grid view)
# ─────────────────────────────────────────────────────────


class ImageCard(QFrame):
    """A fixed-size card showing one cropped image + short caption."""
    clicked = Signal(int)   # emits index into rows list
    edit_requested = Signal(int)   # emits index for editing
    remove_flags_requested = Signal(int)   # emits index for flag clearing

    def __init__(self, index: int, row: AssignmentRow, pixmap=None, error: str = "",
                 *, box_count: int = 1) -> None:
        super().__init__()
        self._index = index
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(CARD_W, CARD_H)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        self.img_label = QLabel("Loading...")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedHeight(CARD_H - 120)   # leave room for caption + buttons
        if error:
            self.set_error(error)
        elif pixmap is not None:
            self.set_pixmap(pixmap)
        layout.addWidget(self.img_label, 1)

        boxes_text = f"  [{box_count} boxes]" if box_count > 1 else ""
        caption = QLabel(
            f"#{index + 1}  id={row.result_id}{boxes_text}\n"
            f"{row.predicted_label}  conf={row.predicted_probability_pct:.1f}%\n"
            f"dist={row.distance_to_center:.3f}"
        )
        caption.setWordWrap(True)
        layout.addWidget(caption)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        edit_btn = QPushButton("Edit")
        edit_btn.setFixedSize(70, 30)
        edit_btn.clicked.connect(lambda: self.edit_requested.emit(self._index))
        remove_btn = QPushButton("Remove flags")
        remove_btn.setFixedSize(110, 30)
        remove_btn.clicked.connect(lambda: self.remove_flags_requested.emit(self._index))
        btn_row.addStretch(1)
        btn_row.addWidget(edit_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

    def set_pixmap(self, pixmap) -> None:
        self.img_label.setPixmap(
            pixmap.scaled(CARD_W - 8, CARD_H - 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.img_label.setText("")

    def set_error(self, error: str) -> None:
        self.img_label.setPixmap(QPixmap())
        self.img_label.setText(error)
        self.img_label.setWordWrap(True)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.clicked.emit(self._index)
        super().mouseDoubleClickEvent(event)


# ─────────────────────────────────────────────────────────
#  Helper – group rows by image path
# ─────────────────────────────────────────────────────────


def _group_rows_by_image(rows: list[AssignmentRow]) -> list[list[AssignmentRow]]:
    """Group rows that share the same image into a single list.

    Returns a list of groups; each group is a list of AssignmentRow
    whose image_rel_path (or image_path) match.
    """
    groups: dict[str, list[AssignmentRow]] = defaultdict(list)
    for row in rows:
        key = row.image_rel_path or row.image_path or str(row.result_id)
        groups[key].append(row)
    return list(groups.values())


class _CropSignals(QObject):
    finished = Signal(int, int, QImage, str)


class _CropTask(QRunnable):
    def __init__(
        self,
        generation: int,
        index: int,
        row: AssignmentRow,
        image_loader: ImageLoader,
        image_root: Path | None,
        padding_ratio: float,
        image_size: int,
        signals: _CropSignals,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.index = index
        self.row = row
        self.image_loader = image_loader
        self.image_root = image_root
        self.padding_ratio = padding_ratio
        self.image_size = image_size
        self.signals = signals

    @Slot()
    def run(self) -> None:
        try:
            image = self.image_loader.crop_image(
                self.row,
                self.image_root,
                padding_ratio=self.padding_ratio,
                image_size=self.image_size,
            )
            self.signals.finished.emit(self.generation, self.index, image, "")
        except Exception as exc:
            self.signals.finished.emit(self.generation, self.index, QImage(), str(exc))


# ─────────────────────────────────────────────────────────
#  Card Grid View  (scroll area of fixed ImageCard cells)
#  – all cards visible, image loading runs in worker threads
# ─────────────────────────────────────────────────────────


class CardGridView(QWidget):
    """Shows all images as fixed-size cards in a scrollable grid."""
    image_selected = Signal(int)   # user clicked a card → switch to single view
    edit_requested = Signal(int)   # user clicked Edit on a card
    remove_flags_requested = Signal(int)   # user clicked Remove flags on a card

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setSpacing(8)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll)

        # Image loading state
        self._grouped_items: list[tuple[int, AssignmentRow, list[AssignmentRow]]] = []
        self._image_loader: ImageLoader | None = None
        self._image_root: Path | None = None
        self._padding_ratio = 0.05
        self._image_size = 240
        self._cards_by_index: dict[int, ImageCard] = {}
        self._generation = 0
        self._thread_pool = QThreadPool.globalInstance()
        self._crop_signals = _CropSignals()
        self._crop_signals.finished.connect(self._on_crop_finished)

    def populate(
        self,
        rows: list[AssignmentRow],
        image_loader: ImageLoader,
        image_root: Path | None,
        padding_ratio: float,
        image_size: int,
    ) -> None:
        # Clear old widgets
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cards_by_index.clear()
        self._generation += 1

        self._image_loader = image_loader
        self._image_root = image_root
        self._padding_ratio = padding_ratio
        self._image_size = image_size

        # Group rows by image
        image_groups = _group_rows_by_image(rows)
        self._grouped_items = []
        for idx, group in enumerate(image_groups):
            representative = group[0]  # first row as representative
            self._grouped_items.append((idx, representative, group))

        if not self._grouped_items:
            return
        self.scroll.verticalScrollBar().setValue(0)
        self._render_all_cards()

    def _render_all_cards(self) -> None:
        if self._image_loader is None:
            return
        cols = COLS
        for i, (idx, representative, group) in enumerate(self._grouped_items):
            card = ImageCard(idx, representative, box_count=len(group))
            card.clicked.connect(self.image_selected.emit)
            card.edit_requested.connect(self.edit_requested.emit)
            card.remove_flags_requested.connect(self.remove_flags_requested.emit)
            self.grid.addWidget(card, i // cols, i % cols)
            self._cards_by_index[idx] = card
            self._start_crop(idx, representative)

        # Ensure last row stretches
        total_rows = max(1, len(self._grouped_items) // cols + 1)
        self.grid.setRowStretch(total_rows, 1)

    def _start_crop(self, index: int, row: AssignmentRow) -> None:
        if self._image_loader is None:
            return
        task = _CropTask(
            self._generation,
            index,
            row,
            self._image_loader,
            self._image_root,
            self._padding_ratio,
            self._image_size,
            self._crop_signals,
        )
        self._thread_pool.start(task)

    @Slot(int, int, QImage, str)
    def _on_crop_finished(self, generation: int, index: int, image: QImage, error: str) -> None:
        if generation != self._generation:
            return
        card = self._cards_by_index.get(index)
        if card is None:
            return
        if error:
            card.set_error(error)
        else:
            card.set_pixmap(QPixmap.fromImage(image))

# ─────────────────────────────────────────────────────────
#  Single Image View  (one large image + prev/next)
# ─────────────────────────────────────────────────────────


class SingleImageView(QWidget):
    """Shows one image at a time with navigation controls."""

    edit_requested = Signal(int)
    remove_flags_requested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[AssignmentRow] = []
        self.image_loader: ImageLoader | None = None
        self.image_root: Path | None = None
        self.padding_ratio = 0.05
        self.image_size = 600
        self._current_idx = 0
        self._generation = 0
        self._thread_pool = QThreadPool.globalInstance()
        self._crop_signals = _CropSignals()
        self._crop_signals.finished.connect(self._on_crop_finished)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(6)

        # Navigation bar
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setFixedSize(120, 36)
        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedSize(120, 36)
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setFixedSize(100, 36)
        self.remove_flags_btn = QPushButton("Remove flags")
        self.remove_flags_btn.setFixedSize(130, 36)
        self.idx_label = QLabel("")
        self.idx_label.setAlignment(Qt.AlignCenter)
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)
        self.edit_btn.clicked.connect(lambda: self.edit_requested.emit(self._current_idx))
        self.remove_flags_btn.clicked.connect(lambda: self.remove_flags_requested.emit(self._current_idx))
        nav.addWidget(self.prev_btn)
        nav.addStretch(1)
        nav.addWidget(self.idx_label)
        nav.addStretch(1)
        nav.addWidget(self.edit_btn)
        nav.addWidget(self.remove_flags_btn)
        nav.addWidget(self.next_btn)
        root.addLayout(nav)

        # Image display (scrollable for very tall images)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.img_label)
        root.addWidget(self.scroll, 1)

        # Caption
        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setFont(QFont("Arial", 11))
        root.addWidget(self.caption)

    def load(
        self,
        rows: list[AssignmentRow],
        image_loader: ImageLoader,
        image_root: Path | None,
        padding_ratio: float,
        image_size: int,
        start_index: int = 0,
    ) -> None:
        self.rows = rows
        self.image_loader = image_loader
        self.image_root = image_root
        self.padding_ratio = padding_ratio
        self.image_size = image_size
        self._current_idx = max(0, min(start_index, len(rows) - 1))
        self._show_current()

    def go_to(self, index: int) -> None:
        self._current_idx = max(0, min(index, len(self.rows) - 1))
        self._show_current()

    def update_settings(self, *, padding_ratio: float, image_size: int) -> None:
        self.padding_ratio = padding_ratio
        self.image_size = image_size
        self._show_current()

    def _prev(self) -> None:
        if self._current_idx > 0:
            self._current_idx -= 1
            self._show_current()

    def _next(self) -> None:
        if self._current_idx < len(self.rows) - 1:
            self._current_idx += 1
            self._show_current()

    def _show_current(self) -> None:
        if not self.rows or self.image_loader is None:
            return
        idx = self._current_idx
        row = self.rows[idx]
        n = len(self.rows)

        self.idx_label.setText(f"{idx + 1} / {n}")
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < n - 1)

        try:
            self._generation += 1
            self.img_label.setPixmap(QPixmap())
            self.img_label.setText("Loading...")
            task = _CropTask(
                self._generation,
                idx,
                row,
                self.image_loader,
                self.image_root,
                self.padding_ratio,
                self.image_size,
                self._crop_signals,
            )
            self._thread_pool.start(task)
        except Exception as exc:
            self.img_label.setText(str(exc))

        self.caption.setText(
            f"id={row.result_id}  |  {row.predicted_label}  |  "
            f"conf={row.predicted_probability_pct:.1f}%  |  dist={row.distance_to_center:.3f}  |  "
            f"suggested={row.suggested_label}"
        )

    @Slot(int, int, QImage, str)
    def _on_crop_finished(self, generation: int, index: int, image: QImage, error: str) -> None:
        if generation != self._generation or index != self._current_idx:
            return
        if error:
            self.img_label.setPixmap(QPixmap())
            self.img_label.setText(error)
        else:
            self.img_label.setText("")
            self.img_label.setPixmap(QPixmap.fromImage(image))


# ─────────────────────────────────────────────────────────
#  Detail Page
# ─────────────────────────────────────────────────────────


class DetailPage(QWidget):
    back_requested = Signal()
    remove_group_flags_requested = Signal()
    remove_image_flags_requested = Signal(list)

    # View mode constants
    MODE_GRID   = "grid"
    MODE_SINGLE = "single"

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()
        self.image_loader = image_loader
        self.rows: list[AssignmentRow] = []
        self.cluster: ClusterSummary | None = None
        self.image_root: Path | None = None
        self.image_size = 240
        self.padding_ratio = 0.05
        self._view_mode = self.MODE_GRID

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(6)

        # ── Top bar ──────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)

        self.back_button = QPushButton("Back")
        self.back_button.setFixedSize(100, 36)
        self.back_button.clicked.connect(self.back_requested.emit)
        top.addWidget(self.back_button)

        self.title = QLabel("No group selected")
        self.title.setFont(QFont("Arial", 15, QFont.Bold))
        top.addWidget(self.title, 1)

        # View mode toggle buttons
        self.grid_btn = QPushButton("Grid")
        self.grid_btn.setFixedSize(100, 36)
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(True)
        self.grid_btn.clicked.connect(lambda: self._switch_view(self.MODE_GRID))
        top.addWidget(self.grid_btn)

        self.single_btn = QPushButton("Single")
        self.single_btn.setFixedSize(100, 36)
        self.single_btn.setCheckable(True)
        self.single_btn.setChecked(False)
        self.single_btn.clicked.connect(lambda: self._switch_view(self.MODE_SINGLE))
        top.addWidget(self.single_btn)

        self.remove_group_flags_btn = QPushButton("Remove group flags")
        self.remove_group_flags_btn.setFixedSize(170, 36)
        self.remove_group_flags_btn.clicked.connect(self.remove_group_flags_requested.emit)
        top.addWidget(self.remove_group_flags_btn)

        root.addLayout(top)

        # ── Gallery stack (Grid / Single) ────────────────
        self.gallery_stack = QStackedWidget()

        self.card_grid_view = CardGridView()
        self.card_grid_view.image_selected.connect(self._on_card_selected)
        self.card_grid_view.edit_requested.connect(self._on_edit_requested)
        self.card_grid_view.remove_flags_requested.connect(self._on_remove_card_flags_requested)
        self.gallery_stack.addWidget(self.card_grid_view)   # index 0

        self.single_view = SingleImageView()
        self.single_view.edit_requested.connect(self._on_edit_requested)
        self.single_view.remove_flags_requested.connect(self._on_remove_single_flags_requested)
        self.gallery_stack.addWidget(self.single_view)      # index 1

        root.addWidget(self.gallery_stack, 1)

        # Keyboard shortcuts for single-view navigation
        QShortcut(QKeySequence(Qt.Key_Left),  self, self.single_view._prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.single_view._next)

        # ── Data table (toggle-able, hidden by default) ──
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(180)
        self.table.setMinimumHeight(120)
        self.table.setVisible(False)
        root.addWidget(self.table)

    # ── Public API ───────────────────────────────────────

    def toggle_table(self) -> None:
        self.table.setVisible(not self.table.isVisible())

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
        self.title.setText(f"{cluster.cluster_key} | {len(rows)} images")
        self._refresh_gallery()
        self._render_table()

    def update_image_settings(self, *, image_size: int, padding_ratio: float) -> None:
        self.image_size = int(image_size)
        self.padding_ratio = float(padding_ratio)
        self._refresh_gallery()

    # ── Internal ─────────────────────────────────────────

    def _switch_view(self, mode: str) -> None:
        self._view_mode = mode
        if mode == self.MODE_GRID:
            self.gallery_stack.setCurrentIndex(0)
            self.grid_btn.setChecked(True)
            self.single_btn.setChecked(False)
        else:
            self.gallery_stack.setCurrentIndex(1)
            self.grid_btn.setChecked(False)
            self.single_btn.setChecked(True)

    def _on_card_selected(self, index: int) -> None:
        """Clicking a card in grid view jumps to that image in single view."""
        row_index = self._row_index_for_image_group(index)
        if row_index is None:
            return
        self._switch_view(self.MODE_SINGLE)
        self.single_view.go_to(row_index)

    def _row_index_for_image_group(self, group_index: int) -> int | None:
        image_groups = _group_rows_by_image(self.rows)
        if not (0 <= group_index < len(image_groups)):
            return None
        first_row = image_groups[group_index][0]
        for row_index, row in enumerate(self.rows):
            if row is first_row:
                return row_index
        key = first_row.image_rel_path or first_row.image_path or str(first_row.result_id)
        for row_index, row in enumerate(self.rows):
            if (row.image_rel_path or row.image_path or str(row.result_id)) == key:
                return row_index
        return None

    def _on_edit_requested(self, index: int) -> None:
        """Open the Edit window for the row(s) at *index*."""
        # Determine which rows to edit (group by image)
        image_groups = _group_rows_by_image(self.rows)
        if 0 <= index < len(image_groups):
            group = image_groups[index]
        else:
            # Fallback: single row
            if 0 <= index < len(self.rows):
                group = [self.rows[index]]
            else:
                return

        edit_win = EditWindow(group, self.image_root, parent=self)
        edit_win.saved.connect(lambda modified: self._apply_edit(index, modified))
        edit_win.show()

    def _on_remove_card_flags_requested(self, index: int) -> None:
        image_groups = _group_rows_by_image(self.rows)
        if 0 <= index < len(image_groups):
            self.remove_image_flags_requested.emit([int(row.result_id) for row in image_groups[index]])

    def _on_remove_single_flags_requested(self, index: int) -> None:
        if not (0 <= index < len(self.rows)):
            return
        row = self.rows[index]
        key = row.image_rel_path or row.image_path or str(row.result_id)
        result_ids = [
            int(item.result_id)
            for item in self.rows
            if (item.image_rel_path or item.image_path or str(item.result_id)) == key
        ]
        self.remove_image_flags_requested.emit(result_ids)

    def clear_flags_for_results(self, result_ids: set[int]) -> None:
        self.rows = [
            replace(row, is_outlier=0, label_suspect=0) if int(row.result_id) in result_ids else row
            for row in self.rows
        ]
        self._refresh_gallery()
        self._render_table()

    def clear_flags_for_all_rows(self) -> None:
        self.rows = [replace(row, is_outlier=0, label_suspect=0) for row in self.rows]
        self._refresh_gallery()
        self._render_table()

    def _apply_edit(self, index: int, modified_rows: list[AssignmentRow]) -> None:
        """Replace the rows for the edited image group with the modified versions."""
        image_groups = _group_rows_by_image(self.rows)
        if 0 <= index < len(image_groups):
            old_group = image_groups[index]
            old_ids = {id(r) for r in old_group}
            # Rebuild full row list: keep non-edited, replace edited group
            new_rows: list[AssignmentRow] = []
            inserted = False
            for r in self.rows:
                if id(r) in old_ids:
                    if not inserted:
                        new_rows.extend(modified_rows)
                        inserted = True
                    # skip old rows
                else:
                    new_rows.append(r)
            if not inserted:
                new_rows.extend(modified_rows)
            self.rows = new_rows
            self._refresh_gallery()
            self._render_table()

    def _refresh_gallery(self) -> None:
        # Refresh card grid
        self.card_grid_view.populate(
            self.rows,
            self.image_loader,
            self.image_root,
            self.padding_ratio,
            self.image_size,
        )
        # Refresh single view (keep current position if possible)
        cur = getattr(self.single_view, "_current_idx", 0)
        self.single_view.load(
            self.rows,
            self.image_loader,
            self.image_root,
            self.padding_ratio,
            self.image_size,
            start_index=cur,
        )

    def _render_table(self) -> None:
        headers = ["result_id", "image", "label", "clip%", "det", "distance", "suggested"]
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
