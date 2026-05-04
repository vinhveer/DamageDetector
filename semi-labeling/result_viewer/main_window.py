from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

from db import FeatureGroupStore, SourceStore, merge_source_meta
from image_loader import ImageLoader
from models import LABELS, ClusterSummary, GroupRun
from paths import default_feature_db, default_image_root, default_source_db
from widgets import DetailPage, GroupListPage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Semi-labeling Step5 Result Viewer")
        self.resize(1500, 950)
        self.feature_store: FeatureGroupStore | None = None
        self.source_store: SourceStore | None = None
        self.runs: list[GroupRun] = []
        self.clusters_by_label: dict[str, list[ClusterSummary]] = {label: [] for label in LABELS}
        self.current_run: GroupRun | None = None
        self.image_loader = ImageLoader()

        # ── Central: stacked widget ──────────────────────
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.group_tabs = QTabWidget()
        self.group_pages: dict[str, GroupListPage] = {}
        for label in LABELS:
            page = GroupListPage()
            page.opened.connect(self.open_group)
            self.group_pages[label] = page
            self.group_tabs.addTab(page, label)
        self.detail_page = DetailPage(self.image_loader)
        self.detail_page.back_requested.connect(self.show_group_list)
        self.detail_page.remove_group_flags_requested.connect(self.remove_current_group_flags)
        self.detail_page.remove_image_flags_requested.connect(self.remove_image_flags)
        self.stack.addWidget(self.group_tabs)
        self.stack.addWidget(self.detail_page)

        # ── Sidebar as a collapsible QDockWidget ─────────
        self._build_dock_sidebar()
        self.load_all()

    def _build_dock_sidebar(self) -> None:
        dock = QDockWidget("Settings", self)
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)

        # ── Database paths ───────────────────────────────
        db_group = QGroupBox("Database Paths")
        db_layout = QVBoxLayout(db_group)
        db_layout.setSpacing(4)

        self.feature_db_input = QLineEdit(str(default_feature_db()))
        self.source_db_input = QLineEdit(str(default_source_db()))
        self.image_root_input = QLineEdit(str(default_image_root()))

        db_layout.addWidget(QLabel("Feature DB"))
        db_layout.addWidget(self._path_row(self.feature_db_input, self._pick_feature_db))
        db_layout.addWidget(QLabel("Source DB"))
        db_layout.addWidget(self._path_row(self.source_db_input, self._pick_source_db))
        db_layout.addWidget(QLabel("Image root"))
        db_layout.addWidget(self._path_row(self.image_root_input, self._pick_image_root))

        self.load_button = QPushButton("Load SQLite")
        self.load_button.setFixedSize(140, 36)
        self.load_button.clicked.connect(self.load_all)
        db_layout.addWidget(self.load_button)
        layout.addWidget(db_group)

        # ── Run selection ────────────────────────────────
        run_group = QGroupBox("Run")
        run_layout = QVBoxLayout(run_group)
        run_layout.setSpacing(4)

        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self.on_run_changed)
        run_layout.addWidget(self.run_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("All clusters", "all")
        self.mode_combo.addItem("Non-outlier clusters", "non_outlier")
        self.mode_combo.addItem("Outliers only", "outlier")
        self.mode_combo.addItem("Label suspect only", "label_suspect")
        self.mode_combo.currentIndexChanged.connect(self.reload_clusters)
        run_layout.addWidget(QLabel("View mode"))
        run_layout.addWidget(self.mode_combo)
        layout.addWidget(run_group)

        # ── Display settings ─────────────────────────────
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(4)

        display_layout.addWidget(QLabel("Image size"))
        self.image_size = QSpinBox()
        self.image_size.setRange(100, 900)
        self.image_size.setSingleStep(20)
        self.image_size.setValue(240)
        self.image_size.valueChanged.connect(self.update_detail_settings)
        display_layout.addWidget(self.image_size)

        display_layout.addWidget(QLabel("Crop padding"))
        self.crop_padding = QDoubleSpinBox()
        self.crop_padding.setRange(0.0, 0.3)
        self.crop_padding.setSingleStep(0.01)
        self.crop_padding.setValue(0.05)
        self.crop_padding.valueChanged.connect(self.update_detail_settings)
        display_layout.addWidget(self.crop_padding)
        layout.addWidget(display_group)

        layout.addStretch(1)

        # ── Dock + toolbar ───────────────────────────────
        dock.setWidget(container)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.dock = dock

        toolbar = self.addToolBar("View")
        toolbar.setMovable(False)

        toggle_sidebar = dock.toggleViewAction()
        toggle_sidebar.setText("Toggle Sidebar")
        toolbar.addAction(toggle_sidebar)

        self.toggle_table_action = QAction("Toggle Table", self)
        self.toggle_table_action.setEnabled(False)   # enabled only on detail page
        self.toggle_table_action.triggered.connect(self._on_toggle_table)
        toolbar.addAction(self.toggle_table_action)

    # ── Helpers ──────────────────────────────────────────

    def _path_row(self, line_edit: QLineEdit, picker) -> QWidget:
        widget = QWidget()
        h = QHBoxLayout(widget)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        button = QPushButton("...")
        button.setFixedSize(48, 36)
        button.clicked.connect(picker)
        h.addWidget(line_edit, 1)
        h.addWidget(button)
        return widget

    def _pick_feature_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select feature_groups.sqlite3", self.feature_db_input.text(), "SQLite (*.sqlite3 *.db);;All files (*)")
        if path:
            self.feature_db_input.setText(path)

    def _pick_source_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select damage_scan.sqlite3", self.source_db_input.text(), "SQLite (*.sqlite3 *.db);;All files (*)")
        if path:
            self.source_db_input.setText(path)

    def _pick_image_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select HinhAnh folder", self.image_root_input.text())
        if path:
            self.image_root_input.setText(path)

    # ── Data logic ───────────────────────────────────────

    def load_all(self) -> None:
        feature_db = Path(self.feature_db_input.text()).expanduser()
        if not feature_db.is_file():
            QMessageBox.warning(self, "Missing DB", f"Feature DB not found:\n{feature_db}")
            return
        self.feature_store = FeatureGroupStore(feature_db)
        try:
            self.runs = self.feature_store.list_runs()
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run in self.runs:
            self.run_combo.addItem(
                f"{run.created_at_utc} | {run.grouping_run_id[:8]} | {run.total_clusters} groups | {run.model_name}",
                run.grouping_run_id,
            )
        self.run_combo.blockSignals(False)
        if self.runs:
            self.current_run = self.runs[0]
            if self.current_run.source_db_path and Path(self.current_run.source_db_path).is_file():
                self.source_db_input.setText(self.current_run.source_db_path)
        self.source_store = SourceStore(Path(self.source_db_input.text()).expanduser()) if Path(self.source_db_input.text()).expanduser().is_file() else None
        self.image_loader.clear()
        self.reload_clusters()

    def on_run_changed(self) -> None:
        idx = self.run_combo.currentIndex()
        if 0 <= idx < len(self.runs):
            self.current_run = self.runs[idx]
            if self.current_run.source_db_path and Path(self.current_run.source_db_path).is_file():
                self.source_db_input.setText(self.current_run.source_db_path)
            self.source_store = SourceStore(Path(self.source_db_input.text()).expanduser()) if Path(self.source_db_input.text()).expanduser().is_file() else None
            self.reload_clusters()

    def reload_clusters(self) -> None:
        if self.feature_store is None or self.current_run is None:
            return
        mode = str(self.mode_combo.currentData() or "all")
        self.clusters_by_label = {
            label: self.feature_store.list_clusters(self.current_run.grouping_run_id, label, mode)
            for label in LABELS
        }
        self.render_group_pages()

    def render_group_pages(self) -> None:
        for label, page in self.group_pages.items():
            page.set_groups(label, self.clusters_by_label.get(label, []))

    def open_group(self, label_scope: str, cluster_key: str) -> None:
        if self.feature_store is None or self.current_run is None:
            return
        cluster = next((item for item in self.clusters_by_label.get(label_scope, []) if item.cluster_key == cluster_key), None)
        if cluster is None:
            return
        rows = self.feature_store.list_assignments(self.current_run.grouping_run_id, cluster_key)
        if self.source_store is not None:
            meta = self.source_store.source_meta([int(row.result_id) for row in rows])
            rows = merge_source_meta(rows, meta)
        self.stack.setCurrentWidget(self.detail_page)
        self.toggle_table_action.setEnabled(True)
        self.detail_page.set_detail(
            cluster,
            rows,
            image_root=Path(self.image_root_input.text()).expanduser() if self.image_root_input.text().strip() else None,
            image_size=int(self.image_size.value()),
            padding_ratio=float(self.crop_padding.value()),
        )

    def update_detail_settings(self) -> None:
        self.detail_page.update_image_settings(
            image_size=int(self.image_size.value()),
            padding_ratio=float(self.crop_padding.value()),
        )

    def remove_current_group_flags(self) -> None:
        if self.feature_store is None or self.current_run is None or self.detail_page.cluster is None:
            return
        changed = self.feature_store.clear_flags_for_cluster(
            self.current_run.grouping_run_id,
            self.detail_page.cluster.cluster_key,
        )
        self.detail_page.clear_flags_for_all_rows()
        self.reload_clusters()
        QMessageBox.information(self, "Flags removed", f"Removed flags from {changed} rows in this group.")

    def remove_image_flags(self, result_ids: list[int]) -> None:
        if self.feature_store is None or self.current_run is None:
            return
        changed = self.feature_store.clear_flags_for_results(self.current_run.grouping_run_id, result_ids)
        self.detail_page.clear_flags_for_results({int(result_id) for result_id in result_ids})
        self.reload_clusters()
        QMessageBox.information(self, "Flags removed", f"Removed flags from {changed} rows for this image.")

    def show_group_list(self) -> None:
        self.stack.setCurrentWidget(self.group_tabs)
        self.toggle_table_action.setEnabled(False)

    def _on_toggle_table(self) -> None:
        self.detail_page.toggle_table()
