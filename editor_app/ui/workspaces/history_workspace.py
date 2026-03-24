from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from editor_app.domain.models import RunSummary


class HistoryWorkspace(QtWidgets.QWidget):
    openRunRequested = QtCore.Signal(str)
    loadItemRequested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._run_bundle_by_dir: dict[str, dict] = {}
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title_host = QtWidgets.QVBoxLayout()
        title_host.setSpacing(2)
        title = QtWidgets.QLabel("Run History", self)
        self._subtitle = QtWidgets.QLabel("Browse previous runs and reopen outputs.", self)
        title_host.addWidget(title)
        title_host.addWidget(self._subtitle)
        header.addLayout(title_host)
        header.addStretch(1)
        self._refresh_btn = QtWidgets.QPushButton("Refresh", self)
        header.addWidget(self._refresh_btn)
        root.addLayout(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self._run_list = QtWidgets.QTreeWidget(splitter)
        self._run_list.setHeaderLabels(["Run ID", "Workflow", "Status", "Created"])
        self._run_list.setRootIsDecorated(False)
        self._run_list.setAlternatingRowColors(True)
        self._run_list.setUniformRowHeights(True)
        self._run_list.itemSelectionChanged.connect(self._show_selected_run)
        self._run_list.itemDoubleClicked.connect(self._emit_open_run)

        right_host = QtWidgets.QWidget(splitter)
        right_layout = QtWidgets.QVBoxLayout(right_host)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)
        self._right_tabs = QtWidgets.QTabWidget(right_host)
        right_layout.addWidget(self._right_tabs, 1)

        detail_host = QtWidgets.QWidget(self._right_tabs)
        detail_layout = QtWidgets.QVBoxLayout(detail_host)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(10)

        self._detail_scroll = QtWidgets.QScrollArea(detail_host)
        self._detail_scroll.setWidgetResizable(True)
        detail_layout.addWidget(self._detail_scroll, 1)

        self._detail_content = QtWidgets.QWidget(self._detail_scroll)
        self._detail_scroll.setWidget(self._detail_content)
        self._detail_content_layout = QtWidgets.QVBoxLayout(self._detail_content)
        self._detail_content_layout.setContentsMargins(0, 0, 0, 0)
        self._detail_content_layout.setSpacing(12)

        self._run_summary_group = self._make_group("Run Summary")
        self._run_summary_form = self._detail_stack_layout(self._run_summary_group)
        self._run_id_value = self._value_label()
        self._workflow_value = self._value_label()
        self._status_value = self._value_label()
        self._created_value = self._value_label()
        self._scope_value = self._value_label()
        self._add_detail_stack_row(self._run_summary_form, "Run ID", self._run_id_value)
        self._add_detail_stack_row(self._run_summary_form, "Workflow", self._workflow_value)
        self._add_detail_stack_row(self._run_summary_form, "Status", self._status_value)
        self._add_detail_stack_row(self._run_summary_form, "Created", self._created_value)
        self._add_detail_stack_row(self._run_summary_form, "Scope", self._scope_value)
        self._detail_content_layout.addWidget(self._run_summary_group)

        self._artifact_group = self._make_group("Artifacts")
        self._artifact_form = self._detail_stack_layout(self._artifact_group)
        self._run_dir_value = self._path_value()
        self._output_dir_value = self._path_value()
        self._items_value = self._value_label()
        self._add_detail_stack_row(self._artifact_form, "Run folder", self._run_dir_value)
        self._add_detail_stack_row(self._artifact_form, "Output folder", self._output_dir_value)
        self._add_detail_stack_row(self._artifact_form, "Items", self._items_value)
        self._detail_content_layout.addWidget(self._artifact_group)

        self._request_group = self._make_group("Request Parameters")
        self._request_form = self._detail_stack_layout(self._request_group)
        self._request_value = QtWidgets.QPlainTextEdit(self._detail_content)
        self._request_value.setReadOnly(True)
        self._request_value.setPlaceholderText("No request metadata available.")
        self._request_value.setMinimumHeight(220)
        self._add_detail_stack_row(self._request_form, "Parameters", self._request_value)
        self._detail_content_layout.addWidget(self._request_group)
        self._detail_content_layout.addStretch(1)

        item_host = QtWidgets.QWidget(self._right_tabs)
        item_layout = QtWidgets.QVBoxLayout(item_host)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(10)
        self._item_list = QtWidgets.QTreeWidget(item_host)
        self._item_list.setHeaderLabels(["Image", "Masks", "Overlay", "Isolate"])
        self._item_list.setRootIsDecorated(False)
        self._item_list.setAlternatingRowColors(True)
        self._item_list.setUniformRowHeights(True)
        self._item_list.itemSelectionChanged.connect(self._show_selected_item)
        self._item_list.itemDoubleClicked.connect(self._emit_load_current_item)
        item_layout.addWidget(self._item_list, 1)
        item_buttons = QtWidgets.QHBoxLayout()
        self._load_btn = QtWidgets.QPushButton("Load History Into Editor", item_host)
        self._load_btn.clicked.connect(self._emit_load_current_item)
        self._open_btn = QtWidgets.QPushButton("Open Run Folder", item_host)
        self._open_btn.clicked.connect(self._emit_open_current)
        item_buttons.addWidget(self._load_btn)
        item_buttons.addWidget(self._open_btn)
        item_layout.addLayout(item_buttons)
        self._right_tabs.addTab(detail_host, "Details")
        self._right_tabs.addTab(item_host, "Items")

        splitter.addWidget(self._run_list)
        splitter.addWidget(right_host)
        splitter.setSizes([360, 980])
        root.addWidget(splitter, 1)
        self._run_items_by_dir: dict[str, list[dict]] = {}
        self._load_btn.setEnabled(False)

    def refresh_button(self) -> QtWidgets.QPushButton:
        return self._refresh_btn

    def set_runs(
        self,
        runs: list[RunSummary],
        *,
        run_items_by_dir: dict[str, list[dict]] | None = None,
        run_bundles_by_dir: dict[str, dict] | None = None,
    ) -> None:
        self._run_items_by_dir = dict(run_items_by_dir or {})
        self._run_bundle_by_dir = dict(run_bundles_by_dir or {})
        self._run_list.clear()
        self._item_list.clear()
        for run in runs:
            item = QtWidgets.QTreeWidgetItem(
                [
                    run.run_id,
                    run.workflow,
                    run.status,
                    run.created_at,
                ]
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, run.run_dir)
            self._style_status_item(item, run.status)
            self._run_list.addTopLevelItem(item)
        self._subtitle.setText(f"{len(runs)} runs available")
        if self._run_list.topLevelItemCount() > 0:
            self._run_list.setCurrentItem(self._run_list.topLevelItem(0))
        else:
            self._clear_details()

    def _show_selected_run(self) -> None:
        item = self._run_list.currentItem()
        if item is None:
            self._item_list.clear()
            self._clear_details()
            return
        run_dir = Path(str(item.data(0, QtCore.Qt.ItemDataRole.UserRole) or ""))
        self._item_list.clear()
        items = list(self._run_items_by_dir.get(str(run_dir), []))
        for entry in items:
            image_path = str(entry.get("image_path") or "")
            image_name = Path(image_path).name if image_path else f"item_{entry.get('_item_index', 0)}"
            has_mask = "yes" if entry.get("mask_path") else ""
            has_overlay = "yes" if entry.get("overlay_path") else ""
            has_isolate = "yes" if entry.get("isolate_path") else ""
            child = QtWidgets.QTreeWidgetItem([image_name, has_mask, has_overlay, has_isolate])
            child.setData(0, QtCore.Qt.ItemDataRole.UserRole, dict(entry))
            self._item_list.addTopLevelItem(child)
        if self._item_list.topLevelItemCount() > 0:
            self._item_list.setCurrentItem(self._item_list.topLevelItem(0))
            self._load_btn.setEnabled(True)
        bundle = dict(self._run_bundle_by_dir.get(str(run_dir)) or {})
        run_meta = dict(bundle.get("run") or {})
        request_meta = dict(bundle.get("request") or {})
        self._run_id_value.setText(str(run_meta.get("run_id") or run_dir.name))
        self._workflow_value.setText(str(run_meta.get("workflow") or "-"))
        self._set_status_label(str(run_meta.get("status") or "-"))
        self._created_value.setText(str(run_meta.get("created_at") or "-"))
        self._scope_value.setText(str(run_meta.get("scope") or "-"))
        self._run_dir_value.setText(str(run_meta.get("run_dir") or run_dir))
        self._output_dir_value.setText(str(run_meta.get("output_dir") or (run_dir / "outputs")))
        self._items_value.setText(str(len(items)))
        self._request_value.setPlainText(self._format_request_payload(request_meta))

    def _show_selected_item(self) -> None:
        item = self._item_list.currentItem()
        self._load_btn.setEnabled(item is not None)

    def _emit_open_current(self) -> None:
        item = self._run_list.currentItem()
        if item is None:
            return
        run_dir = str(item.data(0, QtCore.Qt.ItemDataRole.UserRole) or "")
        if run_dir:
            self.openRunRequested.emit(run_dir)

    def _emit_open_run(self, item: QtWidgets.QTreeWidgetItem) -> None:
        if item is None:
            return
        run_dir = str(item.data(0, QtCore.Qt.ItemDataRole.UserRole) or "")
        if run_dir:
            self.openRunRequested.emit(run_dir)

    def _emit_load_current_item(self, *_args) -> None:
        item = self._item_list.currentItem()
        if item is None:
            return
        payload = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            self.loadItemRequested.emit(dict(payload))

    def _make_group(self, title: str) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title, self._detail_content)
        font = group.font()
        font.setBold(False)
        group.setFont(font)
        return group

    def _detail_stack_layout(self, parent: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        return layout

    def _add_detail_stack_row(self, layout: QtWidgets.QVBoxLayout, label: str, widget: QtWidgets.QWidget) -> None:
        block = QtWidgets.QWidget(self._detail_content)
        block_layout = QtWidgets.QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(6)
        title = QtWidgets.QLabel(label, block)
        title_font = title.font()
        title_font.setBold(False)
        title.setFont(title_font)
        block_layout.addWidget(title)
        block_layout.addWidget(widget)
        layout.addWidget(block)

    def _value_label(self) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel("-", self._detail_content)
        label.setWordWrap(True)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        font = label.font()
        font.setPointSize(font.pointSize() + 2)
        label.setFont(font)
        return label

    def _path_value(self) -> QtWidgets.QLineEdit:
        widget = QtWidgets.QLineEdit(self._detail_content)
        widget.setReadOnly(True)
        widget.setPlaceholderText("-")
        widget.setMinimumHeight(34)
        return widget

    def _style_status_item(self, item: QtWidgets.QTreeWidgetItem, status: str) -> None:
        _ = (item, status)

    def _set_status_label(self, status: str) -> None:
        text = str(status or "-")
        self._status_value.setText(text)
        font = self._status_value.font()
        font.setBold(False)
        self._status_value.setFont(font)

    def _format_request_payload(self, payload: dict) -> str:
        if not payload:
            return ""
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            return str(payload)

    def _clear_details(self) -> None:
        self._run_id_value.setText("-")
        self._workflow_value.setText("-")
        self._created_value.setText("-")
        self._scope_value.setText("-")
        self._set_status_label("-")
        self._run_dir_value.setText("")
        self._output_dir_value.setText("")
        self._items_value.setText("0")
        self._request_value.setPlainText("")
        self._load_btn.setEnabled(False)
