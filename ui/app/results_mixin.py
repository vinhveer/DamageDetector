from __future__ import annotations

from PySide6 import QtWidgets

from ui.models.layer import LayerKind
from ui.services.detect_process import DetectionRow


class ResultsMixin:
    def _refresh_results(self) -> None:
        # Canvas shows boxes for every visible detection group; the table shows
        # the active group's filtered rows only.
        self._render_groups()
        rows = self._filtered_rows()

        table = self._detect_panel.table
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(row.roi_index)))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(row.group_name))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{row.score:.3f}"))
            table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{int(row.x1)},{int(row.y1)},{int(row.x2)},{int(row.y2)}"))
        self._sync_state()

    def _filtered_rows(self) -> list[DetectionRow]:
        if not hasattr(self, "_detect_panel"):
            return []
        min_score = float(self._detect_panel.min_score.value())
        cls = self._detect_panel.class_filter.currentData()
        rows = [row for row in self._rows if row.score >= min_score]
        if cls and cls != "all":
            rows = [row for row in rows if row.group_name == cls]
        return rows

    def _on_table_selected(self) -> None:
        rows = self._filtered_rows()
        idx = self._detect_panel.table.currentRow()
        if 0 <= idx < len(rows):
            self._canvas.select_roi(rows[idx].roi_index)

    def _on_layer_visibility(self, layer_id: str, visible: bool) -> None:
        layer = self._layers.by_id(layer_id)
        if layer is None:
            return
        if layer.kind == LayerKind.rois:
            for item in self._canvas.roi_items():
                item.setVisible(visible)
        elif layer.kind == LayerKind.detections:
            # Re-render so box visibility + emphasis stay consistent, then apply
            # mask visibility for this specific group.
            self._render_groups()
            self._canvas.set_group_visible(layer_id, visible)

    def _on_layer_opacity(self, layer_id: str, opacity: float) -> None:
        layer = self._layers.by_id(layer_id)
        if layer is None:
            return
        if layer.kind == LayerKind.detections:
            self._canvas.set_group_opacity(layer_id, opacity)
