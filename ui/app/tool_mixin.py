from __future__ import annotations

from PySide6 import QtCore

from ui.canvas.tools import RectRoiTool
from ui.core.roi_commands import AddRoiCommand, DeleteRoiCommand, MoveRoiCommand


class ToolMixin:
    def _wire_tool_signals(self) -> None:
        self._tools_palette.toolSelected.connect(self._set_active_tool)
        rect_tool = self._tools.get("rect_roi")
        if isinstance(rect_tool, RectRoiTool):
            rect_tool.roiCommitted.connect(self._on_roi_committed)
        self._canvas.roiMoved.connect(self._on_roi_moved)

    def _set_active_tool(self, name: str) -> None:
        tool = self._tools.get(name)
        if tool is None:
            return
        self._canvas.set_active_tool(tool)
        self._tools_palette.set_active(name)
        self._tools_palette.set_options_widget(tool.options_widget())
        self._update_tool_status(tool.label)
        self._canvas.setFocus()

    def _update_tool_status(self, tool_label: str) -> None:
        if hasattr(self, "_sb_tool"):
            self._sb_tool.setText(f"Tool: {tool_label}")
        else:
            self.statusBar().showMessage(f"Tool: {tool_label}")

    def _on_roi_committed(self, rect: QtCore.QRectF) -> None:
        if self._canvas.image_rect().isEmpty():
            return
        self._undo.push(AddRoiCommand(self._canvas, rect))

    def _on_roi_moved(self, roi_index: int, before: QtCore.QPointF, after: QtCore.QPointF) -> None:
        # Don't push if this was triggered by undo/redo
        self._undo.push(MoveRoiCommand(self._canvas, roi_index, before, after))

    def _delete_selected_rois(self) -> None:
        selected = [item for item in self._canvas.roi_items() if item.isSelected()]
        if not selected:
            return
        for item in selected:
            rect = item.mapRectToScene(item.rect()).normalized()
            self._undo.push(DeleteRoiCommand(self._canvas, item.roi_index, rect))

    def _on_rois_changed(self) -> None:
        self._sync_state()
