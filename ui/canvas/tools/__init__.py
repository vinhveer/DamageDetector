"""Canvas tools: pan, select, rect ROI, etc."""
from ui.canvas.tools.base import Tool
from ui.canvas.tools.builtin import PanTool, RectRoiTool, SelectTool

TOOL_REGISTRY: dict[str, type[Tool]] = {
    PanTool.name: PanTool,
    SelectTool.name: SelectTool,
    RectRoiTool.name: RectRoiTool,
}

__all__ = ["TOOL_REGISTRY", "Tool", "PanTool", "RectRoiTool", "SelectTool"]
