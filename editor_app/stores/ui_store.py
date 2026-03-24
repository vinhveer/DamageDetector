from __future__ import annotations

from PySide6 import QtCore


class UiStore(QtCore.QObject):
    workspaceViewChanged = QtCore.Signal(str)
    settingsChanged = QtCore.Signal()
    layoutChanged = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.current_workspace_view: str = "editor"
        self.settings: dict = {}
        self.main_splitter_sizes: list[int] = []
        self.left_splitter_sizes: list[int] = []

    def set_workspace_view(self, name: str) -> None:
        self.current_workspace_view = str(name or "editor")
        self.workspaceViewChanged.emit(self.current_workspace_view)

    def set_settings(self, settings: dict) -> None:
        self.settings = dict(settings or {})
        self.settingsChanged.emit()

    def set_layout(self, *, main_splitter_sizes: list[int] | None = None, left_splitter_sizes: list[int] | None = None) -> None:
        if main_splitter_sizes is not None:
            self.main_splitter_sizes = [int(value) for value in main_splitter_sizes]
        if left_splitter_sizes is not None:
            self.left_splitter_sizes = [int(value) for value in left_splitter_sizes]
        self.layoutChanged.emit()
