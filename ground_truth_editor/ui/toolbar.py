from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class ToolbarActions:
    predict_with: QtGui.QAction
    isolate_object: QtGui.QAction
    model_settings: QtGui.QAction
    open_folder: QtGui.QAction
    open_image: QtGui.QAction
    open_mask: QtGui.QAction
    save_mask: QtGui.QAction
    prev_image: QtGui.QAction
    next_image: QtGui.QAction
    stop: QtGui.QAction
    folder_history: QtGui.QAction


class ToolbarController:
    """
    Renders a top toolbar that can *fully hide* buttons based on action state.

    We intentionally rebuild the toolbar items instead of relying on QWidget visibility,
    because some styles/platforms still show disabled text or leave gaps/separators.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        self._window = window
        self._tb = QtWidgets.QToolBar("Main", window)
        self._tb.setMovable(False)
        self._tb.setFloatable(False)
        self._tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        window.addToolBar(self._tb)

        self._actions: ToolbarActions | None = None
        self._quick_label = QtWidgets.QLabel(
            "Views: Ctrl+1 Overlay, Ctrl+2 Image, Ctrl+3 Mask, Ctrl+4 Explorer",
            window,
        )


        self._last_signature: tuple[int, ...] | None = None

    def set_actions(self, actions: ToolbarActions) -> None:
        self._actions = actions
        self.render()

    def render(self) -> None:
        if self._actions is None:
            return

        a = self._actions

        groups: list[list[QtGui.QAction]] = [
            [a.predict_with],
            [a.model_settings],
            [a.open_folder, a.open_image],
            [a.open_mask, a.save_mask],
            [a.prev_image, a.next_image],
            [a.stop],
            [a.folder_history],
            [a.isolate_object],
        ]

        visible_groups: list[list[QtGui.QAction]] = []
        for g in groups:
            vis = [x for x in g if x.isEnabled()]
            if vis:
                visible_groups.append(vis)

        signature = tuple(id(x) for g in visible_groups for x in g)
        if signature == self._last_signature:
            return
        self._last_signature = signature

        self._tb.clear()

        first = True
        for g in visible_groups:
            if not first:
                self._tb.addSeparator()
            for act in g:
                self._tb.addAction(act)
            first = False

        spacer = QtWidgets.QWidget(self._window)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self._tb.addWidget(spacer)
        self._tb.addWidget(self._quick_label)
