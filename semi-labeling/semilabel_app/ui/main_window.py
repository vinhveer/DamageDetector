from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..services.settings_service import SettingsService
from ..stores.prototype_store import PrototypeStore
from ..stores.review_store import ReviewStore
from ..stores.run_store import RunStore
from .workspaces.distribution_ws import DistributionWorkspace
from .workspaces.export_ws import ExportWorkspace
from .workspaces.prototype_ws import PrototypeWorkspace
from .workspaces.review_ws import ReviewWorkspace
from .workspaces.runsteps_ws import RunStepsWorkspace


class MainWindow(QtWidgets.QMainWindow):
    # key, label, QStyle.StandardPixmap name for the sidebar icon.
    NAV_ITEMS = [
        ("prototype", "Prototype", "SP_FileDialogContentsView"),
        ("review", "Đánh giá", "SP_FileDialogListView"),
        ("distribution", "Phân bố / QA", "SP_FileDialogInfoView"),
        ("runsteps", "Chạy bước", "SP_MediaPlay"),
        ("export", "Xuất dữ liệu", "SP_DialogSaveButton"),
    ]

    SIDEBAR_WIDTH = 200
    SIDEBAR_COLLAPSED = 56

    def __init__(
        self,
        *,
        settings: dict,
        settings_service: SettingsService,
        review_store: ReviewStore,
        prototype_store: PrototypeStore,
        run_store: RunStore,
        review_controller,
        prototype_controller,
        run_controller,
        export_controller,
    ) -> None:
        super().__init__()
        self.setWindowTitle("semilabel_app")
        self.resize(1500, 920)
        self._settings = settings
        self._settings_service = settings_service
        self._review_store = review_store
        self._prototype_store = prototype_store
        self._run_store = run_store
        self._review_controller = review_controller
        self._prototype_controller = prototype_controller
        self._run_controller = run_controller
        self._export_controller = export_controller
        self._nav_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._sidebar_collapsed = False

        central = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_top_bar(central))
        root.addWidget(self._build_settings_bar(central))

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        nav = QtWidgets.QFrame(central)
        nav.setObjectName("SideNav")
        nav.setFixedWidth(self.SIDEBAR_WIDTH)
        self._nav = nav
        nav_layout = QtWidgets.QVBoxLayout(nav)
        nav_layout.setContentsMargins(8, 12, 8, 12)
        nav_layout.setSpacing(4)
        style = self.style()
        for key, label, icon_name in self.NAV_ITEMS:
            btn = QtWidgets.QPushButton(f"  {label}", nav)
            btn.setCheckable(True)
            btn.setMinimumHeight(38)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            pixmap = getattr(QtWidgets.QStyle.StandardPixmap, icon_name, None)
            if pixmap is not None:
                btn.setIcon(style.standardIcon(pixmap))
                btn.setIconSize(QtCore.QSize(18, 18))
            btn.setToolTip(label)
            btn.clicked.connect(lambda _checked=False, k=key: self.show_workspace(k))
            nav_layout.addWidget(btn)
            self._nav_buttons[key] = btn
        nav_layout.addStretch(1)
        body.addWidget(nav)

        self._workspaces = QtWidgets.QStackedWidget(central)
        self._workspace_widgets = {
            "prototype": PrototypeWorkspace(
                self._prototype_store,
                self._run_store,
                self._prototype_controller,
                self._workspaces,
            ),
            "review": ReviewWorkspace(self._review_store, self._review_controller, self._workspaces),
            "distribution": DistributionWorkspace(
                self._review_store,
                self._review_controller,
                self._settings,
                self._workspaces,
            ),
            "runsteps": RunStepsWorkspace(
                self._run_store,
                self._run_controller,
                self._settings,
                self._workspaces,
            ),
            "export": ExportWorkspace(
                self._run_store,
                self._export_controller,
                self._settings,
                self._workspaces,
            ),
        }
        for widget in self._workspace_widgets.values():
            self._workspaces.addWidget(widget)
        body.addWidget(self._workspaces, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Sẵn sàng")
        self._build_actions()
        self.show_workspace("prototype")

    def _build_top_bar(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame(parent)
        bar.setObjectName("TopBar")
        bar.setFixedHeight(46)
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 12, 6)
        layout.setSpacing(8)

        style = self.style()
        self._collapse_btn = QtWidgets.QToolButton(bar)
        self._collapse_btn.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowLeft))
        self._collapse_btn.setAutoRaise(True)
        self._collapse_btn.setToolTip("Thu gọn thanh bên (Ctrl+B)")
        self._collapse_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._collapse_btn.clicked.connect(self._toggle_sidebar)
        layout.addWidget(self._collapse_btn)

        self._page_title = QtWidgets.QLabel("Prototype", bar)
        self._page_title.setObjectName("PageTitle")
        layout.addWidget(self._page_title)
        layout.addStretch(1)

        self._top_save_btn = QtWidgets.QPushButton("Lưu cài đặt", bar)
        self._top_save_btn.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        self._top_save_btn.clicked.connect(self._save_settings_from_bar)
        layout.addWidget(self._top_save_btn)
        return bar

    def _build_settings_bar(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame(parent)
        bar.setObjectName("SettingsBar")
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(8)

        self._db_edit = QtWidgets.QLineEdit(str(self._settings.get("db_path") or ""), bar)
        self._run_edit = QtWidgets.QLineEdit(str(self._settings.get("run_id") or "myrun"), bar)
        self._image_root_edit = QtWidgets.QLineEdit(str(self._settings.get("image_root") or ""), bar)
        self._model_edit = QtWidgets.QLineEdit(str(self._settings.get("model_name") or ""), bar)
        self._run_edit.setMaximumWidth(120)
        self._model_edit.setMaximumWidth(220)
        self._db_edit.setPlaceholderText("resemi.sqlite3")
        self._image_root_edit.setPlaceholderText("data/HinhAnh")

        layout.addWidget(QtWidgets.QLabel("CSDL", bar))
        layout.addWidget(self._db_edit, 3)
        layout.addWidget(QtWidgets.QLabel("Run", bar))
        layout.addWidget(self._run_edit, 0)
        layout.addWidget(QtWidgets.QLabel("Thư mục ảnh", bar))
        layout.addWidget(self._image_root_edit, 2)
        layout.addWidget(QtWidgets.QLabel("DINOv2", bar))
        layout.addWidget(self._model_edit, 0)
        save = QtWidgets.QPushButton("Lưu", bar)
        save.clicked.connect(self._save_settings_from_bar)
        layout.addWidget(save)
        return bar

    def _build_actions(self) -> None:
        for idx, (key, label, _icon) in enumerate(self.NAV_ITEMS, start=1):
            action = QtGui.QAction(f"Workspace: {label}", self)
            action.setShortcut(QtGui.QKeySequence(f"Ctrl+{idx}"))
            action.triggered.connect(lambda _checked=False, k=key: self.show_workspace(k))
            self.addAction(action)
        toggle = QtGui.QAction("Toggle sidebar", self)
        toggle.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        toggle.triggered.connect(self._toggle_sidebar)
        self.addAction(toggle)

    def _toggle_sidebar(self) -> None:
        self._sidebar_collapsed = not self._sidebar_collapsed
        style = self.style()
        if self._sidebar_collapsed:
            self._nav.setFixedWidth(self.SIDEBAR_COLLAPSED)
            self._collapse_btn.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowRight))
            for (key, label, _icon), button in zip(self.NAV_ITEMS, self._nav_buttons.values()):
                button.setText("")
        else:
            self._nav.setFixedWidth(self.SIDEBAR_WIDTH)
            self._collapse_btn.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowLeft))
            for (key, label, _icon), button in zip(self.NAV_ITEMS, self._nav_buttons.values()):
                button.setText(f"  {label}")

    def show_workspace(self, name: str) -> None:
        key = str(name)
        widget = self._workspace_widgets.get(key, self._workspace_widgets["prototype"])
        self._workspaces.setCurrentWidget(widget)
        for nav_key, button in self._nav_buttons.items():
            button.setChecked(nav_key == key)
        label = self._nav_label(key)
        self._page_title.setText(label)
        self.statusBar().showMessage(label)

    def _nav_label(self, key: str) -> str:
        for nav_key, label, _icon in self.NAV_ITEMS:
            if nav_key == key:
                return label
        return key

    def _current_settings_payload(self) -> dict:
        payload = dict(self._settings)
        payload.update(
            {
                "db_path": self._db_edit.text().strip(),
                "run_id": self._run_edit.text().strip() or "myrun",
                "image_root": self._image_root_edit.text().strip(),
                "model_name": self._model_edit.text().strip(),
            }
        )
        return payload

    def _save_settings_from_bar(self) -> None:
        self._settings.clear()
        self._settings.update(self._current_settings_payload())
        for controller in (
            self._review_controller,
            self._prototype_controller,
            self._run_controller,
            self._export_controller,
        ):
            controller.update_settings(self._settings)
        self._persist_state()
        self.statusBar().showMessage("Settings saved", 2500)

    def restore_persisted_state(self, payload: dict) -> None:
        workspace = str(payload.get("current_workspace") or "prototype")
        QtCore.QTimer.singleShot(0, lambda: self.show_workspace(workspace))

    def _persist_state(self) -> None:
        self._settings_service.save(
            {
                "settings": dict(self._current_settings_payload()),
                "current_workspace": self._active_workspace_key(),
            }
        )

    def _active_workspace_key(self) -> str:
        current = self._workspaces.currentWidget()
        for key, widget in self._workspace_widgets.items():
            if widget is current:
                return key
        return "prototype"

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._settings.clear()
        self._settings.update(self._current_settings_payload())
        self._persist_state()
        return super().closeEvent(event)
