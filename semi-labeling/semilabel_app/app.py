from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtWidgets

from .config.defaults import DEFAULT_SETTINGS, migrate_settings
from .controllers.export_controller import ExportController
from .controllers.prototype_controller import PrototypeController
from .controllers.review_controller import ReviewController
from .controllers.run_controller import RunController
from .services.settings_service import SettingsService
from .stores.prototype_store import PrototypeStore
from .stores.review_store import ReviewStore
from .stores.run_store import RunStore
from .ui.main_window import MainWindow


def build_window() -> MainWindow:
    settings_service = SettingsService()
    persisted = settings_service.load()
    settings = dict(DEFAULT_SETTINGS)
    settings.update(migrate_settings(dict(persisted.get("settings") or {})))

    review_store = ReviewStore()
    prototype_store = PrototypeStore()
    run_store = RunStore()

    review_controller = ReviewController(review_store, settings)
    prototype_controller = PrototypeController(prototype_store, run_store, settings)
    run_controller = RunController(run_store, settings)
    export_controller = ExportController(run_store, settings)

    window = MainWindow(
        settings=settings,
        settings_service=settings_service,
        review_store=review_store,
        prototype_store=prototype_store,
        run_store=run_store,
        review_controller=review_controller,
        prototype_controller=prototype_controller,
        run_controller=run_controller,
        export_controller=export_controller,
    )
    window.restore_persisted_state(persisted)
    return window


def run() -> None:
    QtCore.QLoggingCategory.setFilterRules("qt.accessibility.table.warning=false")
    app = QtWidgets.QApplication(sys.argv)
    ideal = max(2, (QtCore.QThread.idealThreadCount() or 4) - 2)
    QtCore.QThreadPool.globalInstance().setMaxThreadCount(min(8, ideal))
    app.setApplicationName("semilabel_app")
    app.setOrganizationName("DamageDetector")
    window = build_window()
    window.show()
    if os.environ.get("SEMILABEL_APP_SMOKE") == "1":
        QtCore.QTimer.singleShot(250, app.quit)
    raise SystemExit(app.exec())


def main() -> None:
    run()
