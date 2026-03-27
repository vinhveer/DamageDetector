from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtWidgets

from editor_app.config.prediction_settings import default_prediction_config, prediction_summary
from editor_app.controllers.compare_controller import CompareController
from editor_app.controllers.editor_controller import EditorController
from editor_app.controllers.history_controller import HistoryController
from editor_app.controllers.prediction_controller import PredictionController
from editor_app.stores.prediction_store import PredictionStore
from editor_app.stores.ui_store import UiStore
from editor_app.stores.workspace_store import WorkspaceStore
from editor_app.ui.dialogs import PredictRunDialog
from inference_api.prediction_models import PredictionConfig, SCOPE_CURRENT


class PredictionActions(QtCore.QObject):
    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget,
        workspace_store: WorkspaceStore,
        ui_store: UiStore,
        prediction_store: PredictionStore,
        prediction_controller: PredictionController,
        editor_controller: EditorController,
        history_controller: HistoryController,
        compare_controller: CompareController,
        show_workspace: Callable[[str], None],
        show_error: Callable[[str], None],
        show_status: Callable[[str, int], None],
        persist_state: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self._workspace_store = workspace_store
        self._ui_store = ui_store
        self._prediction_store = prediction_store
        self._prediction_controller = prediction_controller
        self._editor_controller = editor_controller
        self._history_controller = history_controller
        self._compare_controller = compare_controller
        self._show_workspace = show_workspace
        self._show_error = show_error
        self._show_status = show_status
        self._persist_state = persist_state
        self._pending_roi_submission: dict | None = None

    def run_predict_dialog(self) -> None:
        dlg = PredictRunDialog(
            self._parent,
            has_image=bool(self._workspace_store.current_image_path),
            has_folder=bool(self._workspace_store.images),
            default_config=default_prediction_config(),
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        config = dlg.get_result()
        settings = dict(self._ui_store.settings)
        self._ui_store.set_settings(settings)
        job_id = self._prediction_controller.submit(config=config, settings=settings)
        if job_id:
            self._prediction_store.set_active_job(job_id)
            self._show_workspace("runs")
            self._persist_state()
            self._show_status(f"Queued {prediction_summary(config)} with saved settings.", 4000)

    def run_predict_dialog_roi(self, start_roi_selection: Callable[[], None]) -> None:
        if not self._workspace_store.current_image_path:
            self._show_error("Open an image before running ROI prediction.")
            return
        dlg = PredictRunDialog(
            self._parent,
            has_image=True,
            has_folder=False,
            default_config=default_prediction_config(),
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        config = dlg.get_result()
        settings = dict(self._ui_store.settings)
        self._ui_store.set_settings(settings)
        roi = self._workspace_store.current_roi_box()
        if roi is not None:
            job_id = self._prediction_controller.submit(
                config=PredictionConfig(
                    task_group=config.task_group,
                    segmentation_model=config.segmentation_model,
                    detection_model=config.detection_model,
                    scope=SCOPE_CURRENT,
                ),
                settings=settings,
                roi_box=roi,
            )
            if job_id:
                self._prediction_store.set_active_job(job_id)
                self._show_workspace("runs")
                self._persist_state()
                self._show_status(f"Queued {prediction_summary(config)} on selected ROI.", 4000)
            return
        self._pending_roi_submission = {"config": config.normalized(), "settings": settings}
        self._show_workspace("editor")
        start_roi_selection()
        self._show_status(f"Select ROI to run {prediction_summary(config)} with saved settings.", 5000)

    def run_isolate(self) -> None:
        if not self._workspace_store.current_image_path:
            self._show_error("Open an image before running isolate.")
            return
        settings = dict(self._ui_store.settings)
        labels = [part.strip() for part in str(settings.get("isolate_labels") or "").split(",") if part.strip()]
        job_id = self._prediction_controller.submit_isolate(
            settings=dict(settings),
            target_labels=labels,
            outside_value=255 if bool(settings.get("isolate_outside_white") or False) else 0,
            crop_to_bbox=bool(settings.get("isolate_crop") or False),
        )
        if job_id:
            self._prediction_store.set_active_job(job_id)
            self._show_workspace("runs")
            self._persist_state()
            self._show_status("Queued isolate with saved settings.", 4000)

    def on_roi_box_selected(self, roi_box_obj) -> None:
        pending = dict(self._pending_roi_submission or {})
        self._pending_roi_submission = None
        if not pending:
            return
        if roi_box_obj is None:
            self._show_status("ROI selection cancelled.", 3000)
            return
        roi = tuple(int(x) for x in roi_box_obj)
        self._editor_controller.add_roi(roi)
        config = pending.get("config")
        if isinstance(config, dict):
            config = PredictionConfig(
                task_group=str(config.get("task_group") or ""),
                segmentation_model=str(config.get("segmentation_model") or ""),
                detection_model=str(config.get("detection_model") or ""),
                scope=str(config.get("scope") or SCOPE_CURRENT),
            )
        if config is None:
            return
        job_id = self._prediction_controller.submit(
            config=PredictionConfig(
                task_group=config.task_group,
                segmentation_model=config.segmentation_model,
                detection_model=config.detection_model,
                scope=SCOPE_CURRENT,
            ),
            settings=dict(pending.get("settings") or {}),
            roi_box=roi,
        )
        if job_id:
            self._prediction_store.set_active_job(job_id)
            self._show_workspace("runs")
            self._persist_state()

    def on_roi_canceled(self) -> None:
        if self._pending_roi_submission is not None:
            self._pending_roi_submission = None
            self._show_status("ROI selection cancelled.", 3000)

    def load_run_item_into_editor(self, payload: dict) -> str | None:
        try:
            image_path = self._editor_controller.load_history_item(payload)
        except Exception as exc:
            self._show_error(str(exc))
            return None
        self._show_workspace("editor")
        self._show_status(f"Loaded run item: {Path(image_path).name}", 5000)
        return image_path

    def on_history_run_selected(self, run_dir: str, set_selected_run_details: Callable[[str, dict, list[dict]], None]) -> tuple[dict, list[dict]]:
        if not run_dir:
            return {}, []
        bundle, items = self._history_controller.load_run_details(run_dir)
        set_selected_run_details(run_dir, bundle=bundle, items=items)
        return bundle, items

    def compare_selected_run(self, run_dir: str) -> None:
        try:
            results = self._compare_controller.run_compare(run_dir=run_dir)
        except Exception as exc:
            self._show_error(str(exc))
            return
        if not results:
            QtWidgets.QMessageBox.information(self._parent, "Compare", "No matching masks found for this run.")
            return
        self._show_workspace("compare")
