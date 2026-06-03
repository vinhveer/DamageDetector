from __future__ import annotations

from .run_controller import RunController
from ..stores.run_store import RunStore


class ExportController:
    def __init__(self, store: RunStore, settings: dict) -> None:
        self.store = store
        self.settings = settings
        self._run_controller = RunController(store, settings)

    def update_settings(self, settings: dict) -> None:
        self.settings.update(settings)
        self._run_controller.update_settings(self.settings)

    def export_dataset(self, output_dir: str | None = None, fmt: str | None = None) -> None:
        flags = {
            "--db": self.settings["db_path"],
            "--run-id": self.settings.get("run_id", "myrun"),
            "--image-root": self.settings.get("image_root", ""),
            "--output-dir": output_dir or self.settings.get("export_dir", ""),
            "--format": fmt or self.settings.get("export_format", "yolo"),
        }
        self._run_controller.run_step("export_dataset", flags)
