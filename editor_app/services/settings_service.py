from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6 import QtCore

from editor_app.paths import repo_root


class SettingsService:
    def __init__(self) -> None:
        self._legacy_path = repo_root() / ".editor_app.json"
        self._path = self._resolve_settings_path()

    def _resolve_settings_path(self) -> Path:
        location = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppConfigLocation)
        if location:
            return Path(location) / "editor_app.json"
        return self._legacy_path

    def load(self) -> dict[str, Any]:
        source = self._path if self._path.is_file() else self._legacy_path
        if not source.is_file():
            return {}
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if source == self._legacy_path and self._path != self._legacy_path:
                try:
                    self.save(payload)
                except Exception:
                    pass
            return payload
        except Exception:
            return {}

    def save(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str) + "\n", encoding="utf-8")
