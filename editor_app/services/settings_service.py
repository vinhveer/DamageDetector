from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from editor_app.paths import repo_root


class SettingsService:
    def __init__(self) -> None:
        self._path = repo_root() / ".editor_app.json"

    def load(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save(self, payload: dict[str, Any]) -> None:
        self._path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str) + "\n", encoding="utf-8")
