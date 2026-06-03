from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def make_handoff_path(db_path: str | Path, *, kind: str, run_id: str) -> Path:
    db = Path(db_path).expanduser().resolve()
    root = db.parent / "handoff"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(run_id or "run")).strip("_") or "run"
    safe_kind = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(kind or "request")).strip("_") or "request"
    return root / f"{safe_kind}_{safe_run}_{stamp}.json"


def write_handoff_json(db_path: str | Path, payload: dict[str, Any], *, kind: str, run_id: str) -> Path:
    path = make_handoff_path(db_path, kind=kind, run_id=run_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_handoff_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("handoff JSON must be an object")
    return payload