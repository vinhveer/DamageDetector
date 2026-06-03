from __future__ import annotations

import sys
from pathlib import Path


def _ensure_semilabeling_on_path() -> None:
    semi_labeling_dir = Path(__file__).resolve().parents[1] / "semi-labeling"
    path = str(semi_labeling_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    _ensure_semilabeling_on_path()
    from semilabel_app.app import main as app_main

    app_main()
