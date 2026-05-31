"""Import bootstrap for the step/tool CLIs.

Ensures the ``semi-labeling`` directory is on sys.path so the top-level
``lib`` / ``steps`` / ``tools`` packages import cleanly whether invoked as
``python -m steps.stepNN`` or as a direct file path.
"""
from __future__ import annotations

import sys
from pathlib import Path

# bootstrap.py -> lib/ -> semi-labeling/
_SEMI_LABELING_DIR = Path(__file__).resolve().parents[1]


def ensure_on_path() -> None:
    parent = str(_SEMI_LABELING_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def ensure_embedder_on_path() -> None:
    """step03 needs the DINOv2 embedder from old/step3_embedding/."""
    ensure_on_path()
    step3 = _SEMI_LABELING_DIR / "old" / "step3_embedding"
    if step3.is_dir() and str(step3) not in sys.path:
        sys.path.insert(0, str(step3))
