"""Import bootstrap for resemi step/tool CLIs.

Ensures the ``semi-labeling`` directory (which contains the ``resemi`` package)
is on sys.path so the modules import cleanly whether invoked as
``python -m resemi.steps.stepNN`` or as a direct file path.
"""
from __future__ import annotations

import sys
from pathlib import Path

# bootstrap.py -> lib/ -> resemi/ -> semi-labeling/
_SEMI_LABELING_DIR = Path(__file__).resolve().parents[2]


def ensure_on_path() -> None:
    parent = str(_SEMI_LABELING_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def ensure_embedder_on_path() -> None:
    """step03 needs the DINOv2 embedder from step3_embedding/."""
    ensure_on_path()
    step3 = _SEMI_LABELING_DIR / "step3_embedding"
    if step3.is_dir() and str(step3) not in sys.path:
        sys.path.insert(0, str(step3))
