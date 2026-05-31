"""Import bootstrap for the step/tool CLIs.

Ensures the ``semi-labeling`` directory is on sys.path so the top-level
``shared`` / ``steps`` / ``tools`` packages import cleanly whether invoked as
``python -m steps.stepNN.main`` or as a direct file path.
"""
from __future__ import annotations

import sys
from pathlib import Path

# bootstrap.py -> runtime/ -> shared/ -> semi-labeling/
_SEMI_LABELING_DIR = Path(__file__).resolve().parents[2]


def ensure_on_path() -> None:
    parent = str(_SEMI_LABELING_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def ensure_repo_root_on_path() -> None:
    """step03's DINOv2 embedder imports `torch_runtime` from the DamageDetector
    repo root, so make sure that root is importable."""
    ensure_on_path()
    repo_root = _SEMI_LABELING_DIR.parent  # semi-labeling/ -> DamageDetector/
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
