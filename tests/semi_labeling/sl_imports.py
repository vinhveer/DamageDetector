"""Isolated import of step modules.

The ``step*`` packages use bare imports (``from output_store import ...``) and several
steps define modules with the same name (``output_store``, ``source_store``, ...). To
let one pytest session import modules from multiple steps, ``load_step`` ensures only
the target step dir is on ``sys.path`` (among step dirs) and purges cached bare modules
that were loaded from a different step dir before importing.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # DamageDetector/
SEMILABEL = ROOT / "semi-labeling"
_ALL_STEP_DIRS = (
    {str(p.resolve()) for p in SEMILABEL.iterdir() if p.is_dir()} if SEMILABEL.is_dir() else set()
)


def load_step(step: str, name: str):
    """Import top-level module ``name`` from ``semi-labeling/<step>/``."""
    step_dir = str((SEMILABEL / step).resolve())
    # Purge bare top-level modules cached from a different step dir.
    for mod_name, mod in list(sys.modules.items()):
        if "." in mod_name:
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        parent = str(Path(f).resolve().parent)
        if parent in _ALL_STEP_DIRS and parent != step_dir:
            del sys.modules[mod_name]
    # Keep only the target step dir (among step dirs) on sys.path, at the front.
    sys.path[:] = [p for p in sys.path if str(Path(p).resolve()) not in _ALL_STEP_DIRS]
    sys.path.insert(0, step_dir)
    return importlib.import_module(name)


def load_step_file(step: str, relpath: str, modname: str):
    """Load a standalone .py file at semi-labeling/<step>/<relpath> under a unique
    module name (for subpackage modules that import only stdlib/numpy)."""
    import importlib.util

    path = SEMILABEL / step / relpath
    spec = importlib.util.spec_from_file_location(modname, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module
