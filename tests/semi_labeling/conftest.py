"""Shared fixtures, strategies, and import setup for semi-labeling property tests.

Pure-core modules under ``object_detection.damage_scan`` and the ``step*`` packages
are imported here without triggering the heavy package ``__init__`` (which pulls in
torch / GroundingDINO), so property tests stay CPU-only and fast.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

from hypothesis import HealthCheck, settings

ROOT = Path(__file__).resolve().parents[2]  # DamageDetector/

# Make `import object_detection...` resolvable and helper modules importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _stub_package(name: str, path: Path) -> None:
    """Register a lightweight namespace package so submodule imports skip the
    real (heavy) package ``__init__`` while keeping relative imports working."""
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = mod


_stub_package("object_detection", ROOT / "object_detection")
_stub_package("object_detection.damage_scan", ROOT / "object_detection" / "damage_scan")

# Property tests run >=100 examples per the design's Testing Strategy.
settings.register_profile(
    "semi_labeling",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("semi_labeling")

