"""Shared path resolution for resemi.

Single source of truth for LAB_ROOT and default artifact locations, so the
step CLIs and tools don't each re-implement lab-root discovery.
"""
from __future__ import annotations

from pathlib import Path


def _resolve_lab_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "DamageDetector").exists() and (candidate / "infer_results").exists():
            return candidate
    # runtime/ -> shared/ -> semi-labeling/ -> DamageDetector/ -> Lab
    return current.parents[4]


LAB_ROOT = _resolve_lab_root()

# Inputs come from the existing pipeline artifacts under infer_results/.
SEMI_RESULTS = LAB_ROOT / "infer_results" / "semi-labeling"

# resemi v2 writes its own consolidated output here (fresh runs).
RESEMI_OUTPUT_DIR = LAB_ROOT / "model_with_inference" / "semi_labeling"


def default_resemi_db() -> Path:
    return RESEMI_OUTPUT_DIR / "resemi.sqlite3"


def default_source_db() -> Path:
    return SEMI_RESULTS / "step2_sematic" / "damage_scan.sqlite3"


def default_dedup_db() -> Path:
    return SEMI_RESULTS / "step4_class_aware_dedup" / "dedup.sqlite3"


def default_image_root() -> Path:
    return LAB_ROOT / "data" / "HinhAnh"
