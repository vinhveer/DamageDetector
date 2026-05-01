from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def lab_root() -> Path:
    return repo_root().parent


def default_source_db() -> Path:
    return lab_root() / "infer_results" / "semi-labeling" / "2_sematic" / "damage_scan.sqlite3"


def default_feature_db() -> Path:
    giant = default_source_db().parent / "step4_feature_grouping_giant_agglo045_full_restart" / "feature_groups.sqlite3"
    if giant.is_file():
        return giant
    return default_source_db().parent / "step4_feature_grouping" / "feature_groups.sqlite3"


def default_image_root() -> Path:
    return lab_root() / "HinhAnh"
