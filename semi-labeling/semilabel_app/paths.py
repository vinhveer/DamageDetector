from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def lab_root() -> Path:
    return repo_root().parent


def semi_labeling_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def default_resemi_db() -> Path:
    return lab_root() / "model_with_inference" / "semi_labeling" / "pipeline.sqlite3"


def default_image_root() -> Path:
    return lab_root() / "data" / "HinhAnh"


def default_export_dir() -> Path:
    return lab_root() / "model_with_inference" / "semi_labeling" / "exports"
