from __future__ import annotations

from importlib import resources
from pathlib import Path


def _load_lazy_config():
    try:
        from detectron2.config import LazyConfig
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "StableDINO requires Detectron2, but it is not installed or cannot be imported. "
            "Install a Detectron2 build matching your Torch/CUDA/Python environment first. "
            "YOLO and damage-scan do not require Detectron2."
        ) from exc
    return LazyConfig


def _load_detrex_resources():
    try:
        return resources.files("detrex.config").joinpath("configs")
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "StableDINO requires detrex, but it is not installed or cannot be imported. "
            "Install detrex separately only in an environment that supports its build. "
            "YOLO and damage-scan do not require detrex."
        ) from exc


def get_detrex_config(config_path: str):
    """Load a detrex config without going through pkg_resources."""
    LazyConfig = _load_lazy_config()
    config_root = _load_detrex_resources()
    cfg_file = Path(str(config_root.joinpath(config_path)))
    if not cfg_file.exists():
        raise RuntimeError(f"{config_path} not available in detrex configs!")
    return LazyConfig.load(str(cfg_file))
