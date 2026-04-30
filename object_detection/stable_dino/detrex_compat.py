from __future__ import annotations

from importlib import resources
from pathlib import Path


def _load_lazy_config():
    try:
        from detectron2.config import LazyConfig
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "StableDINO requires Detectron2, but it is not installed or cannot be imported. "
            "Install a Detectron2 wheel matching your Torch/CUDA/Python environment first. "
            "Do not install Detectron2 from the default git source on Colab unless you know "
            "the build is supported for that runtime."
        ) from exc
    return LazyConfig


def get_detrex_config(config_path: str):
    """Load a detrex config without going through pkg_resources."""
    LazyConfig = _load_lazy_config()
    config_root = resources.files("detrex.config").joinpath("configs")
    cfg_file = Path(str(config_root.joinpath(config_path)))
    if not cfg_file.exists():
        raise RuntimeError(f"{config_path} not available in detrex configs!")
    return LazyConfig.load(str(cfg_file))
