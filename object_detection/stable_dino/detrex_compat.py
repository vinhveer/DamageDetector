from __future__ import annotations

from importlib import resources
from pathlib import Path

from detectron2.config import LazyConfig


def get_detrex_config(config_path: str):
    """Load a detrex config without going through pkg_resources."""
    config_root = resources.files("detrex.config").joinpath("configs")
    cfg_file = Path(str(config_root.joinpath(config_path)))
    if not cfg_file.exists():
        raise RuntimeError(f"{config_path} not available in detrex configs!")
    return LazyConfig.load(str(cfg_file))
