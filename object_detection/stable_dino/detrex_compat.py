from __future__ import annotations

from importlib import resources
from pathlib import Path
import pkgutil
from importlib.machinery import FileFinder


def _ensure_python312_pkg_resources_compat() -> None:
    if not hasattr(pkgutil, "ImpImporter"):
        class ImpImporter:  # pragma: no cover - compatibility shim
            pass

        pkgutil.ImpImporter = ImpImporter
    if not hasattr(pkgutil, "ImpLoader"):
        class ImpLoader:  # pragma: no cover - compatibility shim
            pass

        pkgutil.ImpLoader = ImpLoader
    if not hasattr(FileFinder, "find_module"):
        def find_module(self, fullname: str, path: object | None = None):
            spec = self.find_spec(fullname)
            return spec.loader if spec is not None else None

        FileFinder.find_module = find_module


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
    _ensure_python312_pkg_resources_compat()
    LazyConfig = _load_lazy_config()
    config_root = _load_detrex_resources()
    cfg_file = Path(str(config_root.joinpath(config_path)))
    if not cfg_file.exists():
        raise RuntimeError(f"{config_path} not available in detrex configs!")
    return LazyConfig.load(str(cfg_file))
