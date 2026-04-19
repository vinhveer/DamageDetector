"""Shared dataset namespace for segmentation pipelines."""

from importlib import import_module

__all__ = ["core", "sam_finetune", "unet"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
