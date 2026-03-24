from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal


@lru_cache(maxsize=1)
def get_torch():
    import torch

    return torch


@lru_cache(maxsize=1)
def get_torch_nn_functional():
    import torch.nn.functional as functional

    return functional


@lru_cache(maxsize=1)
def get_torch_nn():
    import torch.nn as nn

    return nn


@lru_cache(maxsize=1)
def get_torch_optim():
    import torch.optim as optim

    return optim


@lru_cache(maxsize=1)
def get_torch_distributed():
    import torch.distributed as dist

    return dist


@lru_cache(maxsize=1)
def get_torch_cudnn():
    import torch.backends.cudnn as cudnn

    return cudnn


@lru_cache(maxsize=1)
def get_torch_utils_data():
    import torch.utils.data as utils_data

    return utils_data


@lru_cache(maxsize=1)
def get_torch_utils_data_distributed():
    import torch.utils.data.distributed as utils_data_distributed

    return utils_data_distributed


@lru_cache(maxsize=1)
def get_torch_utils_data_dataloader():
    import torch.utils.data.dataloader as utils_data_dataloader

    return utils_data_dataloader


@lru_cache(maxsize=1)
def get_torch_nn_parameter():
    import torch.nn.parameter as parameter

    return parameter


class _LazyProxy:
    def __init__(self, loader) -> None:
        self._loader = loader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader(), name)

    def __call__(self, *args, **kwargs):
        return self._loader()(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(self._loader())


def has_mps() -> bool:
    torch = get_torch()
    try:
        return bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def has_cuda() -> bool:
    torch = get_torch()
    try:
        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        return False


def select_device_str(preference: str) -> Literal["cpu", "mps", "cuda"]:
    """
    Select torch device string with priority: cuda -> mps -> cpu.

    - preference can be: auto|cpu|mps|cuda
    - explicit requests fall back sensibly if the requested backend is unavailable
    """
    pref = str(preference or "auto").strip().lower()

    if pref == "cpu":
        return "cpu"

    cuda = has_cuda()
    mps = has_mps()

    if pref == "cuda":
        if cuda:
            return "cuda"
        if mps:
            return "mps"
        return "cpu"

    if pref == "mps":
        if mps:
            return "mps"
        if cuda:
            return "cuda"
        return "cpu"

    if pref != "auto":
        raise ValueError("device must be auto/cpu/mps/cuda")

    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


def select_torch_device(preference: str):
    torch = get_torch()
    return torch.device(select_device_str(preference))


def describe_device_fallback(preference: str, resolved: str) -> str | None:
    pref = str(preference or "auto").strip().lower()
    actual = str(resolved or "").strip().lower()
    if pref in {"", "auto", actual}:
        return None
    return f"WARN: requested device={pref}, but using {actual}."


torch = _LazyProxy(get_torch)
nn = _LazyProxy(get_torch_nn)
F = _LazyProxy(get_torch_nn_functional)
optim = _LazyProxy(get_torch_optim)
dist = _LazyProxy(get_torch_distributed)
cudnn = _LazyProxy(get_torch_cudnn)
Tensor = get_torch().Tensor
DataLoader = get_torch_utils_data().DataLoader
Dataset = get_torch_utils_data().Dataset
DistributedSampler = get_torch_utils_data_distributed().DistributedSampler
default_collate = get_torch_utils_data_dataloader().default_collate
Parameter = get_torch_nn_parameter().Parameter


__all__ = [
    "F",
    "Tensor",
    "DataLoader",
    "Dataset",
    "DistributedSampler",
    "cudnn",
    "default_collate",
    "describe_device_fallback",
    "dist",
    "get_torch",
    "get_torch_cudnn",
    "get_torch_distributed",
    "get_torch_nn",
    "get_torch_nn_functional",
    "get_torch_nn_parameter",
    "get_torch_optim",
    "get_torch_utils_data",
    "get_torch_utils_data_dataloader",
    "get_torch_utils_data_distributed",
    "has_cuda",
    "has_mps",
    "nn",
    "optim",
    "Parameter",
    "select_device_str",
    "select_torch_device",
    "torch",
]
