from __future__ import annotations

from typing import Literal

from torch_runtime import select_device_str as _select_device_str
from torch_runtime import select_torch_device as _select_torch_device


def select_device_str(preference: str, *, torch=None) -> Literal["cpu", "mps", "cuda"]:
    return _select_device_str(preference)


def select_torch_device(preference: str, *, torch=None):
    return _select_torch_device(preference)
