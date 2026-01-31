from __future__ import annotations

from typing import Literal


def _has_mps(torch) -> bool:
    try:
        return bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def select_device_str(preference: str, *, torch) -> Literal["cpu", "mps", "cuda"]:
    """
    Select torch device string with priority: cpu -> mps -> cuda.

    - preference can be: auto|cpu|mps|cuda
    - When preference is explicit (mps/cuda), we try that first, then fall back by priority.
    """
    pref = str(preference or "auto").strip().lower()

    if pref == "cpu":
        return "cpu"

    has_mps = _has_mps(torch)
    has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())

    if pref == "mps":
        if has_mps:
            return "mps"
        return "cpu"

    if pref == "cuda":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"

    if pref != "auto":
        raise ValueError("device must be auto/cpu/mps/cuda")

    # auto: priority cuda -> mps -> cpu
    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"


def select_torch_device(preference: str, *, torch):
    return torch.device(select_device_str(preference, torch=torch))
