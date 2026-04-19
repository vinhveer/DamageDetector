"""Canonical no-finetune SAM mode."""

__all__ = ["SamParams", "SamRunner", "get_sam_service"]


def __getattr__(name: str):
    if name == "get_sam_service":
        from .client import get_sam_service

        return get_sam_service
    if name in {"SamParams", "SamRunner"}:
        from .engine import SamParams, SamRunner

        return {"SamParams": SamParams, "SamRunner": SamRunner}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
