__all__ = ["UnetParams", "UnetRunner", "get_unet_service"]


def __getattr__(name: str):
    if name == "get_unet_service":
        from .client import get_unet_service

        return get_unet_service
    if name in {"UnetParams", "UnetRunner"}:
        from .engine import UnetParams, UnetRunner

        return {"UnetParams": UnetParams, "UnetRunner": UnetRunner}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
