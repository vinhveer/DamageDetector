__all__ = ["SamFinetuneParams", "SamFinetuneRunner", "get_sam_finetune_service"]


def __getattr__(name: str):
    if name == "get_sam_finetune_service":
        from .client import get_sam_finetune_service

        return get_sam_finetune_service
    if name in {"SamFinetuneParams", "SamFinetuneRunner"}:
        from .engine import SamFinetuneParams, SamFinetuneRunner

        return {"SamFinetuneParams": SamFinetuneParams, "SamFinetuneRunner": SamFinetuneRunner}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
