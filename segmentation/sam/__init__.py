"""SAM namespace exposing no-finetune and finetune modes."""

from .shared import MODE_FINETUNE, MODE_NO_FINETUNE, available_sam_modes, normalize_sam_mode

__all__ = [
    "MODE_NO_FINETUNE",
    "MODE_FINETUNE",
    "available_sam_modes",
    "normalize_sam_mode",
    "get_sam_service",
    "get_sam_runner",
    "get_sam_params",
]


def get_sam_service(mode: str = MODE_NO_FINETUNE):
    resolved = normalize_sam_mode(mode)
    if resolved == MODE_FINETUNE:
        from .finetune import get_sam_finetune_service

        return get_sam_finetune_service()
    from .runtime import get_sam_service as get_runtime_sam_service

    return get_runtime_sam_service()


def get_sam_runner(mode: str = MODE_NO_FINETUNE):
    resolved = normalize_sam_mode(mode)
    if resolved == MODE_FINETUNE:
        from .finetune import SamFinetuneRunner

        return SamFinetuneRunner
    from .runtime import SamRunner

    return SamRunner


def get_sam_params(mode: str = MODE_NO_FINETUNE):
    resolved = normalize_sam_mode(mode)
    if resolved == MODE_FINETUNE:
        from .finetune import SamFinetuneParams

        return SamFinetuneParams
    from .runtime import SamParams

    return SamParams
