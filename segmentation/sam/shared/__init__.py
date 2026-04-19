"""Shared helpers for SAM mode resolution and future runtime/finetune convergence."""

MODE_NO_FINETUNE = "no_finetune"
MODE_FINETUNE = "finetune"

_MODE_ALIASES = {
    "runtime": MODE_NO_FINETUNE,
    "no_finetune": MODE_NO_FINETUNE,
    "base": MODE_NO_FINETUNE,
    "pure": MODE_NO_FINETUNE,
    "sam": MODE_NO_FINETUNE,
    "finetune": MODE_FINETUNE,
    "with_finetune": MODE_FINETUNE,
    "have_finetune": MODE_FINETUNE,
    "sam_finetune": MODE_FINETUNE,
}


def normalize_sam_mode(mode: str | None) -> str:
    value = str(mode or MODE_NO_FINETUNE).strip().lower()
    normalized = _MODE_ALIASES.get(value)
    if normalized is None:
        raise ValueError(
            f"Unsupported SAM mode: {mode!r}. Expected one of: "
            f"{', '.join(sorted(set(_MODE_ALIASES)))}"
        )
    return normalized


def available_sam_modes() -> tuple[str, str]:
    return (MODE_NO_FINETUNE, MODE_FINETUNE)


__all__ = [
    "MODE_NO_FINETUNE",
    "MODE_FINETUNE",
    "normalize_sam_mode",
    "available_sam_modes",
]
