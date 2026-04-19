from __future__ import annotations

import sys

from .shared import MODE_FINETUNE, MODE_NO_FINETUNE


def _print_help() -> int:
    print(
        "Available SAM modes:\n"
        f"  - {MODE_NO_FINETUNE}: python -m segmentation.sam.no_finetune\n"
        f"  - {MODE_FINETUNE}: python -m segmentation.sam.finetune\n"
        "\nTop-level dispatch:\n"
        f"  - python -m segmentation.sam {MODE_NO_FINETUNE} [args...]\n"
        f"  - python -m segmentation.sam {MODE_FINETUNE} [args...]\n"
        "\nCompatibility entry points:\n"
        "  - python -m segmentation.sam.runtime\n"
        "\nFinetune utilities:\n"
        "  - python -m segmentation.sam.finetune.train\n"
        "  - python -m segmentation.sam.finetune.test\n"
        "  - python -m segmentation.sam.finetune.pseudo_label"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _print_help()
    mode = str(argv[0]).strip().lower()
    if mode in {"no_finetune", "runtime", "base", "pure", "sam"}:
        from .no_finetune.cli import main as no_finetune_main

        return int(no_finetune_main(argv[1:]))
    if mode in {"finetune", "with_finetune", "have_finetune", "sam_finetune"}:
        from .finetune.cli import main as finetune_main

        return int(finetune_main(argv[1:]))
    return _print_help()


if __name__ == "__main__":
    raise SystemExit(main())
