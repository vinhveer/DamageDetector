from __future__ import annotations


def main() -> int:
    print(
        "Available segmentation tools:\n"
        "  - python -m segmentation.unet\n"
        "  - python -m segmentation.unet.train\n"
        "  - python -m segmentation.sam.runtime\n"
        "  - python -m segmentation.sam.finetune\n"
        "  - python -m segmentation.sam.finetune.train\n"
        "  - python -m segmentation.sam.finetune.test\n"
        "  - python -m segmentation.sam.finetune.pseudo_label"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

