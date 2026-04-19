from __future__ import annotations


def main() -> int:
    print(
        "Available GroundingDINO tools:\n"
        "  - python -m object_detection.grounding_dino.image\n"
        "  - python -m object_detection.grounding_dino.folder"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

