from __future__ import annotations


def main() -> int:
    print(
        "Available object detection tools:\n"
        "  - python -m object_detection.dino\n"
        "  - python -m object_detection.grounding_dino.image\n"
        "  - python -m object_detection.grounding_dino.folder\n"
        "  - python -m object_detection.yolo"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

