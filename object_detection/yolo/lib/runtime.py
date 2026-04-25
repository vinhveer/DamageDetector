from __future__ import annotations

from typing import Any

from torch_runtime import describe_device_fallback, select_device_str


def load_yolo_class() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Cannot import ultralytics. Install it first with: pip install ultralytics"
        ) from exc
    return YOLO


def resolve_device(preference: str) -> str:
    resolved = select_device_str(preference)
    fallback = describe_device_fallback(preference, resolved)
    if fallback:
        print(fallback)
    return resolved
