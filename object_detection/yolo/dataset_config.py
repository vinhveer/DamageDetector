from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class YoloDatasetConfig:
    yaml_path: Path
    dataset_root: Path
    names: list[str]
    nc: int
    raw: dict[str, Any]


def _normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, dict):
        items = sorted(((int(key), str(value)) for key, value in raw_names.items()), key=lambda item: item[0])
        return [value for _, value in items]
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    raise ValueError("YOLO data config must define 'names' as a list or mapping.")


def _resolve_root(yaml_path: Path, data: dict[str, Any]) -> Path:
    root_value = str(data.get("path") or "").strip()
    if not root_value:
        return yaml_path.parent.resolve()
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def load_yolo_dataset_config(path: str | Path) -> YoloDatasetConfig:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"YOLO data config not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YOLO data config: {yaml_path}")

    names = _normalize_names(data.get("names"))
    nc = int(data.get("nc") or len(names))
    if nc != len(names):
        raise ValueError(f"Dataset nc={nc} does not match names={len(names)} in {yaml_path}")
    return YoloDatasetConfig(
        yaml_path=yaml_path,
        dataset_root=_resolve_root(yaml_path, data),
        names=names,
        nc=nc,
        raw=data,
    )


def _normalize_profile(profile: str) -> str:
    value = str(profile or "balanced").strip().lower()
    if value in {"default", "medium"}:
        return "balanced"
    if value == "strong":
        return "aggressive"
    return value


def build_yolo_augmentation_overrides(profile: str) -> dict[str, float]:
    normalized = _normalize_profile(profile)
    if normalized == "light":
        return {
            "degrees": 5.0,
            "translate": 0.03,
            "scale": 0.08,
            "shear": 1.0,
            "perspective": 0.0,
            "fliplr": 0.5,
            "flipud": 0.1,
            "mosaic": 0.15,
            "mixup": 0.0,
            "hsv_h": 0.01,
            "hsv_s": 0.3,
            "hsv_v": 0.2,
        }
    if normalized == "aggressive":
        return {
            "degrees": 20.0,
            "translate": 0.1,
            "scale": 0.35,
            "shear": 4.0,
            "perspective": 0.0005,
            "fliplr": 0.5,
            "flipud": 0.5,
            "mosaic": 0.9,
            "mixup": 0.15,
            "hsv_h": 0.02,
            "hsv_s": 0.6,
            "hsv_v": 0.45,
        }
    return {
        "degrees": 10.0,
        "translate": 0.06,
        "scale": 0.2,
        "shear": 2.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.3,
        "mosaic": 0.5,
        "mixup": 0.05,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
    }
