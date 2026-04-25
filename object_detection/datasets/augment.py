from __future__ import annotations


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


def build_stable_dino_augmentation(profile: str, image_size: int) -> dict[str, object]:
    normalized = _normalize_profile(profile)
    if normalized == "light":
        return {
            "image_size": int(image_size),
            "min_scale": 0.8,
            "max_scale": 1.25,
            "random_flip": "horizontal",
        }
    if normalized == "aggressive":
        return {
            "image_size": int(image_size),
            "min_scale": 0.1,
            "max_scale": 2.0,
            "random_flip": "horizontal",
        }
    return {
        "image_size": int(image_size),
        "min_scale": 0.3,
        "max_scale": 1.7,
        "random_flip": "horizontal",
    }
