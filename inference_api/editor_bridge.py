from __future__ import annotations

from typing import Any

from inference_api.request_builder import build_isolate_request, build_prediction_request
from inference_api.prediction_models import (
    DETECTION_DINO,
    DETECTION_NONE,
    SEGMENTATION_SAM,
    SEGMENTATION_SAM_LORA,
    SEGMENTATION_UNET,
    TASK_GROUP_CRACK_ONLY,
    PredictionConfig,
)


_LEGACY_MODE_TO_CONFIG: dict[str, PredictionConfig] = {
    "sam_dino": PredictionConfig(
        task_group=TASK_GROUP_CRACK_ONLY,
        segmentation_model=SEGMENTATION_SAM,
        detection_model=DETECTION_DINO,
    ),
    "sam_dino_ft": PredictionConfig(
        task_group=TASK_GROUP_CRACK_ONLY,
        segmentation_model=SEGMENTATION_SAM_LORA,
        detection_model=DETECTION_DINO,
    ),
    "sam_only": PredictionConfig(
        task_group=TASK_GROUP_CRACK_ONLY,
        segmentation_model=SEGMENTATION_SAM,
        detection_model=DETECTION_NONE,
    ),
    "sam_only_ft": PredictionConfig(
        task_group=TASK_GROUP_CRACK_ONLY,
        segmentation_model=SEGMENTATION_SAM_LORA,
        detection_model=DETECTION_NONE,
    ),
    "unet_only": PredictionConfig(
        task_group=TASK_GROUP_CRACK_ONLY,
        segmentation_model=SEGMENTATION_UNET,
        detection_model=DETECTION_NONE,
    ),
    "unet_dino": PredictionConfig(
        task_group=TASK_GROUP_CRACK_ONLY,
        segmentation_model=SEGMENTATION_UNET,
        detection_model=DETECTION_DINO,
    ),
}


def normalize_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized == "unet":
        return "unet_dino"
    return normalized


def build_editor_request(
    mode: str,
    settings: dict[str, Any],
    *,
    image_path: str | None = None,
    image_paths: list[str] | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
    output_dir: str | None = None,
    target_labels: list[str] | None = None,
    outside_value: int | None = None,
    crop_to_bbox: bool | None = None,
    max_depth: int | None = None,
    min_box_px: int | None = None,
):
    normalized = normalize_mode(mode)
    if normalized == "isolate":
        return build_isolate_request(
            settings,
            image_path=str(image_path or ""),
            output_dir=output_dir,
            roi_box=roi_box,
            target_labels=target_labels,
            outside_value=outside_value,
            crop_to_bbox=crop_to_bbox,
        )
    if normalized == "sam_tiled":
        raise ValueError("Legacy mode 'sam_tiled' is no longer exposed in editor_app.")
    config = _LEGACY_MODE_TO_CONFIG.get(normalized)
    if config is None:
        raise ValueError(f"Unknown predict mode: {mode}")
    scope = "folder" if image_paths else "current"
    config = PredictionConfig(
        task_group=config.task_group,
        segmentation_model=config.segmentation_model,
        detection_model=config.detection_model,
        scope=scope,
    )
    return build_prediction_request(
        config,
        settings,
        image_path=image_path,
        image_paths=image_paths,
        roi_box=roi_box,
        output_dir=output_dir,
    )
