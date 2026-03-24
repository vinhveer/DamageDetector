from __future__ import annotations

from inference_api.prediction_models import (
    DETECTION_LABELS,
    SEGMENTATION_LABELS,
    TASK_GROUP_CRACK_ONLY,
    TASK_GROUP_LABELS,
    PredictionConfig,
)


DEFAULT_EDITOR_SETTINGS = {
    "sam_checkpoint": "",
    "sam_model_type": "auto",
    "sam_lora_checkpoint": "auto",
    "sam_lora_middle_dim": 32,
    "sam_lora_scaling_factor": 0.2,
    "sam_lora_rank": 4,
    "invert_mask": False,
    "min_area": 0,
    "dilate": 0,
    "sam_seed": 1337,
    "sam_overlay_alpha": 0.45,
    "sam_auto_profile": "auto",
    "sam_points_per_side": -1,
    "sam_points_per_batch": -1,
    "sam_pred_iou_thresh": -1.0,
    "sam_stability_score_thresh": -1.0,
    "sam_stability_score_offset": -1.0,
    "sam_box_nms_thresh": -1.0,
    "sam_crop_n_layers": -1,
    "sam_crop_overlap_ratio": -1.0,
    "sam_crop_nms_thresh": -1.0,
    "sam_crop_n_points_downscale_factor": -1,
    "sam_min_mask_region_area": -1,
    "dino_checkpoint": "IDEA-Research/grounding-dino-base",
    "dino_config_id": "IDEA-Research/grounding-dino-base",
    "box_threshold": 0.25,
    "text_threshold": 0.25,
    "max_dets": 20,
    "dino_nms_iou_threshold": 0.5,
    "dino_parent_contain_threshold": 0.7,
    "dino_recursive_min_box_px": 48,
    "dino_recursive_max_depth": 3,
    "device": "auto",
    "unet_model": "",
    "unet_threshold": 0.5,
    "unet_post": True,
    "unet_mode": "letterbox",
    "unet_input_size": 512,
    "unet_overlap": 0,
    "unet_tile_batch": 4,
    "crack_text_queries": "crack",
    "more_damage_text_queries": "mold,stain,spall,damage,column",
    "more_damage_max_masks": 8,
    "isolate_labels": "mold,stain,spall,damage,column",
    "isolate_crop": False,
    "isolate_outside_white": False,
}


def migrate_editor_settings(payload: dict | None) -> dict:
    settings = dict(payload or {})
    legacy_delta = str(settings.get("delta_checkpoint") or "").strip()
    if "sam_lora_checkpoint" not in settings:
        settings["sam_lora_checkpoint"] = legacy_delta or "auto"
    if "sam_lora_middle_dim" not in settings:
        settings["sam_lora_middle_dim"] = int(settings.get("middle_dim") or 32)
    if "sam_lora_scaling_factor" not in settings:
        settings["sam_lora_scaling_factor"] = float(settings.get("scaling_factor") or 0.2)
    if "sam_lora_rank" not in settings:
        settings["sam_lora_rank"] = int(settings.get("rank") or 4)
    legacy_queries = str(settings.get("text_queries") or "").strip()
    if "crack_text_queries" not in settings:
        settings["crack_text_queries"] = "crack"
    if "more_damage_text_queries" not in settings:
        settings["more_damage_text_queries"] = legacy_queries or "mold,stain,spall,damage,column"
    if "isolate_labels" not in settings:
        settings["isolate_labels"] = "mold,stain,spall,damage,column"
    for legacy_key in ("delta_type", "delta_checkpoint", "middle_dim", "scaling_factor", "rank", "text_queries"):
        settings.pop(legacy_key, None)
    return settings


def prediction_summary(config: PredictionConfig) -> str:
    normalized = config.normalized()
    return (
        f"{TASK_GROUP_LABELS.get(normalized.task_group, normalized.task_group)} | "
        f"{SEGMENTATION_LABELS.get(normalized.segmentation_model, normalized.segmentation_model)} | "
        f"{DETECTION_LABELS.get(normalized.detection_model, normalized.detection_model)}"
    )


def default_prediction_config() -> PredictionConfig:
    return PredictionConfig(
        task_group=TASK_GROUP_CRACK_ONLY,
        segmentation_model="sam",
        detection_model="dino",
        scope="current",
    )
