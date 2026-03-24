from __future__ import annotations

from typing import Any

from inference_api.contracts import InferenceRequest
from inference_api.prediction_models import (
    DETECTION_DINO,
    SEGMENTATION_SAM_LORA,
    SEGMENTATION_UNET,
    TASK_GROUP_CRACK_ONLY,
    TASK_GROUP_MORE_DAMAGE,
    PredictionConfig,
)
from inference_api.workflow_resolver import ResolvedWorkflow, resolve_workflow


def build_prediction_request(
    config: PredictionConfig,
    settings: dict[str, Any],
    *,
    image_path: str | None = None,
    image_paths: list[str] | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
    output_dir: str | None = None,
) -> InferenceRequest:
    resolved = resolve_workflow(config)
    if resolved.workflow.startswith("sam"):
        _require_sam_settings(settings)
        if resolved.segmentation_model == SEGMENTATION_SAM_LORA:
            _require_sam_lora_settings(settings)
    if resolved.workflow.startswith("unet"):
        _require_unet_settings(settings)
    if resolved.detection_model == DETECTION_DINO:
        missing = _require_dino_settings(settings)
        if missing is not None:
            raise ValueError(missing)
    out_dir = str(output_dir or "")
    params: dict[str, Any] = {}
    if resolved.segmentation_model == SEGMENTATION_UNET:
        params["unet"] = _build_unet_params(settings, output_dir=out_dir, roi_box=roi_box, task_group=resolved.task_group)
    elif resolved.segmentation_model == SEGMENTATION_SAM_LORA:
        params["sam"] = _build_sam_lora_params(settings, output_dir=out_dir, roi_box=roi_box, task_group=resolved.task_group)
    else:
        params["sam"] = _build_sam_params(settings, output_dir=out_dir, roi_box=roi_box, task_group=resolved.task_group)
    if resolved.detection_model == DETECTION_DINO:
        params["dino"] = _build_dino_params(settings, output_dir=out_dir, roi_box=roi_box, task_group=resolved.task_group)
    return InferenceRequest(
        workflow=resolved.workflow,
        image_path=image_path,
        image_paths=list(image_paths) if image_paths is not None else None,
        roi_box=roi_box,
        params=params,
        selection=config.normalized().to_dict(),
        resolved=resolved.to_dict(),
        client_tag="editor_app",
        source="editor",
    )


def build_isolate_request(
    settings: dict[str, Any],
    *,
    image_path: str,
    output_dir: str | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
    target_labels: list[str] | None = None,
    outside_value: int | None = None,
    crop_to_bbox: bool | None = None,
) -> InferenceRequest:
    _require_sam_settings(settings)
    missing = _require_dino_settings(settings)
    if missing is not None:
        raise ValueError(missing)
    out_dir = str(output_dir or "")
    return InferenceRequest(
        workflow="isolate",
        image_path=image_path,
        roi_box=roi_box,
        params={
            "sam": _build_sam_params(settings, output_dir=out_dir, roi_box=roi_box, task_group=TASK_GROUP_MORE_DAMAGE),
            "dino": _build_dino_params(settings, output_dir=out_dir, roi_box=roi_box, task_group=TASK_GROUP_MORE_DAMAGE),
            "target_labels": [label.strip() for label in (target_labels or []) if str(label).strip()],
            "outside_value": int(outside_value or 0),
            "crop_to_bbox": bool(crop_to_bbox or False),
        },
        selection={
            "task_group": TASK_GROUP_MORE_DAMAGE,
            "segmentation_model": "sam",
            "detection_model": "dino",
            "scope": "current",
            "task_group_label": "Predict for More Damage",
            "segmentation_model_label": "SAM",
            "detection_model_label": "DINO",
            "scope_label": "Current image",
        },
        resolved={
            "workflow": "isolate",
            "task_group": TASK_GROUP_MORE_DAMAGE,
            "segmentation_model": "sam",
            "detection_model": "dino",
        },
        client_tag="editor_app",
        source="editor",
    )


def _split_queries(raw: Any) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _int_value(settings: dict[str, Any], key: str, default: int) -> int:
    value = settings.get(key)
    if value in (None, ""):
        return int(default)
    return int(value)


def _float_value(settings: dict[str, Any], key: str, default: float) -> float:
    value = settings.get(key)
    if value in (None, ""):
        return float(default)
    return float(value)


def _require_sam_settings(settings: dict[str, Any]) -> None:
    sam_ckpt = str(settings.get("sam_checkpoint") or "").strip()
    if not sam_ckpt:
        raise ValueError("SAM checkpoint is required.")


def _require_sam_lora_settings(settings: dict[str, Any]) -> None:
    delta_ckpt = str(settings.get("sam_lora_checkpoint") or "").strip()
    if not delta_ckpt:
        raise ValueError("LoRA checkpoint is required.")


def _require_dino_settings(settings: dict[str, Any]) -> str | None:
    gdino_ckpt = str(settings.get("dino_checkpoint") or "").strip()
    if not gdino_ckpt:
        return "DINO checkpoint is required."
    config_id = str(settings.get("dino_config_id") or "").strip()
    if not config_id:
        return "DINO config ID is required."
    return None


def _require_unet_settings(settings: dict[str, Any]) -> None:
    model_path = str(settings.get("unet_model") or "").strip()
    if not model_path:
        raise ValueError("UNet model is required.")


def _build_sam_common(
    settings: dict[str, Any],
    *,
    output_dir: str,
    roi_box: tuple[int, int, int, int] | None,
    task_group: str,
) -> dict[str, Any]:
    return {
        "sam_checkpoint": str(settings.get("sam_checkpoint") or "").strip(),
        "sam_model_type": str(settings.get("sam_model_type") or "auto"),
        "invert_mask": bool(settings.get("invert_mask") or False),
        "sam_min_component_area": _int_value(settings, "min_area", 0),
        "sam_dilate_iters": _int_value(settings, "dilate", 0),
        "seed": _int_value(settings, "sam_seed", 1337),
        "overlay_alpha": _float_value(settings, "sam_overlay_alpha", 0.45),
        "device": str(settings.get("device") or "auto"),
        "output_dir": output_dir or "results_sam",
        "roi_box": roi_box,
        "task_group": str(task_group or TASK_GROUP_CRACK_ONLY),
        "more_damage_max_masks": _int_value(settings, "more_damage_max_masks", 8),
        "sam_auto_profile": str(settings.get("sam_auto_profile") or "auto"),
        "sam_points_per_side": _int_value(settings, "sam_points_per_side", -1),
        "sam_points_per_batch": _int_value(settings, "sam_points_per_batch", -1),
        "sam_pred_iou_thresh": float(settings.get("sam_pred_iou_thresh") if settings.get("sam_pred_iou_thresh") not in (None, "") else -1.0),
        "sam_stability_score_thresh": float(settings.get("sam_stability_score_thresh") if settings.get("sam_stability_score_thresh") not in (None, "") else -1.0),
        "sam_stability_score_offset": float(settings.get("sam_stability_score_offset") if settings.get("sam_stability_score_offset") not in (None, "") else -1.0),
        "sam_box_nms_thresh": float(settings.get("sam_box_nms_thresh") if settings.get("sam_box_nms_thresh") not in (None, "") else -1.0),
        "sam_crop_n_layers": _int_value(settings, "sam_crop_n_layers", -1),
        "sam_crop_overlap_ratio": float(settings.get("sam_crop_overlap_ratio") if settings.get("sam_crop_overlap_ratio") not in (None, "") else -1.0),
        "sam_crop_nms_thresh": float(settings.get("sam_crop_nms_thresh") if settings.get("sam_crop_nms_thresh") not in (None, "") else -1.0),
        "sam_crop_n_points_downscale_factor": _int_value(settings, "sam_crop_n_points_downscale_factor", -1),
        "sam_min_mask_region_area": _int_value(settings, "sam_min_mask_region_area", -1),
    }


def _build_sam_params(
    settings: dict[str, Any],
    *,
    output_dir: str,
    roi_box: tuple[int, int, int, int] | None,
    task_group: str,
) -> dict[str, Any]:
    return _build_sam_common(settings, output_dir=output_dir, roi_box=roi_box, task_group=task_group)


def _build_sam_lora_params(
    settings: dict[str, Any],
    *,
    output_dir: str,
    roi_box: tuple[int, int, int, int] | None,
    task_group: str,
) -> dict[str, Any]:
    params = _build_sam_common(settings, output_dir=output_dir, roi_box=roi_box, task_group=task_group)
    params.update(
        {
            "delta_type": "lora",
            "delta_checkpoint": str(settings.get("sam_lora_checkpoint") or "").strip(),
            "middle_dim": _int_value(settings, "sam_lora_middle_dim", 32),
            "scaling_factor": _float_value(settings, "sam_lora_scaling_factor", 0.2),
            "rank": _int_value(settings, "sam_lora_rank", 4),
        }
    )
    return params


def _build_unet_params(
    settings: dict[str, Any],
    *,
    output_dir: str,
    roi_box: tuple[int, int, int, int] | None,
    task_group: str,
) -> dict[str, Any]:
    return {
        "model_path": str(settings.get("unet_model") or "").strip(),
        "output_dir": output_dir or "results_unet",
        "threshold": _float_value(settings, "unet_threshold", 0.5),
        "apply_postprocessing": bool(settings.get("unet_post") or False),
        "mode": str(settings.get("unet_mode") or "tile"),
        "input_size": _int_value(settings, "unet_input_size", 256),
        "tile_overlap": _int_value(settings, "unet_overlap", 0),
        "tile_batch_size": _int_value(settings, "unet_tile_batch", 4),
        "device": str(settings.get("device") or "auto"),
        "roi_box": roi_box,
        "task_group": str(task_group or TASK_GROUP_CRACK_ONLY),
    }


def _queries_for_task_group(task_group: str, settings: dict[str, Any]) -> list[str]:
    if task_group == TASK_GROUP_MORE_DAMAGE:
        return _split_queries(settings.get("more_damage_text_queries"))
    return _split_queries(settings.get("crack_text_queries"))


def _build_dino_params(
    settings: dict[str, Any],
    *,
    output_dir: str,
    roi_box: tuple[int, int, int, int] | None,
    task_group: str,
) -> dict[str, Any]:
    return {
        "gdino_checkpoint": str(settings.get("dino_checkpoint") or "").strip(),
        "gdino_config_id": str(settings.get("dino_config_id") or "").strip(),
        "text_queries": _queries_for_task_group(task_group, settings),
        "box_threshold": _float_value(settings, "box_threshold", 0.25),
        "text_threshold": _float_value(settings, "text_threshold", 0.25),
        "max_dets": _int_value(settings, "max_dets", 20),
        "device": str(settings.get("device") or "auto"),
        "output_dir": output_dir or "results_dino",
        "roi_box": roi_box,
        "nms_iou_threshold": _float_value(settings, "dino_nms_iou_threshold", 0.5),
        "parent_contain_threshold": _float_value(settings, "dino_parent_contain_threshold", 0.7),
        "recursive_min_box_px": _int_value(settings, "dino_recursive_min_box_px", 48),
        "recursive_max_depth": _int_value(settings, "dino_recursive_max_depth", 3),
    }
