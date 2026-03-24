from __future__ import annotations

from inference_api.prediction_models import (
    DETECTION_DINO,
    DETECTION_NONE,
    SEGMENTATION_SAM,
    SEGMENTATION_SAM_LORA,
    SEGMENTATION_UNET,
    SCOPE_CURRENT,
    SCOPE_FOLDER,
    TASK_GROUP_CRACK_ONLY,
    TASK_GROUP_MORE_DAMAGE,
    PredictionConfig,
    ResolvedWorkflow,
)


_CRACK_ONLY_WORKFLOWS: dict[tuple[str, str], str] = {
    (SEGMENTATION_SAM, DETECTION_NONE): "sam_only",
    (SEGMENTATION_SAM, DETECTION_DINO): "sam_dino",
    (SEGMENTATION_SAM_LORA, DETECTION_NONE): "sam_only_ft",
    (SEGMENTATION_SAM_LORA, DETECTION_DINO): "sam_dino_ft",
    (SEGMENTATION_UNET, DETECTION_NONE): "unet_only",
    (SEGMENTATION_UNET, DETECTION_DINO): "unet_dino",
}

_MORE_DAMAGE_WORKFLOWS: dict[tuple[str, str], str] = {
    (SEGMENTATION_SAM, DETECTION_NONE): "sam_only",
    (SEGMENTATION_SAM, DETECTION_DINO): "sam_dino",
}


def validate_prediction_config(config: PredictionConfig) -> PredictionConfig:
    normalized = config.normalized()
    if normalized.task_group not in {TASK_GROUP_CRACK_ONLY, TASK_GROUP_MORE_DAMAGE}:
        raise ValueError(f"Unsupported task group: {config.task_group}")
    if normalized.segmentation_model not in {SEGMENTATION_SAM, SEGMENTATION_SAM_LORA, SEGMENTATION_UNET}:
        raise ValueError(f"Unsupported segmentation model: {config.segmentation_model}")
    if normalized.detection_model not in {DETECTION_DINO, DETECTION_NONE}:
        raise ValueError(f"Unsupported detection model: {config.detection_model}")
    if normalized.scope not in {SCOPE_CURRENT, SCOPE_FOLDER}:
        raise ValueError(f"Unsupported scope: {config.scope}")
    if normalized.task_group == TASK_GROUP_MORE_DAMAGE:
        if normalized.segmentation_model != SEGMENTATION_SAM:
            raise ValueError("Predict for More Damage currently supports SAM only.")
        if normalized.detection_model not in {DETECTION_DINO, DETECTION_NONE}:
            raise ValueError("Predict for More Damage supports DINO or No Detect only.")
    return normalized


def resolve_workflow(config: PredictionConfig) -> ResolvedWorkflow:
    normalized = validate_prediction_config(config)
    if normalized.task_group == TASK_GROUP_MORE_DAMAGE:
        workflow = _MORE_DAMAGE_WORKFLOWS.get((normalized.segmentation_model, normalized.detection_model))
    else:
        workflow = _CRACK_ONLY_WORKFLOWS.get((normalized.segmentation_model, normalized.detection_model))
    if not workflow:
        raise ValueError(
            "Unsupported prediction combination: "
            f"{normalized.task_group} / {normalized.segmentation_model} / {normalized.detection_model}"
        )
    return ResolvedWorkflow(
        workflow=workflow,
        task_group=normalized.task_group,
        segmentation_model=normalized.segmentation_model,
        detection_model=normalized.detection_model,
    )
