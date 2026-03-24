from __future__ import annotations

from dataclasses import dataclass


TASK_GROUP_CRACK_ONLY = "crack_only"
TASK_GROUP_MORE_DAMAGE = "more_damage"

SEGMENTATION_SAM = "sam"
SEGMENTATION_SAM_LORA = "sam_lora"
SEGMENTATION_UNET = "unet"

DETECTION_DINO = "dino"
DETECTION_NONE = "none"

SCOPE_CURRENT = "current"
SCOPE_FOLDER = "folder"

TASK_GROUP_LABELS: dict[str, str] = {
    TASK_GROUP_CRACK_ONLY: "Predict for Crack Only",
    TASK_GROUP_MORE_DAMAGE: "Predict for More Damage",
}

SEGMENTATION_LABELS: dict[str, str] = {
    SEGMENTATION_SAM: "SAM",
    SEGMENTATION_SAM_LORA: "SAM Finetune with LoRA",
    SEGMENTATION_UNET: "UNet",
}

DETECTION_LABELS: dict[str, str] = {
    DETECTION_DINO: "DINO",
    DETECTION_NONE: "No Detect",
}

SCOPE_LABELS: dict[str, str] = {
    SCOPE_CURRENT: "Current image",
    SCOPE_FOLDER: "Whole folder",
}


@dataclass(frozen=True)
class PredictionConfig:
    task_group: str
    segmentation_model: str
    detection_model: str
    scope: str = SCOPE_CURRENT

    def normalized(self) -> "PredictionConfig":
        return PredictionConfig(
            task_group=str(self.task_group or "").strip().lower(),
            segmentation_model=str(self.segmentation_model or "").strip().lower(),
            detection_model=str(self.detection_model or "").strip().lower(),
            scope=str(self.scope or SCOPE_CURRENT).strip().lower(),
        )

    def to_dict(self) -> dict[str, str]:
        cfg = self.normalized()
        return {
            "task_group": cfg.task_group,
            "segmentation_model": cfg.segmentation_model,
            "detection_model": cfg.detection_model,
            "scope": cfg.scope,
            "task_group_label": TASK_GROUP_LABELS.get(cfg.task_group, cfg.task_group),
            "segmentation_model_label": SEGMENTATION_LABELS.get(cfg.segmentation_model, cfg.segmentation_model),
            "detection_model_label": DETECTION_LABELS.get(cfg.detection_model, cfg.detection_model),
            "scope_label": SCOPE_LABELS.get(cfg.scope, cfg.scope),
        }


@dataclass(frozen=True)
class ResolvedWorkflow:
    workflow: str
    task_group: str
    segmentation_model: str
    detection_model: str

    def to_dict(self) -> dict[str, str]:
        return {
            "workflow": self.workflow,
            "task_group": self.task_group,
            "segmentation_model": self.segmentation_model,
            "detection_model": self.detection_model,
        }
