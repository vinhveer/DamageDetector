from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "object_detection").exists() and (parent / "segmentation").exists():
            return parent
    return here.parents[2]


def lab_root() -> Path:
    return repo_root().parent


def default_yolo_model() -> Path:
    return (
        lab_root()
        / "model_with_inference"
        / "semi_labeling_training"
        / "myrun_yolo26x_img768_b16_100ep"
        / "weights"
        / "best.pt"
    )


def default_stabledino_checkpoint() -> Path:
    return (
        lab_root()
        / "model_with_inference"
        / "semi_labeling_training"
        / "myrun_stabledino_r50_img768_b16_28600it"
        / "model_best.pth"
    )


def default_sam_vit_h_checkpoint() -> Path:
    return repo_root() / "models" / "sam" / "sam_vit_h_4b8939.pth"


def default_sam_vit_b_checkpoint() -> Path:
    return repo_root() / "models" / "sam" / "sam_vit_b_01ec64.pth"


def default_unet_model() -> Path:
    return (
        lab_root()
        / "model_with_inference"
        / "crack_segmentation"
        / "unet_v2_cldice_centerline_ema_b16_img512"
        / "best_model.pth"
    )


def default_sam_finetune_delta() -> Path:
    return (
        lab_root()
        / "model_with_inference"
        / "crack_segmentation"
        / "sam_ablation_b2_lora_hq_coarse"
        / "best_model.pth"
    )
