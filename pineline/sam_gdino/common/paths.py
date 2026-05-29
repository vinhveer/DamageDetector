from __future__ import annotations

from pathlib import Path


LAB_ROOT = Path("/Users/nguyenquangvinh/Desktop/Lab")
RESULTS_ROOT = LAB_ROOT / "infer_results" / "pineline" / "sam_gdino"

STEP1_DIR = RESULTS_ROOT / "step1_gdino_bridge"
STEP2_DIR = RESULTS_ROOT / "step2_sam_bridge_crop"
STEP3_DIR = RESULTS_ROOT / "step3_gdino_damage"
STEP4_DIR = RESULTS_ROOT / "step4_openclip_semantic"
STEP5_DIR = RESULTS_ROOT / "step5_embedding"
STEP6_DIR = RESULTS_ROOT / "step6_route_segment"

STEP1_DB = STEP1_DIR / "bridge_detections.sqlite3"
STEP1_OVERLAY_DIR = STEP1_DIR / "overlays"
STEP1_SUMMARY_CSV = STEP1_DIR / "summary.csv"
STEP2_DB = STEP2_DIR / "bridge_crops.sqlite3"
STEP2_CROPS_DIR = STEP2_DIR / "images"
STEP2_MASKS_DIR = STEP2_DIR / "masks"
STEP2_OVERLAY_DIR = STEP2_DIR / "overlays"
STEP2_SUMMARY_CSV = STEP2_DIR / "summary.csv"

DEFAULT_SAM_CHECKPOINT = LAB_ROOT / "DamageDetector" / "models" / "sam" / "sam_vit_h_4b8939.pth"
STEP3_DB = STEP3_DIR / "damage_detections.sqlite3"
STEP3_RGB_DIR = STEP3_DIR / "rgb"
STEP3_OVERLAY_DIR = STEP3_DIR / "overlays"
STEP3_SUMMARY_CSV = STEP3_DIR / "summary.csv"
STEP4_DB = STEP4_DIR / "semantic_labels.sqlite3"
STEP4_CROPS_DIR = STEP4_DIR / "crops"
STEP4_SUMMARY_CSV = STEP4_DIR / "summary.csv"
STEP5_DB = STEP5_DIR / "dedup.sqlite3"
STEP5_SUMMARY_CSV = STEP5_DIR / "summary.csv"
STEP6_DB = STEP6_DIR / "masks.sqlite3"
STEP6_MASKS_DIR = STEP6_DIR / "masks"
STEP6_OVERLAY_DIR = STEP6_DIR / "overlays"
STEP6_SUMMARY_CSV = STEP6_DIR / "summary.csv"

DEFAULT_SAM_LORA_BASE_CHECKPOINT = LAB_ROOT / "training_runs" / "v2" / "sam-finetune-lora-hq" / "sam_vit_b_01ec64.pth"
DEFAULT_SAM_LORA_DELTA = (
    LAB_ROOT / "training_runs" / "v2" / "sam-finetune-lora-hq" /
    "outputs_sam_ablation_b2_lora_hq_refine_ddp" /
    "generic_768_pretrain_vit_b_30k_epo16_bs2_gbs4_lr0.0002_s3407_type_lora_stage_refine_r4_run_b2_lora_hq_refine_r4" /
    "best_model.pth"
)


def ensure_dirs() -> None:
    for d in (
        STEP1_DIR, STEP2_DIR, STEP2_CROPS_DIR, STEP2_MASKS_DIR, STEP2_OVERLAY_DIR,
        STEP3_DIR, STEP4_DIR, STEP5_DIR,
        STEP6_DIR, STEP6_MASKS_DIR, STEP6_OVERLAY_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "object_detection").exists() and (parent / "inference_api").exists():
            return parent
    return here.parents[3]
