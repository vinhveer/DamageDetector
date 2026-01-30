from pathlib import Path

# Common Defaults
DEFAULT_OVERLAY_OPACITY = 120
DEFAULT_BRUSH_RADIUS = 18

# UNet Defaults
DEFAULT_UNET_MODEL_PATH = str(Path("../DamageDetectorModels/best_model.pth"))
DEFAULT_UNET_OUT_DIR = "results_unet"
DEFAULT_UNET_THRESHOLD = 0.5
DEFAULT_UNET_INPUT_SIZE = 256
DEFAULT_UNET_TILE_BATCH = 4
DEFAULT_UNET_OVERLAP = 0  # 0 means auto (usually input_size // 2)

# SAM + DINO Defaults
DEFAULT_SAM_CHECKPOINT = str(Path("../DamageDetectorModels/sam_vit_h_4b8939.pth"))
DEFAULT_SAM_OUT_DIR = "results_sam_dino"
DEFAULT_GDINO_CHECKPOINT = "IDEA-Research/grounding-dino-base"
DEFAULT_GDINO_CONFIG = "IDEA-Research/grounding-dino-base"
DEFAULT_TEXT_QUERIES = "crack,mold,stain,spall,damage,column"
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.25
DEFAULT_MAX_DETS = 20
DEFAULT_MIN_AREA = 0
DEFAULT_DILATE = 0
