from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "object_detection").exists() and (parent / "inference_api").exists():
            return parent
    return here.parents[3]


LAB_ROOT = _repo_root().parent
RESULTS_ROOT = LAB_ROOT / "infer_results" / "pineline" / "house_cutout"

# ── Step 1: cắt nhà bằng GDINO + SAM ──────────────────────────────────────────
STEP1_DIR = RESULTS_ROOT / "step1_sam_house_crop"
STEP1_DB = STEP1_DIR / "house_crops.sqlite3"
STEP1_WORK_DIR = STEP1_DIR / "work"          # working RGB (tif→png, đã downscale)
STEP1_CUTOUTS_DIR = STEP1_DIR / "cutouts"     # RGBA cutout nhà → input cho step2
STEP1_MASKS_DIR = STEP1_DIR / "masks"         # mask nhị phân
STEP1_OVERLAY_DIR = STEP1_DIR / "overlays"
STEP1_SUMMARY_CSV = STEP1_DIR / "summary.csv"

# ── Step 2: detect damage trên cutout (giống dino_cutout) ─────────────────────
STEP2_DIR = RESULTS_ROOT / "step2_gdino_detect"
STEP2_DB = STEP2_DIR / "detections.sqlite3"
STEP2_RGB_DIR = STEP2_DIR / "rgb"
STEP2_OVERLAY_DIR = STEP2_DIR / "overlays"
STEP2_SUMMARY_CSV = STEP2_DIR / "summary.csv"

# Thư mục ảnh gốc (gồm "NTT - 16m Lan 3.tif")
DEFAULT_INPUT_DIR = LAB_ROOT / "data" / "HinhAnhThucTe"


def repo_root() -> Path:
    return _repo_root()


def default_sam_checkpoint() -> Path:
    """Chọn SAM checkpoint có thật trong models/sam/ (ưu tiên vit_h → vit_l → vit_b).

    CLAUDE.md liệt kê sam_vit_h nhưng môi trường hiện tại chỉ có sam_vit_b. Hàm này
    fallback sang file đang tồn tại để pipeline chạy được ngay; vẫn trả về đường dẫn
    vit_h mặc định nếu chẳng có file nào (để thông báo lỗi rõ ràng ở runner).
    """
    sam_dir = _repo_root() / "models" / "sam"
    for name in (
        "sam_vit_h_4b8939.pth",
        "sam_vit_l_0b3195.pth",
        "sam_vit_b_01ec64.pth",
    ):
        candidate = sam_dir / name
        if candidate.exists():
            return candidate
    return sam_dir / "sam_vit_h_4b8939.pth"


def ensure_step1_dirs() -> None:
    for d in (
        STEP1_DIR, STEP1_WORK_DIR, STEP1_CUTOUTS_DIR,
        STEP1_MASKS_DIR, STEP1_OVERLAY_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def ensure_step2_dirs() -> None:
    for d in (STEP2_DIR, STEP2_RGB_DIR, STEP2_OVERLAY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def ensure_dirs() -> None:
    ensure_step1_dirs()
    ensure_step2_dirs()
