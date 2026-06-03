from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "object_detection").exists() and (parent / "inference_api").exists():
            return parent
    return here.parents[3]


LAB_ROOT = _repo_root().parent
RESULTS_ROOT = LAB_ROOT / "infer_results" / "pineline" / "dino_cutout"

STEP1_DIR = RESULTS_ROOT / "step1_gdino_detect"
STEP1_DB = STEP1_DIR / "detections.sqlite3"
STEP1_RGB_DIR = STEP1_DIR / "rgb"          # ảnh RGBA đã convert sang RGB
STEP1_OVERLAY_DIR = STEP1_DIR / "overlays"
STEP1_SUMMARY_CSV = STEP1_DIR / "summary.csv"

DEFAULT_INPUT_DIR = (
    LAB_ROOT / "model_with_inference" / "pineline_detect_damage" / "cutouts"
)


def ensure_dirs() -> None:
    for d in (STEP1_DIR, STEP1_RGB_DIR, STEP1_OVERLAY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def repo_root() -> Path:
    return _repo_root()
