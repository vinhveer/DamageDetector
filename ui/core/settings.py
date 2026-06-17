from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_SETTINGS_DIR = Path.home() / ".damagedetector" / "ui"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"
_LEGACY_FILE = Path(__file__).resolve().parents[2] / ".editor_app.json"

_VERSION = 1
_LAB_DIR = Path(__file__).resolve().parents[3]
_MODEL_ROOT = _LAB_DIR / "model_with_inference"
_DEFAULT_SAM_LORA_DIR = _MODEL_ROOT / "crack_segmentation/sam_lora_hq_coarse_refine/model"
_DEFAULT_STABLEDINO = _MODEL_ROOT / "semi_labeling_training/myrun_stabledino_r50_img768_b16_28600it/model_best.pth"


@dataclass
class UiSettings:
    # --- app identity ---
    app_name: str = "DamageDetector"
    default_width: int = 1280
    default_height: int = 800

    # --- model paths ---
    dino_checkpoint: str = "IDEA-Research/grounding-dino-base"
    dino_config_id: str = "IDEA-Research/grounding-dino-base"
    stabledino_checkpoint: str = str(_DEFAULT_STABLEDINO)
    sam_checkpoint: str = ""
    sam_model_type: str = "auto"
    sam_lora_base_checkpoint: str = str(_DEFAULT_SAM_LORA_DIR / "sam_vit_b_01ec64.pth")
    sam_lora_checkpoint: str = str(_DEFAULT_SAM_LORA_DIR / "coarse_best_model.pth")
    sam_lora_rank: int = 4
    sam_lora_predict_mode: str = "coarse_refine"
    sam_lora_refine_checkpoint: str = str(_DEFAULT_SAM_LORA_DIR / "refine_best_model.pth")
    sam_lora_refine_rank: int = 4
    unet_checkpoint: str = str(_MODEL_ROOT / "crack_segmentation/unet_efficientnet_b4/model/best_model.pth")
    yolo_checkpoint: str = str(_MODEL_ROOT / "crack_object_detection/yolo_26x_img768/model/best.pt")
    yolo_conf: float = 0.10
    yolo_iou: float = 0.45
    yolo_imgsz: int = 768
    yolo_max_dets: int = 50

    # --- thresholds ---
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    max_dets: int = 20
    unet_threshold: float = 0.5
    min_box_px: int = 4
    min_mask_area: int = 0

    # --- device ---
    device: str = "auto"

    # --- UI ---
    theme: str = "system"
    show_grid: bool = False
    default_zoom: str = "fit"
    icon_size: int = 24

    # --- performance ---
    max_jobs_parallel: int = 2
    mask_pixmap_cache_mb: int = 256

    # --- history ---
    recent_files: list[str] = field(default_factory=list)

    def as_inference_settings(self) -> dict[str, Any]:
        """Return flat dict that inference_api/request_builder.py expects."""
        return {
            "dino_checkpoint": self.dino_checkpoint,
            "dino_config_id": self.dino_config_id,
            "stabledino_checkpoint": self.stabledino_checkpoint,
            "sam_checkpoint": self.sam_checkpoint,
            "sam_model_type": self.sam_model_type,
            "sam_lora_base_checkpoint": self.sam_lora_base_checkpoint,
            "sam_lora_checkpoint": self.sam_lora_checkpoint,
            "sam_lora_rank": self.sam_lora_rank,
            "sam_lora_predict_mode": self.sam_lora_predict_mode,
            "sam_lora_refine_checkpoint": self.sam_lora_refine_checkpoint,
            "sam_lora_refine_rank": self.sam_lora_refine_rank,
            "unet_model": self.unet_checkpoint,
            "yolo_checkpoint": self.yolo_checkpoint,
            "yolo_conf": self.yolo_conf,
            "yolo_iou": self.yolo_iou,
            "yolo_imgsz": self.yolo_imgsz,
            "yolo_max_dets": self.yolo_max_dets,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "max_dets": self.max_dets,
            "unet_threshold": self.unet_threshold,
            "device": self.device,
        }

    def add_recent(self, path: str | Path) -> None:
        p = str(path)
        if p in self.recent_files:
            self.recent_files.remove(p)
        self.recent_files.insert(0, p)
        self.recent_files = self.recent_files[:20]


class SettingsIO:
    @staticmethod
    def load() -> UiSettings:
        if _SETTINGS_FILE.exists():
            try:
                raw = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                return SettingsIO._from_dict(raw)
            except Exception:
                pass
        # Try migrate from legacy .editor_app.json
        if _LEGACY_FILE.exists():
            try:
                raw = json.loads(_LEGACY_FILE.read_text(encoding="utf-8"))
                settings = SettingsIO._migrate_legacy(raw)
                SettingsIO.save(settings)
                return settings
            except Exception:
                pass
        return UiSettings()

    @staticmethod
    def save(settings: UiSettings) -> None:
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {"version": _VERSION}
            data.update(asdict(settings))
            _SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _from_dict(raw: dict[str, Any]) -> UiSettings:
        s = UiSettings()
        for f_name in s.__dataclass_fields__:  # type: ignore[attr-defined]
            if f_name in raw:
                try:
                    setattr(s, f_name, raw[f_name])
                except Exception:
                    pass
        return s

    @staticmethod
    def _migrate_legacy(raw: dict[str, Any]) -> UiSettings:
        s = UiSettings()
        s.sam_checkpoint = str(raw.get("sam_checkpoint") or "")
        s.sam_model_type = str(raw.get("sam_model_type") or "auto")
        s.sam_lora_base_checkpoint = str(raw.get("sam_lora_base_checkpoint") or s.sam_lora_base_checkpoint)
        s.sam_lora_checkpoint = str(raw.get("sam_lora_checkpoint") or raw.get("lora_checkpoint") or s.sam_lora_checkpoint)
        s.sam_lora_refine_checkpoint = str(raw.get("sam_lora_refine_checkpoint") or s.sam_lora_refine_checkpoint)
        s.dino_checkpoint = str(raw.get("dino_checkpoint") or s.dino_checkpoint)
        s.stabledino_checkpoint = str(raw.get("stabledino_checkpoint") or s.stabledino_checkpoint)
        s.box_threshold = float(raw.get("box_threshold") or s.box_threshold)
        s.text_threshold = float(raw.get("text_threshold") or s.text_threshold)
        s.max_dets = int(raw.get("max_dets") or s.max_dets)
        s.unet_checkpoint = str(raw.get("unet_model") or s.unet_checkpoint)
        s.yolo_checkpoint = str(raw.get("yolo_checkpoint") or s.yolo_checkpoint)
        s.yolo_conf = float(raw.get("yolo_conf") or s.yolo_conf)
        s.yolo_iou = float(raw.get("yolo_iou") or s.yolo_iou)
        s.yolo_imgsz = int(raw.get("yolo_imgsz") or s.yolo_imgsz)
        s.yolo_max_dets = int(raw.get("yolo_max_dets") or s.yolo_max_dets)
        s.unet_threshold = float(raw.get("unet_threshold") or s.unet_threshold)
        s.device = str(raw.get("device") or "auto")
        last = raw.get("last_workspace")
        if last:
            s.add_recent(str(last))
        return s
