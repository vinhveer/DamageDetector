from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from object_detection.dino.client import get_dino_service
from object_detection.dino.engine import default_gdino_checkpoint

from .geometry import clip_box
from .models import Box, Detection


class DamageDetector(ABC):
    @abstractmethod
    def detect(
        self,
        *,
        image_path: Path,
        prompt_key: str,
        prompt_text: str,
        box_threshold: float,
        text_threshold: float,
        max_dets: int,
        stage: str,
        source: str,
        image_width: int,
        image_height: int,
        roi_box: Box | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class GroundingDinoDetector(DamageDetector):
    def __init__(
        self,
        *,
        checkpoint: str = "",
        device: str = "auto",
        output_dir: str = "results_damage_scan",
        service_workers: int = 0,
        service_queue_size: int = 0,
        service_batch_size: int = 0,
        service_device_ids: str = "",
    ) -> None:
        self.checkpoint = str(checkpoint or "").strip() or str(default_gdino_checkpoint() or "").strip()
        if not self.checkpoint:
            raise RuntimeError("No GroundingDINO checkpoint available. Pass --checkpoint or download the default model.")
        self.device = str(device or "auto")
        self.output_dir = str(output_dir)
        self._service = get_dino_service(
            num_workers=int(service_workers or 0),
            queue_size=int(service_queue_size or 0),
            batch_size=int(service_batch_size or 0),
            device_ids=str(service_device_ids or "").strip() or None,
        )

    def detect(
        self,
        *,
        image_path: Path,
        prompt_key: str,
        prompt_text: str,
        box_threshold: float,
        text_threshold: float,
        max_dets: int,
        stage: str,
        source: str,
        image_width: int,
        image_height: int,
        roi_box: Box | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> list[Detection]:
        queries = [item.strip() for item in str(prompt_text).split(",") if item.strip()]
        params: dict[str, Any] = {
            "gdino_checkpoint": self.checkpoint,
            "gdino_config_id": "auto",
            "text_queries": queries,
            "box_threshold": float(box_threshold),
            "text_threshold": float(text_threshold),
            "max_dets": int(max_dets),
            "device": self.device,
            "output_dir": self.output_dir,
        }
        if roi_box is not None:
            params["roi_box"] = list(roi_box.as_int_xyxy())

        result = self._service.call(
            "predict",
            {"image_path": str(image_path), "params": params},
            log_fn=log_fn,
        )

        detections: list[Detection] = []
        for raw in list((result or {}).get("detections") or []):
            raw_box = raw.get("box")
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                continue
            clipped = clip_box(
                Box(float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])),
                width=int(image_width),
                height=int(image_height),
            )
            if clipped is None:
                continue
            detections.append(
                Detection(
                    box=clipped,
                    label=str(raw.get("label") or prompt_key),
                    score=float(raw.get("score") or 0.0),
                    prompt_key=str(prompt_key),
                    prompt_text=str(prompt_text),
                    stage=str(stage),
                    source=str(source),
                    model_name=str(raw.get("model_name") or "groundingdino"),
                    raw=dict(raw),
                )
            )
        return detections

    def close(self) -> None:
        self._service.close()
