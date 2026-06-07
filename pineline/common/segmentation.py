from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pineline.common.model_defaults import (
    default_sam_finetune_delta,
    default_sam_vit_b_checkpoint,
    default_unet_model,
)


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class SegmentationConfig:
    enabled: bool = True
    output_dir: str = ""
    sam_checkpoint: str = ""
    sam_model_type: str = "vit_b"
    unet_model: str = ""
    sam_finetune_checkpoint: str = ""
    sam_finetune_delta_type: str = "lora"
    sam_finetune_delta_checkpoint: str = ""
    sam_finetune_model_type: str = "vit_b"
    threshold: float = 0.5
    device: str = "auto"


def default_segmentation_config(
    *,
    enabled: bool = True,
    output_dir: str | Path | None = None,
    sam_checkpoint: str | Path | None = None,
    sam_model_type: str = "vit_b",
    unet_model: str | Path | None = None,
    sam_finetune_checkpoint: str | Path | None = None,
    sam_finetune_delta_type: str = "lora",
    sam_finetune_delta_checkpoint: str | Path | None = None,
    sam_finetune_model_type: str = "vit_b",
    threshold: float = 0.5,
    device: str = "auto",
) -> SegmentationConfig:
    sam_b = default_sam_vit_b_checkpoint()

    def _path(value: str | Path | None, fallback: Path) -> str:
        return str(Path(value or fallback).expanduser().resolve())

    return SegmentationConfig(
        enabled=bool(enabled),
        output_dir=str(Path(output_dir).expanduser().resolve()) if output_dir else "",
        sam_checkpoint=_path(sam_checkpoint, sam_b),
        sam_model_type=str(sam_model_type or "vit_b"),
        unet_model=_path(unet_model, default_unet_model()),
        sam_finetune_checkpoint=_path(sam_finetune_checkpoint, sam_b),
        sam_finetune_delta_type=str(sam_finetune_delta_type or "lora"),
        sam_finetune_delta_checkpoint=_path(sam_finetune_delta_checkpoint, default_sam_finetune_delta()),
        sam_finetune_model_type=str(sam_finetune_model_type or "vit_b"),
        threshold=float(threshold),
        device=str(device or "auto"),
    )


def segmentation_model_metadata(config: SegmentationConfig) -> dict[str, Any]:
    return {
        "other_damage": [
            {
                "segmenter": _sam_segmenter_name(config.sam_model_type),
                "checkpoint": config.sam_checkpoint,
                "model_type": config.sam_model_type,
            }
        ],
        "crack": [
            {"segmenter": "unet", "model": config.unet_model},
            {
                "segmenter": "sam_finetune_vit_b",
                "checkpoint": config.sam_finetune_checkpoint,
                "delta_type": config.sam_finetune_delta_type,
                "delta_checkpoint": config.sam_finetune_delta_checkpoint,
                "model_type": config.sam_finetune_model_type,
            },
        ],
    }


class MultiSegmenter:
    def __init__(self, config: SegmentationConfig, *, log: LogFn | None = None) -> None:
        self.config = config
        self.log = log or (lambda s: None)
        self._sam_service = None
        self._sam_ft_service = None
        self._unet_service = None

    def close(self) -> None:
        for service in (self._sam_service, self._sam_ft_service, self._unet_service):
            if service is None:
                continue
            try:
                service.close()
            except Exception:
                pass

    def segment(self, image_path: Path, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.config.enabled or not detections:
            return []
        out_dir = Path(self.config.output_dir or image_path.parent / "segments")
        out_dir.mkdir(parents=True, exist_ok=True)
        crack = [d for d in detections if str(d.get("label") or "").strip().lower() == "crack"]
        other = [d for d in detections if str(d.get("label") or "").strip().lower() != "crack"]
        rows: list[dict[str, Any]] = []
        self.log(f"  segmentation start: boxes={len(detections)} crack={len(crack)} other={len(other)}")
        if other:
            sam_name = _sam_segmenter_name(self.config.sam_model_type)
            try:
                self.log(f"  segment[{sam_name}] start: {len(other)} boxes")
                sam_rows = self._segment_sam_other(image_path, other, out_dir / sam_name, sam_name)
                rows.extend(sam_rows)
                self.log(f"  segment[{sam_name}] done: {len(sam_rows)} rows")
            except Exception as exc:
                self.log(f"{sam_name} segmentation failed, skipped: {exc}")
        if crack:
            try:
                self.log(f"  segment[unet] start: {len(crack)} boxes")
                unet_rows = self._segment_unet_crack(image_path, crack, out_dir / "unet")
                rows.extend(unet_rows)
                self.log(f"  segment[unet] done: {len(unet_rows)} rows")
            except Exception as exc:
                self.log(f"unet segmentation failed, skipped: {exc}")
            try:
                self.log(f"  segment[sam_finetune_vit_b] start: {len(crack)} boxes")
                sam_ft_rows = self._segment_sam_finetune_crack(image_path, crack, out_dir / "sam_finetune_vit_b")
                rows.extend(sam_ft_rows)
                self.log(f"  segment[sam_finetune_vit_b] done: {len(sam_ft_rows)} rows")
            except Exception as exc:
                self.log(f"sam_finetune_vit_b segmentation failed, skipped: {exc}")
        self.log(f"  segmentation done: {len(rows)} rows")
        return rows

    def _segment_sam_other(
        self,
        image_path: Path,
        detections: list[dict[str, Any]],
        out_dir: Path,
        segmenter_name: str,
    ) -> list[dict[str, Any]]:
        if not Path(self.config.sam_checkpoint).is_file():
            self.log(f"{segmenter_name} checkpoint not found, other-damage segmentation skipped: {self.config.sam_checkpoint}")
            return []
        from segmentation.sam.runtime.client import get_sam_service

        self._sam_service = self._sam_service or get_sam_service()
        boxes = [_segment_box_payload(d) for d in detections]
        result = self._sam_service.call(
            "segment_boxes",
            {
                "image_path": str(image_path),
                "boxes": boxes,
                "params": {
                    "sam_checkpoint": self.config.sam_checkpoint,
                    "sam_model_type": self.config.sam_model_type,
                    "output_dir": str(out_dir),
                    "task_group": "more_damage",
                    "device": self.config.device,
                },
            },
        )
        return _rows_from_box_segment_result(segmenter_name, "other_damage", result, detections)

    def _segment_sam_finetune_crack(self, image_path: Path, detections: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
        if not Path(self.config.sam_finetune_checkpoint).is_file():
            self.log(f"SAM finetune base checkpoint not found, skipped: {self.config.sam_finetune_checkpoint}")
            return []
        if not Path(self.config.sam_finetune_delta_checkpoint).is_file():
            self.log(f"SAM finetune delta checkpoint not found, skipped: {self.config.sam_finetune_delta_checkpoint}")
            return []
        from segmentation.sam.finetune.client import get_sam_finetune_service

        self._sam_ft_service = self._sam_ft_service or get_sam_finetune_service()
        boxes = [_segment_box_payload(d) for d in detections]
        result = self._sam_ft_service.call(
            "segment_boxes",
            {
                "image_path": str(image_path),
                "boxes": boxes,
                "params": {
                    "sam_checkpoint": self.config.sam_finetune_checkpoint,
                    "sam_model_type": self.config.sam_finetune_model_type,
                    "delta_type": self.config.sam_finetune_delta_type,
                    "delta_checkpoint": self.config.sam_finetune_delta_checkpoint,
                    "output_dir": str(out_dir),
                    "task_group": "crack_only",
                    "threshold": "auto",
                    "device": self.config.device,
                },
            },
        )
        return _rows_from_box_segment_result("sam_finetune_vit_b", "crack", result, detections)

    def _segment_unet_crack(self, image_path: Path, detections: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
        if not Path(self.config.unet_model).is_file():
            self.log(f"UNet model not found, crack segmentation skipped: {self.config.unet_model}")
            return []
        from segmentation.unet.client import get_unet_service

        self._unet_service = self._unet_service or get_unet_service()
        rois = []
        for det in detections:
            x1, y1, x2, y2 = [float(v) for v in det.get("box")]
            rois.append([int(x1), int(y1), int(x2), int(y2)])
        result = self._unet_service.call(
            "run_rois",
            {
                "image_path": str(image_path),
                "rois": rois,
                "params": {
                    "model_path": self.config.unet_model,
                    "output_dir": str(out_dir),
                    "threshold": float(self.config.threshold),
                    "mode": "tile",
                    "input_size": 512,
                    "tile_overlap": 0,
                    "device": self.config.device,
                },
            },
        )
        rows = []
        for det in detections:
            rows.append({
                "detector_name": det.get("detector_name"),
                "det_idx": det.get("det_idx"),
                "label": det.get("label"),
                "segmenter_name": "unet",
                "task_group": "crack",
                "mask_path": result.get("mask_path"),
                "overlay_path": result.get("overlay_path"),
                "mask_area_px": _mask_area_in_roi(result.get("mask_path") or "", det.get("box") or [0, 0, 0, 0]),
                "mask_count": 1 if result.get("mask_path") else 0,
                "score": None,
                "extra_json": json.dumps({"output_dir": result.get("output_dir")}, ensure_ascii=False),
            })
        return rows


def _mask_area_from_b64(mask_b64: str | None) -> int:
    if not mask_b64:
        return 0
    import cv2
    import numpy as np

    data = np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    return int(np.count_nonzero(mask > 0))


def _mask_area_in_roi(mask_path: str | Path, box: list[float]) -> int:
    import cv2
    import numpy as np

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    x1i = max(0, min(w, int(np.floor(x1))))
    y1i = max(0, min(h, int(np.floor(y1))))
    x2i = max(0, min(w, int(np.ceil(x2))))
    y2i = max(0, min(h, int(np.ceil(y2))))
    if x2i <= x1i or y2i <= y1i:
        return 0
    return int(np.count_nonzero(mask[y1i:y2i, x1i:x2i] > 0))


def _segment_box_payload(det: dict[str, Any]) -> dict[str, Any]:
    return {
        "box": [float(v) for v in det.get("box")],
        "label": str(det.get("label") or ""),
        "score": float(det.get("score") or 0.0),
        "det_idx": int(det.get("det_idx")),
        "detector_name": str(det.get("detector_name") or ""),
    }


def _sam_segmenter_name(model_type: str) -> str:
    normalized = str(model_type or "vit_b").strip().lower()
    return f"sam_{normalized}"


def _rows_from_box_segment_result(
    segmenter_name: str,
    task_group: str,
    result: dict[str, Any],
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {(str(d.get("detector_name") or ""), int(d.get("det_idx"))): d for d in detections}
    rows = []
    for det in result.get("detections") or []:
        key = (str(det.get("detector_name") or ""), int(det.get("det_idx", -1)))
        source = by_key.get(key)
        if source is None:
            continue
        rows.append({
            "detector_name": source.get("detector_name"),
            "det_idx": source.get("det_idx"),
            "label": source.get("label"),
            "segmenter_name": segmenter_name,
            "task_group": task_group,
            "mask_path": result.get("mask_path"),
            "overlay_path": result.get("overlay_path"),
            "mask_area_px": _mask_area_from_b64(det.get("mask_b64")),
            "mask_count": 1 if det.get("mask_b64") else 0,
            "score": det.get("score"),
            "extra_json": json.dumps({"output_dir": result.get("output_dir")}, ensure_ascii=False),
        })
    return rows
