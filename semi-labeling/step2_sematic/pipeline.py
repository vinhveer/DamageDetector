from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from clip_model import OpenClipSemanticClassifier
from prompts import POS_PROMPTS
from sqlite_store import SourceDetection, Step2SemanticStore


@dataclass(frozen=True)
class Step2SemanticConfig:
    db_path: Path
    source_run_id: str = "latest"
    stage: str = "final"
    limit: int = 0
    detection_ids: tuple[int, ...] = ()
    image_root: Path | None = None
    shard_index: int = 0
    num_shards: int = 1
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"
    batch_size: int = 16
    save_crops: bool = False
    crop_dir: Path | None = None


def _sanitize_crop_box(detection: SourceDetection) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(detection.image_width) - 1, int(math.floor(float(detection.x1)))))
    y1 = max(0, min(int(detection.image_height) - 1, int(math.floor(float(detection.y1)))))
    x2 = max(0, min(int(detection.image_width), int(math.ceil(float(detection.x2)))))
    y2 = max(0, min(int(detection.image_height), int(math.ceil(float(detection.y2)))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop box for detection_id={detection.detection_id}: ({x1}, {y1}, {x2}, {y2})"
        )
    return x1, y1, x2, y2


def _select_shard(detections: list[SourceDetection], *, shard_index: int, num_shards: int) -> list[SourceDetection]:
    if int(num_shards) <= 1:
        return list(detections)
    if int(shard_index) < 0 or int(shard_index) >= int(num_shards):
        raise ValueError(f"Invalid shard selection: shard_index={shard_index}, num_shards={num_shards}")
    return [item for idx, item in enumerate(detections) if idx % int(num_shards) == int(shard_index)]


class Step2SemanticPipeline:
    def __init__(self, config: Step2SemanticConfig) -> None:
        self.config = config
        self.store = Step2SemanticStore(config.db_path)
        self.classifier = OpenClipSemanticClassifier(
            model_name=str(config.model_name),
            pretrained=str(config.pretrained),
            device=str(config.device),
            prompt_groups=POS_PROMPTS,
        )

    def close(self) -> None:
        self.store.close()

    def run(self, *, log_fn: Callable[[str], None] | None = None) -> str:
        semantic_run_id = uuid.uuid4().hex
        source_run_id = self.store.resolve_source_run_id(self.config.source_run_id)
        detections = self.store.list_source_detections(
            source_run_id=source_run_id,
            stage=str(self.config.stage),
            limit=int(self.config.limit),
            detection_ids=[int(item) for item in self.config.detection_ids],
        )
        detections = _select_shard(
            detections,
            shard_index=int(self.config.shard_index),
            num_shards=int(self.config.num_shards),
        )
        if not detections:
            raise RuntimeError("No detections matched the current selection.")

        crop_root = self._resolve_crop_root(semantic_run_id) if bool(self.config.save_crops) else None
        self.store.create_run(
            semantic_run_id=semantic_run_id,
            source_run_id=source_run_id,
            source_stage=str(self.config.stage),
            model_name=self.classifier.model_name,
            pretrained=self.classifier.pretrained,
            device=self.classifier.device,
            prompt_config={"positive": POS_PROMPTS},
            options={
                "limit": int(self.config.limit),
                "detection_ids": [int(item) for item in self.config.detection_ids],
                "image_root": str(self.config.image_root) if self.config.image_root is not None else "",
                "shard_index": int(self.config.shard_index),
                "num_shards": int(self.config.num_shards),
                "save_crops": bool(self.config.save_crops),
                "crop_dir": str(crop_root) if crop_root is not None else "",
            },
        )
        if log_fn is not None:
            log_fn(
                f"openclip_semantic_run_id={semantic_run_id} detections={len(detections)} model={self.classifier.model_name} pretrained={self.classifier.pretrained}"
            )

        ok_count = 0
        error_count = 0
        batch_size = max(1, int(self.config.batch_size or 1))
        for batch_start in range(0, len(detections), batch_size):
            batch = detections[batch_start : batch_start + batch_size]
            prepared: list[tuple[SourceDetection, Image.Image, str]] = []
            for local_index, detection in enumerate(batch, start=batch_start + 1):
                try:
                    crop_image, resolved_image_path = self._load_detection_crop(detection)
                    prepared.append((detection, crop_image, str(resolved_image_path)))
                except Exception as exc:
                    self.store.insert_error_result(
                        semantic_run_id=semantic_run_id,
                        detection=detection,
                        crop_path="",
                        exc=exc,
                    )
                    error_count += 1
                    if log_fn is not None:
                        log_fn(f"[{local_index}/{len(detections)}] detection_id={detection.detection_id} error={exc}")

            if not prepared:
                continue

            classifications = self.classifier.classify_images([item[1] for item in prepared])
            for local_index, ((detection, crop_image, resolved_image_path), classification) in enumerate(
                zip(prepared, classifications, strict=True),
                start=batch_start + 1,
            ):
                crop_path = ""
                classification["resolved_image_path"] = resolved_image_path
                if crop_root is not None:
                    crop_path = str(self._save_crop(crop_root=crop_root, detection=detection, crop_image=crop_image))
                self.store.insert_success_result(
                    semantic_run_id=semantic_run_id,
                    detection=detection,
                    crop_path=crop_path,
                    classification=classification,
                )
                ok_count += 1
                if log_fn is not None:
                    log_fn(
                        f"[{local_index}/{len(detections)}] detection_id={detection.detection_id} label={classification['predicted_label']} pct={classification['predicted_probability_pct']:.2f}"
                    )

        if log_fn is not None:
            log_fn(f"openclip_semantic_done ok={ok_count} error={error_count} db={self.store.db_path}")
        return semantic_run_id

    def _load_detection_crop(self, detection: SourceDetection) -> tuple[Image.Image, Path]:
        image_path = self._resolve_image_path(detection)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        crop_box = _sanitize_crop_box(detection)
        with Image.open(image_path) as image:
            crop_image = image.convert("RGB").crop(crop_box)
        return crop_image, image_path

    def _resolve_image_path(self, detection: SourceDetection) -> Path:
        candidates: list[Path] = []
        if self.config.image_root is not None:
            root = Path(self.config.image_root).expanduser().resolve()
            candidates.append(root / detection.image_rel_path)
            candidates.append(root / detection.image_name)

        stored_raw = str(detection.stored_image_path or "").strip()
        if stored_raw:
            stored_path = Path(stored_raw).expanduser()
            if stored_path.is_absolute():
                candidates.append(stored_path)
            else:
                candidates.append(detection.source_input_dir / stored_raw)

        candidates.append(detection.source_input_dir / detection.image_rel_path)
        candidates.append(detection.source_input_dir / detection.image_name)

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                return candidate.resolve()

        if self.config.image_root is not None:
            return (Path(self.config.image_root).expanduser().resolve() / detection.image_rel_path)
        return (detection.source_input_dir / detection.image_rel_path).expanduser()

    def _resolve_crop_root(self, semantic_run_id: str) -> Path:
        base = self.config.crop_dir
        if base is None:
            base = self.store.db_path.parent / "step2_sematic_crops"
        root = Path(base).expanduser().resolve() / semantic_run_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _save_crop(self, *, crop_root: Path, detection: SourceDetection, crop_image: Image.Image) -> Path:
        rel_path = Path(detection.image_rel_path)
        output_dir = crop_root / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{rel_path.stem}__det{detection.detection_id}.png"
        crop_image.save(output_path)
        return output_path
