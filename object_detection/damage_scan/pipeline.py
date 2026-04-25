from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .detectors import DamageDetector, GroundingDinoDetector
from .geometry import nms_detections
from .models import Detection, ImageInfo
from .overlay import save_overlay
from .prompts import PROMPT_ORDER, PROMPT_SPECS, PromptSpec
from .sqlite_store import DamageScanStore


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DamageScanConfig:
    input_dir: Path
    db_path: Path
    checkpoint: str = ""
    device: str = "auto"
    recursive: bool = False
    limit: int = 0
    final_max_dets_per_class: int = 200
    max_box_fraction_of_image: float = 0.92
    save_overlays: bool = True
    include_full_raw_in_overlay: bool = False
    image_workers: int = 1
    service_workers: int = 0
    service_queue_size: int = 0
    service_batch_size: int = 0
    service_device_ids: str = ""

    def as_jsonable(self) -> dict:
        data = asdict(self)
        data["input_dir"] = str(self.input_dir)
        data["db_path"] = str(self.db_path)
        return data


def iter_images(input_dir: Path, *, recursive: bool, limit: int = 0) -> list[Path]:
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    if int(limit) > 0:
        images = images[: int(limit)]
    return images


def read_image_info(input_dir: Path, image_path: Path) -> ImageInfo:
    with Image.open(image_path) as image:
        width, height = image.size
    return ImageInfo(path=image_path, rel_path=image_path.relative_to(input_dir).as_posix(), width=int(width), height=int(height))


class DamageScanPipeline:
    def __init__(self, config: DamageScanConfig, *, detector: DamageDetector | None = None) -> None:
        self.config = config
        detector_output_dir = str(Path(config.db_path).expanduser().resolve().parent / "detector_cache")
        self.detector = detector or GroundingDinoDetector(
            checkpoint=str(config.checkpoint or ""),
            device=str(config.device),
            output_dir=detector_output_dir,
            service_workers=int(config.service_workers or 0),
            service_queue_size=int(config.service_queue_size or 0),
            service_batch_size=int(config.service_batch_size or 0),
            service_device_ids=str(config.service_device_ids or ""),
        )
        self.store = DamageScanStore(config.db_path)

    def close(self) -> None:
        self.detector.close()
        self.store.close()

    def run(
        self,
        *,
        log_fn: Callable[[str], None] | None = None,
        detector_log_fn: Callable[[str], None] | None = None,
    ) -> str:
        input_dir = Path(self.config.input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")
        run_id = uuid.uuid4().hex
        self.store.create_run(
            run_id=run_id,
            input_dir=input_dir,
            detector_name=self.detector.__class__.__name__,
            checkpoint=getattr(self.detector, "checkpoint", str(self.config.checkpoint or "")),
            device=str(self.config.device),
            config=self.config.as_jsonable(),
        )
        images = iter_images(input_dir, recursive=bool(self.config.recursive), limit=int(self.config.limit))
        if log_fn is not None:
            log_fn(f"Damage scan run_id={run_id} images={len(images)} db={self.config.db_path}")
        image_workers = max(1, int(self.config.image_workers or 1))
        if image_workers <= 1:
            for index, image_path in enumerate(images, start=1):
                self._run_one_image(
                    run_id=run_id,
                    input_dir=input_dir,
                    image_path=image_path,
                    index=index,
                    total=len(images),
                    log_fn=log_fn,
                    detector_log_fn=detector_log_fn,
                )
            return run_id

        with ThreadPoolExecutor(max_workers=image_workers, thread_name_prefix="damage-scan") as executor:
            futures = [
                executor.submit(
                    self._run_one_image,
                    run_id=run_id,
                    input_dir=input_dir,
                    image_path=image_path,
                    index=index,
                    total=len(images),
                    log_fn=log_fn,
                    detector_log_fn=detector_log_fn,
                )
                for index, image_path in enumerate(images, start=1)
            ]
            for future in as_completed(futures):
                future.result()
        return run_id

    def _run_one_image(
        self,
        *,
        run_id: str,
        input_dir: Path,
        image_path: Path,
        index: int,
        total: int,
        log_fn: Callable[[str], None] | None,
        detector_log_fn: Callable[[str], None] | None,
    ) -> None:
        if log_fn is not None:
            log_fn(f"[{index}/{total}] {image_path.name}")
        info = read_image_info(input_dir, image_path)
        image_id = self.store.upsert_image(run_id=run_id, image=info, status="running")
        try:
            self._process_image(
                run_id=run_id,
                image_id=image_id,
                image=info,
                log_fn=detector_log_fn,
                status_log_fn=log_fn,
            )
            self.store.mark_image_done(image_id=image_id)
        except Exception as exc:
            self.store.mark_image_error(image_id=image_id, error=exc)
            if log_fn is not None:
                log_fn(f"  error: {exc.__class__.__name__}: {exc}")

    def _process_image(
        self,
        *,
        run_id: str,
        image_id: int,
        image: ImageInfo,
        log_fn: Callable[[str], None] | None,
        status_log_fn: Callable[[str], None] | None,
    ) -> None:
        overlay_items: list[Detection] = []
        final_items: list[Detection] = []
        persisted_items: list[Detection] = []
        for prompt_key in PROMPT_ORDER:
            spec = PROMPT_SPECS[prompt_key]
            raw, final = self._run_full_prompt(image=image, spec=spec, log_fn=log_fn)
            persisted_items.extend(raw)
            persisted_items.extend(final)
            overlay_items.extend(raw)
            final_items.extend(final)
            if status_log_fn is not None:
                status_log_fn(f"  {prompt_key}: raw={len(raw)} final={len(final)}")

        self.store.insert_detections_bulk(run_id=run_id, image_id=image_id, detections=persisted_items)

        if bool(self.config.save_overlays):
            overlay_path = Path(self.config.db_path).expanduser().resolve().parent / "overlays" / image.rel_path
            overlay_path = overlay_path.with_name(f"{overlay_path.stem}_overlay.png")
            save_overlay(
                image=image,
                detections=overlay_items + final_items,
                output_path=overlay_path,
                include_proposals=False,
                include_proposal_raw=bool(self.config.include_full_raw_in_overlay),
            )
            if status_log_fn is not None:
                status_log_fn(f"  overlay={overlay_path}")

    def _run_full_prompt(
        self,
        *,
        image: ImageInfo,
        spec: PromptSpec,
        log_fn: Callable[[str], None] | None,
    ) -> tuple[list[Detection], list[Detection]]:
        raw_detections = self.detector.detect(
            image_path=image.path,
            prompt_key=spec.key,
            prompt_text=spec.prompt,
            box_threshold=float(spec.box_threshold),
            text_threshold=float(spec.text_threshold),
            max_dets=int(self.config.final_max_dets_per_class),
            stage="full_raw",
            source="full",
            image_width=image.width,
            image_height=image.height,
            roi_box=None,
            log_fn=log_fn,
        )
        raw_detections = [det for det in raw_detections if self._box_within_image_fraction(det, image=image)]
        final = nms_detections(raw_detections, iou_threshold=float(spec.nms_iou), max_dets=int(self.config.final_max_dets_per_class))
        final = [det for det in final if self._box_within_image_fraction(det, image=image)]
        final = [
            Detection(
                box=det.box,
                label=det.label,
                score=float(det.score),
                prompt_key=det.prompt_key,
                prompt_text=det.prompt_text,
                stage="final",
                source="full",
                model_name=det.model_name,
                parent_detection_id=None,
                raw={**dict(det.raw or {}), "final_prompt_key": spec.key, "mode": "full_image"},
            )
            for det in final
        ]
        return raw_detections, final

    def _box_within_image_fraction(self, det: Detection, *, image: ImageInfo) -> bool:
        max_fraction = float(self.config.max_box_fraction_of_image)
        return float(det.box.width) <= float(image.width) * max_fraction and float(det.box.height) <= float(image.height) * max_fraction
