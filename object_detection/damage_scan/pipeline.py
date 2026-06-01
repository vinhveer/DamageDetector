from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .detectors import DamageDetector, GroundingDinoDetector
from .geometry import GeoInput, compute_box_geometry, nms_detections
from .models import Box, Detection, ImageInfo
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
    # Tiled scan is OFF by default (original full-image behavior). Set >0 to enable
    # for high-resolution images (used by the semi-labeling pipeline).
    tiled_threshold: int = 0
    tile_size: int = 1024
    tile_overlap: int = 128
    # Optional recall overrides. 0.0 = keep each prompt spec's own value (default repo
    # behavior). The semi-labeling pipeline passes these to widen recall.
    nms_iou_override: float = 0.0
    box_threshold_override: float = 0.0
    max_box_fraction_of_image: float = 0.92
    save_overlays: bool = True
    include_full_raw_in_overlay: bool = False
    image_workers: int = 1
    service_workers: int = 0
    service_queue_size: int = 0
    service_batch_size: int = 0
    service_device_ids: str = ""
    store_image_path_mode: str = "name"

    def as_jsonable(self) -> dict:
        data = asdict(self)
        data["input_dir"] = str(self.input_dir)
        data["db_path"] = str(self.db_path)
        return data


def stored_image_path(image: ImageInfo, *, mode: str) -> str:
    raw_mode = str(mode or "name").strip().lower()
    if raw_mode == "absolute":
        return str(image.path)
    if raw_mode == "relative":
        return str(image.rel_path)
    return str(image.path.name)


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

        max_in_flight = max(image_workers, image_workers * 2)
        pending = iter(enumerate(images, start=1))
        in_flight = set()

        def submit_next() -> bool:
            try:
                index, image_path = next(pending)
            except StopIteration:
                return False
            future = executor.submit(
                self._run_one_image,
                run_id=run_id,
                input_dir=input_dir,
                image_path=image_path,
                index=index,
                total=len(images),
                log_fn=log_fn,
                detector_log_fn=detector_log_fn,
            )
            in_flight.add(future)
            return True

        with ThreadPoolExecutor(max_workers=image_workers, thread_name_prefix="damage-scan") as executor:
            while len(in_flight) < max_in_flight and submit_next():
                pass
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                while len(in_flight) < max_in_flight and submit_next():
                    pass
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
        info = read_image_info(input_dir, image_path)
        if log_fn is not None:
            max_dim = max(int(info.width), int(info.height))
            mode = "tiled" if (int(self.config.tiled_threshold) > 0 and max_dim > int(self.config.tiled_threshold)) else "full"
            log_fn(f"[{index}/{total}] {image_path.name} {info.width}x{info.height} mode={mode}")
        image_id = self.store.upsert_image(
            run_id=run_id,
            image=info,
            stored_path=stored_image_path(info, mode=str(self.config.store_image_path_mode)),
            status="running",
        )
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

        boxes = self.store.fetch_image_boxes(run_id=run_id, image_id=image_id)
        if boxes:
            geometries = compute_box_geometry(
                [GeoInput(*box) for box in boxes], int(image.width), int(image.height)
            )
            self.store.update_geometry(geometries)

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
        # Recall overrides: 0.0 means "use the prompt spec's own value" (repo default).
        box_threshold = float(self.config.box_threshold_override) if float(self.config.box_threshold_override) > 0.0 else float(spec.box_threshold)
        nms_iou = float(self.config.nms_iou_override) if float(self.config.nms_iou_override) > 0.0 else float(spec.nms_iou)

        max_dim = max(int(image.width), int(image.height))
        use_tiled = int(self.config.tiled_threshold) > 0 and max_dim > int(self.config.tiled_threshold)
        if use_tiled:
            raw_detections = self._detect_tiled_plus_full(image=image, spec=spec, box_threshold=box_threshold, log_fn=log_fn)
        else:
            raw_detections = self.detector.detect(
                image_path=image.path,
                prompt_key=spec.key,
                prompt_text=spec.prompt,
                box_threshold=box_threshold,
                text_threshold=float(spec.text_threshold),
                max_dets=int(self.config.final_max_dets_per_class),
                stage="full_raw",
                source="full",
                image_width=image.width,
                image_height=image.height,
                roi_box=None,
                log_fn=log_fn,
            )
        image_area = max(1, int(image.width) * int(image.height))
        max_area = float(self.config.max_box_fraction_of_image) * float(image_area)
        final_candidates = [det for det in raw_detections if float(det.box.area) <= max_area]
        final_candidates = nms_detections(
            final_candidates,
            iou_threshold=nms_iou,
            max_dets=int(self.config.final_max_dets_per_class),
        )
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
            for det in final_candidates
        ]
        return raw_detections, final

    def _grid_tiles(self, *, width: int, height: int) -> list[Box]:
        """Fixed-size square tiles covering the image, with a small overlap so
        damage straddling a tile edge still lands fully inside some tile."""
        size = int(self.config.tile_size)
        overlap = int(self.config.tile_overlap)
        step = max(1, size - overlap)
        tiles: list[Box] = []
        ys = list(range(0, max(1, height - overlap), step))
        xs = list(range(0, max(1, width - overlap), step))
        for y in ys:
            y1 = y
            y2 = min(height, y + size)
            if y2 - y1 < size and y2 == height:
                y1 = max(0, height - size)
            for x in xs:
                x1 = x
                x2 = min(width, x + size)
                if x2 - x1 < size and x2 == width:
                    x1 = max(0, width - size)
                tiles.append(Box(float(x1), float(y1), float(x2), float(y2)))
        return tiles

    def _detect_tiled_plus_full(
        self,
        *,
        image: ImageInfo,
        spec: PromptSpec,
        box_threshold: float,
        log_fn: Callable[[str], None] | None,
    ) -> list[Detection]:
        """Fixed 1024 grid tiles (catch small damage in HD images) + one
        full-image pass (catch large regions split across tiles), merged."""
        collected: list[Detection] = []
        tiles = self._grid_tiles(width=int(image.width), height=int(image.height))

        detect_rois = getattr(self.detector, "detect_rois", None)
        if callable(detect_rois) and tiles:
            # Batched path: push grid tiles through the worker in groups
            # (GDINO_TILE_BATCH_SIZE per forward) instead of one call per tile.
            # Result is identical to the per-tile loop, just faster on big GPUs.
            collected.extend(
                detect_rois(
                    image_path=image.path,
                    prompt_key=spec.key,
                    prompt_text=spec.prompt,
                    box_threshold=box_threshold,
                    text_threshold=float(spec.text_threshold),
                    max_dets=int(self.config.final_max_dets_per_class),
                    stage="tile_raw",
                    source="tile",
                    image_width=image.width,
                    image_height=image.height,
                    roi_boxes=tiles,
                    log_fn=log_fn,
                )
            )
        else:
            for tile in tiles:
                collected.extend(
                    self.detector.detect(
                        image_path=image.path,
                        prompt_key=spec.key,
                        prompt_text=spec.prompt,
                        box_threshold=box_threshold,
                        text_threshold=float(spec.text_threshold),
                        max_dets=int(self.config.final_max_dets_per_class),
                        stage="tile_raw",
                        source="tile",
                        image_width=image.width,
                        image_height=image.height,
                        roi_box=tile,
                        log_fn=log_fn,
                    )
                )
        collected.extend(
            self.detector.detect(
                image_path=image.path,
                prompt_key=spec.key,
                prompt_text=spec.prompt,
                box_threshold=box_threshold,
                text_threshold=float(spec.text_threshold),
                max_dets=int(self.config.final_max_dets_per_class),
                stage="full_raw",
                source="full",
                image_width=image.width,
                image_height=image.height,
                roi_box=None,
                log_fn=log_fn,
            )
        )
        return collected
