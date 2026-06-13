from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from object_detection.dino.client import get_dino_service
from object_detection.dino.engine import default_gdino_checkpoint

from pineline.common.model_defaults import default_stabledino_checkpoint, default_yolo_model


LogFn = Callable[[str], None]

TRAINED_DETECTOR_TILE_SIZE = 768
TRAINED_DETECTOR_TILE_OVERLAP = 128
TRAINED_DETECTOR_TILE_SIZES = (768, 1536, 2304)
TRAINED_DETECTOR_NMS_IOU = 0.50
TRAINED_DETECTOR_MAX_BLACK_RATIO = 0.35
TRAINED_DETECTOR_BLACK_PIXEL_THRESHOLD = 12


@dataclass(frozen=True)
class DetectionConfig:
    models: tuple[str, ...] = ("gdino", "yolo", "stabledino")
    gdino_checkpoint: str = ""
    yolo_model: str = ""
    stabledino_checkpoint: str = ""
    box_threshold: float = 0.10
    text_threshold: float = 0.10
    yolo_conf: float = 0.05
    yolo_iou: float = 0.45
    stabledino_conf: float = 0.05
    max_dets: int = 150
    device: str = "auto"
    tiled_threshold: int = 400
    tile_scales: tuple[str, ...] = ("small", "medium")
    recursive_max_depth: int = 2
    min_box_px: int = 12
    recursive_tile_batch_size: int = 0
    gdino_service_workers: int = 0
    gdino_service_queue_size: int = 0
    gdino_service_batch_size: int = 0
    gdino_service_device_ids: str = ""
    stabledino_output_dir: str = ""
    disable_tiled_nms: bool = False


def parse_model_names(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ("gdino", "yolo", "stabledino")
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = list(raw)
    names = []
    seen = set()
    aliases = {"groundingdino": "gdino", "grounding_dino": "gdino", "stable-dino": "stabledino"}
    for part in parts:
        name = aliases.get(str(part or "").strip().lower(), str(part or "").strip().lower())
        if not name or name in seen:
            continue
        if name not in {"gdino", "yolo", "stabledino"}:
            raise ValueError(f"Unsupported detector: {name}")
        seen.add(name)
        names.append(name)
    return tuple(names)


def resolve_gdino_checkpoint(raw: str | None) -> str:
    ckpt = str(raw or "").strip() or str(default_gdino_checkpoint() or "").strip()
    if not ckpt:
        raise RuntimeError("No GroundingDINO checkpoint available.")
    return ckpt


def _dedupe_names(names: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in names:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _normalize_class_mapping(raw: Any) -> dict[int, str]:
    if isinstance(raw, dict):
        out: dict[int, str] = {}
        for key, value in raw.items():
            try:
                out[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(raw)}
    return {}


def _load_names_from_coco(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    categories = payload.get("categories") if isinstance(payload, dict) else None
    if not isinstance(categories, list):
        return []
    rows: list[tuple[int, str]] = []
    for item in categories:
        if not isinstance(item, dict):
            continue
        try:
            rows.append((int(item.get("id")), str(item.get("name") or "").strip()))
        except (TypeError, ValueError):
            continue
    return _dedupe_names(name for _, name in sorted(rows, key=lambda row: row[0]))


def _load_names_from_metrics(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    per_class = payload.get("per_class") if isinstance(payload, dict) else None
    if not isinstance(per_class, list):
        return []
    return _dedupe_names(item.get("class") for item in per_class if isinstance(item, dict))


def _load_class_names_near(anchor: Path) -> list[str]:
    search_roots: list[Path] = []
    for root in [anchor.parent, *anchor.parents]:
        if root in search_roots:
            continue
        search_roots.append(root)
        if root.name == "semi_labeling_training":
            break
    for root in search_roots:
        for split in ("train", "val", "test"):
            names = _load_names_from_coco(root / "dataset_cache" / f"{split}_coco.json")
            if names:
                return names
        for filename in ("val_metrics_val.json", "val_metrics_test.json"):
            names = _load_names_from_metrics(root / filename)
            if names:
                return names
    return []


def _tile_boxes(width: int, height: int, *, tile_size: int = TRAINED_DETECTOR_TILE_SIZE, overlap: int = TRAINED_DETECTOR_TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    width = int(width)
    height = int(height)
    tile_size = max(1, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size - 1))
    stride = max(1, tile_size - overlap)

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, max(1, length - tile_size + 1), stride))
        values.append(length - tile_size)
        return sorted(set(values))

    boxes: list[tuple[int, int, int, int]] = []
    for y1 in starts(height):
        for x1 in starts(width):
            boxes.append((x1, y1, min(width, x1 + tile_size), min(height, y1 + tile_size)))
    return boxes


def _tile_has_too_much_black(image, box: tuple[int, int, int, int], *, max_ratio: float = TRAINED_DETECTOR_MAX_BLACK_RATIO) -> bool:
    import numpy as np

    x1, y1, x2, y2 = box
    tile = np.asarray(image.crop((x1, y1, x2, y2)).convert("RGB"))
    if tile.size == 0:
        return True
    black = np.max(tile, axis=2) <= TRAINED_DETECTOR_BLACK_PIXEL_THRESHOLD
    return float(black.mean()) > float(max_ratio)


def _box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _nms_detections(dets: list[dict[str, Any]], *, iou_threshold: float = TRAINED_DETECTOR_NMS_IOU, max_dets: int = 0) -> list[dict[str, Any]]:
    ordered = sorted(dets, key=lambda d: float(d.get("score") or 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for det in ordered:
        box = det.get("box") or []
        label = str(det.get("label") or "")
        if len(box) != 4:
            continue
        if any(label == str(prev.get("label") or "") and _box_iou(box, prev.get("box") or []) > iou_threshold for prev in kept):
            continue
        kept.append(det)
        if max_dets > 0 and len(kept) >= max_dets:
            break
    return kept


def default_detection_config(
    *,
    models: str | Iterable[str] | None = None,
    gdino_checkpoint: str | None = None,
    yolo_model: str | None = None,
    stabledino_checkpoint: str | None = None,
    box_threshold: float = 0.10,
    text_threshold: float = 0.10,
    yolo_conf: float = 0.05,
    yolo_iou: float = 0.45,
    stabledino_conf: float = 0.05,
    max_dets: int = 150,
    device: str = "auto",
    tiled_threshold: int = 400,
    tile_scales: Iterable[str] | None = None,
    recursive_max_depth: int = 2,
    min_box_px: int = 12,
    recursive_tile_batch_size: int = 0,
    gdino_service_workers: int = 0,
    gdino_service_queue_size: int = 0,
    gdino_service_batch_size: int = 0,
    gdino_service_device_ids: str | None = None,
    stabledino_output_dir: str | Path | None = None,
    disable_tiled_nms: bool = False,
) -> DetectionConfig:
    parsed_models = parse_model_names(models)
    return DetectionConfig(
        models=parsed_models,
        gdino_checkpoint=resolve_gdino_checkpoint(gdino_checkpoint) if "gdino" in parsed_models else str(gdino_checkpoint or ""),
        yolo_model=str(yolo_model or default_yolo_model()),
        stabledino_checkpoint=str(stabledino_checkpoint or default_stabledino_checkpoint()),
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        yolo_conf=float(yolo_conf),
        yolo_iou=float(yolo_iou),
        stabledino_conf=float(stabledino_conf),
        max_dets=int(max_dets),
        device=str(device or "auto"),
        tiled_threshold=int(tiled_threshold),
        tile_scales=tuple(tile_scales or ("small", "medium")),
        recursive_max_depth=int(recursive_max_depth),
        min_box_px=int(min_box_px),
        recursive_tile_batch_size=int(recursive_tile_batch_size or 0),
        gdino_service_workers=int(gdino_service_workers or 0),
        gdino_service_queue_size=int(gdino_service_queue_size or 0),
        gdino_service_batch_size=int(gdino_service_batch_size or 0),
        gdino_service_device_ids=str(gdino_service_device_ids or "").strip(),
        stabledino_output_dir=str(stabledino_output_dir or ""),
        disable_tiled_nms=bool(disable_tiled_nms),
    )


class MultiDetector:
    def __init__(self, config: DetectionConfig, *, log: LogFn | None = None) -> None:
        self.config = config
        self.log = log or (lambda s: None)
        self._dino_service = None
        self._yolo_model = None
        self._yolo_device = None
        self._yolo_names: dict[int, str] = {}
        self._stabledino_cache: dict[str, list[dict[str, Any]]] = {}
        self._stabledino_names: list[str] | None = None
        self._stabledino_ready = False
        self._active_models = tuple(self._resolve_active_models())
        self.log(
            "Detection models active="
            f"{','.join(self._active_models) or 'none'} "
            f"gdino_tile_batch={int(self.config.recursive_tile_batch_size or 0) or 'default'} "
            f"gdino_workers={int(self.config.gdino_service_workers or 0) or 'auto'}"
        )

    @property
    def active_models(self) -> tuple[str, ...]:
        return self._active_models

    def _resolve_active_models(self) -> list[str]:
        active = []
        for name in self.config.models:
            if name == "yolo" and not Path(self.config.yolo_model).is_file():
                self.log(f"YOLO model not found, skipped: {self.config.yolo_model}")
                continue
            if name == "stabledino" and not Path(self.config.stabledino_checkpoint).is_file():
                self.log(f"StableDINO checkpoint not found, skipped: {self.config.stabledino_checkpoint}")
                continue
            active.append(name)
        return active

    def close(self) -> None:
        if self._dino_service is not None:
            try:
                self._dino_service.close()
            except Exception:
                pass

    def detect(self, image_path: Path, *, width: int, height: int, queries: list[str], names: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in self.active_models:
            try:
                self.log(f"  detect[{name}] start: {image_path.name}")
                if name == "gdino":
                    dets = self._detect_gdino(image_path, width=width, height=height, queries=queries)
                elif name == "yolo":
                    dets = self._detect_yolo(image_path)
                elif name == "stabledino":
                    dets = self._stabledino_for(image_path, names=names)
                else:
                    dets = []
            except Exception as exc:
                self.log(f"{name} detect failed for {image_path.name}: {exc}")
                continue
            self.log(f"  detect[{name}] done: {len(dets)} boxes")
            for det in dets:
                item = dict(det)
                item["detector_name"] = name
                rows.append(item)
        return rows

    def _stabledino_for(self, image_path: Path, *, names: list[str]) -> list[dict[str, Any]]:
        key = str(Path(image_path).resolve())
        if self._stabledino_ready:
            return list(self._stabledino_cache.get(key) or [])
        stable_names = self._resolve_stabledino_names(names)
        return self._run_stabledino_tiled_batch([image_path], names=stable_names).get(key, [])

    def _resolve_stabledino_names(self, fallback: list[str] | None = None) -> list[str]:
        if self._stabledino_names is None:
            checkpoint = Path(self.config.stabledino_checkpoint).expanduser().resolve()
            self._stabledino_names = _load_class_names_near(checkpoint) or _dedupe_names(fallback or [])
            if not self._stabledino_names:
                self._stabledino_names = ["crack", "mold", "spall"]
            self.log(f"StableDINO classes={','.join(self._stabledino_names)}")
        return list(self._stabledino_names)

    def _detect_gdino(self, image_path: Path, *, width: int, height: int, queries: list[str]) -> list[dict[str, Any]]:
        if self._dino_service is None:
            self._dino_service = get_dino_service(
                num_workers=int(self.config.gdino_service_workers or 0),
                queue_size=int(self.config.gdino_service_queue_size or 0),
                batch_size=int(self.config.gdino_service_batch_size or 0),
                device_ids=self.config.gdino_service_device_ids or None,
            )
        max_dim = max(int(width), int(height))
        params = {
            "gdino_checkpoint": self.config.gdino_checkpoint,
            "gdino_config_id": "auto",
            "text_queries": queries,
            "box_threshold": float(self.config.box_threshold),
            "text_threshold": float(self.config.text_threshold),
            "max_dets": int(self.config.max_dets),
            "device": self.config.device,
            "recursive_tile_scales": list(self.config.tile_scales),
        }
        if int(self.config.recursive_tile_batch_size or 0) > 0:
            params["recursive_tile_batch_size"] = int(self.config.recursive_tile_batch_size)
        if max_dim > int(self.config.tiled_threshold):
            result = self._dino_service.call(
                "recursive_detect",
                {
                    "image_path": str(image_path),
                    "params": params,
                    "target_labels": queries,
                    "max_depth": int(self.config.recursive_max_depth),
                    "min_box_px": int(self.config.min_box_px),
                },
            )
        else:
            result = self._dino_service.call("predict", {"image_path": str(image_path), "params": params})
        return list(result.get("detections") or [])

    def _ensure_yolo(self):
        if self._yolo_model is not None:
            return self._yolo_model, self._yolo_device
        from object_detection.yolo.lib import load_yolo_class, resolve_device

        yolo_cls = load_yolo_class()
        model_path = Path(self.config.yolo_model).expanduser().resolve()
        self._yolo_model = yolo_cls(str(model_path))
        self._yolo_device = resolve_device(self.config.device)
        self._yolo_names = _normalize_class_mapping(getattr(self._yolo_model, "names", None))
        if not self._yolo_names:
            self._yolo_names = {idx: name for idx, name in enumerate(_load_class_names_near(model_path))}
        if self._yolo_names:
            ordered = [self._yolo_names[idx] for idx in sorted(self._yolo_names)]
            self.log(f"YOLO classes={','.join(ordered)}")
        return self._yolo_model, self._yolo_device

    def _detect_yolo(self, image_path: Path) -> list[dict[str, Any]]:
        from PIL import Image

        model, device = self._ensure_yolo()
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if max(width, height) > int(self.config.tiled_threshold):
                return self._detect_yolo_tiled(image, image_path=image_path, model=model, device=device)
        results = model.predict(
            source=str(image_path),
            imgsz=1024,
            conf=float(self.config.yolo_conf),
            iou=float(self.config.yolo_iou),
            max_det=int(self.config.max_dets),
            device=device,
            save=False,
            verbose=False,
        )
        return self._parse_yolo_results(results)

    def _detect_yolo_tiled(self, image, *, image_path: Path, model, device: str) -> list[dict[str, Any]]:
        width, height = image.size
        tiles = _tile_boxes(width, height)
        self.log(
            f"  detect[yolo] tiled: {len(tiles)} patches "
            f"({TRAINED_DETECTOR_TILE_SIZE}px overlap={TRAINED_DETECTOR_TILE_OVERLAP})"
        )
        all_dets: list[dict[str, Any]] = []
        for x1, y1, x2, y2 in tiles:
            tile = image.crop((x1, y1, x2, y2))
            results = model.predict(
                source=tile,
                imgsz=TRAINED_DETECTOR_TILE_SIZE,
                conf=float(self.config.yolo_conf),
                iou=float(self.config.yolo_iou),
                max_det=int(self.config.max_dets),
                device=device,
                save=False,
                verbose=False,
            )
            for det in self._parse_yolo_results(results):
                bx1, by1, bx2, by2 = [float(v) for v in det["box"]]
                det["box"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                all_dets.append(det)
        if self.config.disable_tiled_nms:
            return all_dets[:int(self.config.max_dets)] if int(self.config.max_dets) > 0 else all_dets
        return _nms_detections(all_dets, iou_threshold=float(self.config.yolo_iou), max_dets=int(self.config.max_dets))

    def _parse_yolo_results(self, results) -> list[dict[str, Any]]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = _normalize_class_mapping(getattr(result, "names", None)) or self._yolo_names
        out: list[dict[str, Any]] = []
        for box, score, class_id in zip(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
        ):
            idx = int(class_id)
            out.append({"box": [float(v) for v in box], "score": float(score), "label": str(names.get(idx, idx))})
        return out

    def prepare_stabledino(self, image_paths: list[Path], *, names: list[str]) -> None:
        if "stabledino" not in self.active_models:
            return
        self._stabledino_cache = {}
        self._stabledino_ready = True
        paths = [Path(p) for p in image_paths if Path(p).is_file()]
        if not paths:
            return
        stable_names = self._resolve_stabledino_names(names)
        self.log(f"StableDINO prepass start: {len(paths)} images, classes={','.join(stable_names)}")
        try:
            self._stabledino_cache = self._run_stabledino_tiled_batch(paths, names=stable_names)
            total = sum(len(v) for v in self._stabledino_cache.values())
            self.log(f"StableDINO prepass done: {len(paths)} images, {total} boxes cached.")
        except Exception as exc:
            self.log(f"StableDINO prepass failed (StableDINO will be skipped): {exc}")
            self._stabledino_cache = {}

    def _run_stabledino_tiled_batch(self, image_paths: list[Path], *, names: list[str]) -> dict[str, list[dict[str, Any]]]:
        from PIL import Image
        import shutil

        output_root = Path(self.config.stabledino_output_dir or image_paths[0].parent / "stabledino_tmp")
        tile_root = output_root / "tiles"
        if tile_root.exists():
            shutil.rmtree(tile_root)
        tile_root.mkdir(parents=True, exist_ok=True)

        tile_paths: list[Path] = []
        tile_to_original: dict[str, tuple[str, int, int]] = {}
        for image_index, image_path in enumerate(image_paths, start=1):
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                if max(width, height) > int(self.config.tiled_threshold):
                    all_tiles: list[tuple[int, int, int, int]] = []
                    for tile_size in TRAINED_DETECTOR_TILE_SIZES:
                        overlap = min(TRAINED_DETECTOR_TILE_OVERLAP, max(0, tile_size // 6))
                        all_tiles.extend(_tile_boxes(width, height, tile_size=tile_size, overlap=overlap))
                    seen_tiles: set[tuple[int, int, int, int]] = set()
                    tiles = []
                    skipped_black = 0
                    for tile in all_tiles:
                        if tile in seen_tiles:
                            continue
                        seen_tiles.add(tile)
                        if _tile_has_too_much_black(image, tile):
                            skipped_black += 1
                            continue
                        tiles.append(tile)
                else:
                    tiles = [(0, 0, width, height)]
                    skipped_black = 0
                self.log(
                    f"  StableDINO patches[{image_path.name}]: {len(tiles)} "
                    f"(scales={','.join(str(s) for s in TRAINED_DETECTOR_TILE_SIZES)} "
                    f"overlap={TRAINED_DETECTOR_TILE_OVERLAP} skipped_black={skipped_black})"
                )
                for tile_index, (x1, y1, x2, y2) in enumerate(tiles, start=1):
                    tile_path = tile_root / f"img{image_index:05d}_tile{tile_index:05d}_{x1}_{y1}.png"
                    image.crop((x1, y1, x2, y2)).save(tile_path)
                    resolved_tile = str(tile_path.resolve())
                    tile_paths.append(tile_path)
                    tile_to_original[resolved_tile] = (str(image_path.resolve()), x1, y1)
        if not tile_paths:
            return {}

        tile_cache = self._run_stabledino_batch(tile_paths, names=names)
        merged: dict[str, list[dict[str, Any]]] = {}
        for tile_resolved, dets in tile_cache.items():
            item = tile_to_original.get(str(Path(tile_resolved).resolve()))
            if item is None:
                continue
            original_resolved, offset_x, offset_y = item
            for det in dets:
                box = det.get("box") or []
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                shifted = dict(det)
                shifted["box"] = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
                merged.setdefault(original_resolved, []).append(shifted)
        if self.config.disable_tiled_nms:
            max_dets = int(self.config.max_dets)
            return {
                resolved: (dets[:max_dets] if max_dets > 0 else dets)
                for resolved, dets in merged.items()
            }
        return {
            resolved: _nms_detections(dets, iou_threshold=TRAINED_DETECTOR_NMS_IOU, max_dets=int(self.config.max_dets))
            for resolved, dets in merged.items()
        }

    def _run_stabledino_batch(self, image_paths: list[Path], *, names: list[str]) -> dict[str, list[dict[str, Any]]]:
        from object_detection.semi_training.stable_dino.infer import main as stable_infer_main

        output_root = Path(self.config.stabledino_output_dir or image_paths[0].parent / "stabledino_tmp")
        output_dir = output_root / "batch"
        output_dir.mkdir(parents=True, exist_ok=True)
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        for stale in source_dir.iterdir():
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
        name_to_path: dict[str, str] = {}
        for path in image_paths:
            link = source_dir / path.name
            name_to_path[path.name] = str(path.resolve())
            if link.resolve() == path.resolve():
                continue
            try:
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(path.resolve())
            except OSError:
                import shutil

                shutil.copy2(path, link)
        argv = [
            "--checkpoint", str(Path(self.config.stabledino_checkpoint).expanduser().resolve()),
            "--source", str(source_dir),
            "--output-dir", str(output_dir),
            "--device", self.config.device,
            "--names", *names,
            "--no-previews",
        ]
        stable_infer_main(argv)
        return self._parse_stabledino_outputs(output_dir, name_to_path)

    def _parse_stabledino_outputs(self, output_dir: Path, name_to_path: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        pred_path = output_dir / "coco_instances_results.json"
        coco_path = output_dir / "image_only_coco" / "annotations" / "instances_val.json"
        if not pred_path.is_file() or not coco_path.is_file():
            return {}
        payload = json.loads(pred_path.read_text(encoding="utf-8"))
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        categories = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
        image_id_to_name = {int(img["id"]): str(img.get("file_name", "")) for img in coco.get("images", [])}
        cache: dict[str, list[dict[str, Any]]] = {}
        for item in payload if isinstance(payload, list) else []:
            score = float(item.get("score") or 0.0)
            if score < float(self.config.stabledino_conf):
                continue
            file_name = image_id_to_name.get(int(item.get("image_id", -1)), "")
            resolved = name_to_path.get(file_name)
            if resolved is None:
                continue
            x, y, w, h = [float(v) for v in item.get("bbox", [0, 0, 0, 0])]
            cache.setdefault(resolved, []).append({
                "box": [x, y, x + w, y + h],
                "score": score,
                "label": categories.get(int(item.get("category_id", -1)), ""),
            })
        for resolved, dets in cache.items():
            dets.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
            cache[resolved] = dets[: int(self.config.max_dets)]
        return cache

    def _detect_stabledino_single(self, image_path: Path, *, names: list[str]) -> list[dict[str, Any]]:
        from object_detection.semi_training.stable_dino.infer import main as stable_infer_main

        output_root = Path(self.config.stabledino_output_dir or image_path.parent / "stabledino_tmp")
        output_dir = output_root / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            "--checkpoint", str(Path(self.config.stabledino_checkpoint).expanduser().resolve()),
            "--source", str(image_path),
            "--output-dir", str(output_dir),
            "--device", self.config.device,
            "--names", *names,
            "--no-previews",
        ]
        stable_infer_main(argv)
        return self._parse_stabledino_outputs(output_dir, {image_path.name: str(image_path.resolve())}).get(
            str(image_path.resolve()), []
        )
