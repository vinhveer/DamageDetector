from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from device_utils import select_device_str, select_torch_device
from segmentation.sam.finetune.runtime import (
    load_inference_config,
    resolve_decoder_type,
    resolve_image_size,
    resolve_predict_threshold,
    resolve_refine_settings,
    resolve_tile_settings,
)
from segmentation.sam.finetune.test import _load_finetuned_sam
from segmentation.sam.finetune.tiled_inference import (
    binary_mask_from_score_map,
    coarse_refine_model_score_map,
)
from segmentation.unet.model_io import load_model_from_checkpoint, load_training_config_from_path
from segmentation.unet.predict_lib.inference import predict_probabilities
from segmentation.unet.predict_lib.postprocess import binarize_prediction, postprocess_binary_mask

from crack_test import render, sqlite_store


DEFAULT_IMAGE_IDS = [221, 226, 231, 234, 63, 48, 71, 5]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    rel_path: str
    path: Path
    width: int
    height: int


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_existing_path(path_text: str, input_dir: Path) -> Path:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(input_dir / path)
    candidates.append(input_dir / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found from SQLite path={path_text!r}")


def load_requested_images(source_db: Path, input_dir: Path, image_ids: list[int]) -> list[ImageRecord]:
    placeholders = ",".join("?" for _ in image_ids)
    with sqlite3.connect(str(source_db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT image_id, rel_path, path, width, height
            FROM images
            WHERE image_id IN ({placeholders})
            """,
            image_ids,
        ).fetchall()
    by_id = {int(row["image_id"]): row for row in rows}
    missing = [image_id for image_id in image_ids if image_id not in by_id]
    if missing:
        raise ValueError(f"Missing image_id(s) in source DB: {missing}")
    records: list[ImageRecord] = []
    for image_id in image_ids:
        row = by_id[image_id]
        path = _resolve_existing_path(str(row["path"] or row["rel_path"]), input_dir)
        records.append(
            ImageRecord(
                image_id=image_id,
                rel_path=str(row["rel_path"]),
                path=path,
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
    return records


def _nms(boxes: list[dict], iou_threshold: float) -> list[dict]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda item: float(item["score"]), reverse=True)
    kept: list[dict] = []
    for box in ordered:
        if any(_iou(box["xyxy"], prev["xyxy"]) > iou_threshold for prev in kept if prev["label"] == box["label"]):
            continue
        kept.append(box)
    return kept


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / max(area_a + area_b - inter, 1e-6)


def _clip_boxes(boxes: list[dict], width: int, height: int) -> list[dict]:
    clipped = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box["xyxy"]]
        x1 = max(0.0, min(float(width - 1), x1))
        x2 = max(0.0, min(float(width - 1), x2))
        y1 = max(0.0, min(float(height - 1), y1))
        y2 = max(0.0, min(float(height - 1), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        item = dict(box)
        item["xyxy"] = [x1, y1, x2, y2]
        clipped.append(item)
    return clipped


def _limit_boxes(boxes: list[dict], max_boxes: int) -> list[dict]:
    limit = int(max_boxes)
    if limit <= 0 or len(boxes) <= limit:
        return boxes
    return sorted(boxes, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:limit]


def _filter_boxes_by_labels(boxes: list[dict], labels: set[str]) -> list[dict]:
    if not labels:
        return boxes
    wanted = {label.strip().lower() for label in labels if label.strip()}
    return [box for box in boxes if str(box.get("label", "")).strip().lower() in wanted]


class StableDinoDetector:
    def __init__(self, model_root: Path, device: str, class_names: list[str]) -> None:
        self.model_root = model_root
        self.device = device
        self.class_names = class_names
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        module = _load_module_from_path("crack_stable_dino_tiled", REPO_ROOT / "tools" / "stable_dino_tiled_infer.py")
        import pickle
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate

        stable_dino_root = REPO_ROOT / "object_detection" / "stable_dino"
        if str(stable_dino_root) not in sys.path:
            sys.path.insert(0, str(stable_dino_root))

        config_candidates = [self.model_root / "inference" / "config.yaml.pkl", self.model_root / "config.yaml.pkl"]
        model_candidates = [self.model_root / "model" / "model_best.pth", self.model_root / "model_best.pth"]
        config_path = next((path for path in config_candidates if path.is_file()), None)
        model_path = next((path for path in model_candidates if path.is_file()), None)
        if config_path is None:
            raise FileNotFoundError(f"StableDINO config not found under: {self.model_root}")
        if model_path is None:
            raise FileNotFoundError(f"StableDINO weights not found under: {self.model_root}")
        with open(config_path, "rb") as f:
            cfg = pickle.load(f)
        cfg.model.device = self.device
        model = instantiate(cfg.model)
        model.to(self.device)
        model.eval()
        DetectionCheckpointer(model).load(str(model_path))
        self._model = (module, model)
        return self._model

    def predict(self, image_path: Path, *, tile_size: int, overlap: int, conf: float, nms_iou: float) -> list[dict]:
        module, model = self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")
        w_img, h_img = image.size
        stride = tile_size - overlap
        tiles_x = max(1, int(np.ceil((w_img - overlap) / stride)))
        tiles_y = max(1, int(np.ceil((h_img - overlap) / stride)))
        image_arr = np.array(image)
        all_boxes: list[dict] = []
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x1 = min(tx * stride, max(0, w_img - tile_size))
                y1 = min(ty * stride, max(0, h_img - tile_size))
                x2 = min(w_img, x1 + tile_size)
                y2 = min(h_img, y1 + tile_size)
                tile_arr = image_arr[y1:y2, x1:x2]
                raw = module._predict_tile(model, tile_arr, self.device)
                for det in raw:
                    if float(det["score"]) < conf:
                        continue
                    bx1, by1, bx2, by2 = [float(v) for v in det["xyxy"]]
                    label_idx = int(det["label"])
                    all_boxes.append(
                        {
                            "label": self.class_names[label_idx] if label_idx < len(self.class_names) else str(label_idx),
                            "score": float(det["score"]),
                            "xyxy": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                        }
                    )
        return _clip_boxes(_nms(all_boxes, nms_iou), w_img, h_img)


class YoloDetector:
    def __init__(self, model_path: Path, device: str) -> None:
        from object_detection.yolo.lib import load_yolo_class, resolve_device

        YOLO = load_yolo_class()
        self.model = YOLO(str(model_path))
        self.device = resolve_device(device)

    def predict(self, image_path: Path, *, tile_size: int, overlap: int, conf: float, nms_iou: float) -> list[dict]:
        image = Image.open(image_path).convert("RGB")
        w_img, h_img = image.size
        stride = tile_size - overlap
        tiles_x = max(1, int(np.ceil((w_img - overlap) / stride)))
        tiles_y = max(1, int(np.ceil((h_img - overlap) / stride)))
        all_boxes: list[dict] = []
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x1 = min(tx * stride, max(0, w_img - tile_size))
                y1 = min(ty * stride, max(0, h_img - tile_size))
                x2 = min(w_img, x1 + tile_size)
                y2 = min(h_img, y1 + tile_size)
                tile = image.crop((x1, y1, x2, y2))
                results = self.model.predict(source=tile, imgsz=tile_size, conf=conf, device=self.device, verbose=False)
                for result in results:
                    result_boxes = getattr(result, "boxes", None)
                    if result_boxes is None or getattr(result_boxes, "xyxy", None) is None:
                        continue
                    names = getattr(result, "names", {}) or {}
                    for box, score, cls in zip(
                        result_boxes.xyxy.detach().cpu().tolist(),
                        result_boxes.conf.detach().cpu().tolist(),
                        result_boxes.cls.detach().cpu().tolist(),
                    ):
                        all_boxes.append(
                            {
                                "label": str(names.get(int(cls), "crack")),
                                "score": float(score),
                                "xyxy": [float(box[0]) + x1, float(box[1]) + y1, float(box[2]) + x1, float(box[3]) + y1],
                            }
                        )
        return _clip_boxes(_nms(all_boxes, nms_iou), w_img, h_img)


class SamFineTuneSegmenter:
    def __init__(self, model_dir: Path, device_name: str) -> None:
        self.model_dir = model_dir
        self.device = select_torch_device(device_name)
        self.sam_ckpt = model_dir / "sam_vit_b_01ec64.pth"
        self.coarse_delta = model_dir / "coarse_best_model.pth"
        self.refine_delta = model_dir / "refine_best_model.pth"
        self.coarse_model = None
        self.refine_model = None
        self.settings: dict | None = None

    def _ensure_loaded(self) -> None:
        if self.coarse_model is not None:
            return
        for path in [self.sam_ckpt, self.coarse_delta, self.refine_delta]:
            if not path.is_file():
                raise FileNotFoundError(f"SAM fine-tune file not found: {path}")
        coarse_cfg = load_inference_config(str(self.coarse_delta))
        refine_cfg = load_inference_config(str(self.refine_delta))
        vit_name = str(coarse_cfg.get("vit_name", "vit_b"))
        delta_type = str(coarse_cfg.get("delta_type", "lora"))
        rank = int(coarse_cfg.get("rank", 4))
        middle_dim = int(coarse_cfg.get("middle_dim", 32))
        scaling_factor = float(coarse_cfg.get("scaling_factor", 0.1))
        coarse_img_size = resolve_image_size(str(self.coarse_delta), None)
        refine_img_size = resolve_image_size(str(self.refine_delta), None)
        coarse_decoder = resolve_decoder_type(str(self.coarse_delta), str(coarse_cfg.get("decoder_type", "auto")))
        refine_decoder = resolve_decoder_type(str(self.refine_delta), str(refine_cfg.get("decoder_type", "auto")))
        coarse_model, coarse_img_size, _ = _load_finetuned_sam(
            ckpt=str(self.sam_ckpt),
            vit_name=vit_name,
            img_size=coarse_img_size,
            delta_type=delta_type,
            delta_ckpt=str(self.coarse_delta),
            middle_dim=middle_dim,
            scaling_factor=scaling_factor,
            rank=rank,
            decoder_type=coarse_decoder,
            centerline_head=bool(coarse_cfg.get("centerline_head", False)),
            device=self.device,
        )
        refine_model, refine_img_size, _ = _load_finetuned_sam(
            ckpt=str(self.sam_ckpt),
            vit_name=vit_name,
            img_size=refine_img_size,
            delta_type=delta_type,
            delta_ckpt=str(self.refine_delta),
            middle_dim=middle_dim,
            scaling_factor=scaling_factor,
            rank=rank,
            decoder_type=refine_decoder,
            centerline_head=bool(refine_cfg.get("centerline_head", False)),
            device=self.device,
        )
        coarse_tile_size, coarse_tile_overlap = resolve_tile_settings(str(self.coarse_delta), None, None)
        refine_settings = resolve_refine_settings(str(self.coarse_delta))
        self.coarse_model = coarse_model
        self.refine_model = refine_model
        self.settings = {
            "coarse_img_size": int(coarse_img_size),
            "coarse_tile_size": int(coarse_tile_size),
            "coarse_tile_overlap": int(coarse_tile_overlap),
            "refine_img_size": int(refine_img_size),
            "refine_settings": refine_settings,
            "threshold": float(resolve_predict_threshold(str(self.refine_delta), "auto")),
            "tile_batch_size": int(coarse_cfg.get("tile_batch_size", 1)),
            "refine_batch_size": int(refine_cfg.get("refine_batch_size", 1)),
        }

    def segment_boxes(self, image_path: Path, boxes: list[dict]) -> tuple[np.ndarray, list[dict]]:
        self._ensure_loaded()
        assert self.coarse_model is not None and self.refine_model is not None and self.settings is not None
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = rgb.shape[:2]
        full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        rows: list[dict] = []
        refine_settings = self.settings["refine_settings"]
        for det_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
            x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
            y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
            if x2 <= x1 or y2 <= y1:
                rows.append({"det_idx": det_idx, "mask_area_px": 0, "score": None, "extra_json": {"skipped": "invalid_box"}})
                continue
            crop = rgb[y1:y2, x1:x2]
            score_map, _coarse, refine_outputs = coarse_refine_model_score_map(
                crop,
                coarse_model=self.coarse_model,
                coarse_image_size=self.settings["coarse_img_size"],
                coarse_tile_size=self.settings["coarse_tile_size"],
                coarse_tile_overlap=self.settings["coarse_tile_overlap"],
                refine_model=self.refine_model,
                refine_image_size=self.settings["refine_img_size"],
                refine_tile_size=int(refine_settings["refine_tile_size"]),
                refine_tile_sizes=refine_settings.get("refine_tile_sizes"),
                refine_max_rois=int(refine_settings["refine_max_rois"]),
                refine_roi_padding=int(refine_settings["refine_roi_padding"]),
                refine_merge_mode=str(refine_settings["refine_merge_mode"]),
                refine_score_threshold=float(refine_settings["refine_score_threshold"]),
                positive_band_low=float(refine_settings["positive_band_low"]),
                positive_band_high=float(refine_settings["positive_band_high"]),
                threshold=self.settings["threshold"],
                multimask_output=False,
                use_amp=False,
                tile_batch_size=int(self.settings["tile_batch_size"]),
                refine_batch_size=int(self.settings["refine_batch_size"]),
            )
            crop_mask = binary_mask_from_score_map(score_map, self.settings["threshold"]).astype(np.uint8)
            if crop_mask.shape != (y2 - y1, x2 - x1):
                crop_mask = cv2.resize(crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], crop_mask)
            rows.append(
                {
                    "det_idx": det_idx,
                    "mask_area_px": int(crop_mask.sum()),
                    "score": float(np.mean(score_map)) if score_map.size else None,
                    "extra_json": {"threshold": self.settings["threshold"], "refine_rois": len(refine_outputs)},
                }
            )
        return full_mask, rows


class UnetSegmenter:
    def __init__(self, model_path: Path, device_name: str) -> None:
        self.model_path = model_path
        self.device = select_torch_device(device_name)
        self.model, _ = load_model_from_checkpoint(str(model_path), self.device)
        train_config = load_training_config_from_path(str(model_path)) or {}
        args = train_config.get("args") or {}
        self.input_size = int(args.get("input_size", 512))
        self.threshold = float(args.get("metric_threshold", 0.5))
        self.tile_overlap = self.input_size // 2

    def segment_boxes(self, image_path: Path, boxes: list[dict]) -> tuple[np.ndarray, list[dict]]:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = rgb.shape[:2]
        full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        rows: list[dict] = []
        for det_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
            x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
            y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
            if x2 <= x1 or y2 <= y1:
                rows.append({"det_idx": det_idx, "mask_area_px": 0, "score": None, "extra_json": {"skipped": "invalid_box"}})
                continue
            crop = Image.fromarray(rgb[y1:y2, x1:x2])
            prob = predict_probabilities(
                self.model,
                crop,
                self.device,
                mode="tile",
                input_size=self.input_size,
                tile_overlap=self.tile_overlap,
                tile_batch_size=4,
                gaussian_weight=True,
            )
            crop_mask = binarize_prediction(prob, self.threshold)
            crop_mask = postprocess_binary_mask(crop_mask, True, min_size=20).astype(np.uint8)
            if crop_mask.shape != (y2 - y1, x2 - x1):
                crop_mask = cv2.resize(crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], crop_mask)
            rows.append(
                {
                    "det_idx": det_idx,
                    "mask_area_px": int(crop_mask.sum()),
                    "score": float(np.mean(prob)) if prob.size else None,
                    "extra_json": {"threshold": self.threshold, "input_size": self.input_size},
                }
            )
        return full_mask, rows


def _detection_rows(run_id: str, pipeline_name: str, image_id: int, detector_name: str, boxes: list[dict]) -> list[dict]:
    rows = []
    for det_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box["xyxy"]]
        rows.append(
            {
                "run_id": run_id,
                "pipeline_name": pipeline_name,
                "image_id": image_id,
                "det_idx": det_idx,
                "detector_name": detector_name,
                "label": str(box.get("label") or "crack"),
                "score": float(box.get("score") or 0.0),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return rows


def _segmentation_rows(run_id: str, pipeline_name: str, image_id: int, segmenter_name: str, rows: list[dict], mask_path: str) -> list[dict]:
    return [
        {
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "image_id": image_id,
            "det_idx": int(row["det_idx"]),
            "segmenter_name": segmenter_name,
            "mask_path": mask_path,
            "mask_area_px": int(row["mask_area_px"]),
            "score": row.get("score"),
            "extra_json": row.get("extra_json", {}),
        }
        for row in rows
    ]


def _artifact_rows(run_id: str, pipeline_name: str, image_id: int, artifacts: dict[str, str], mask_path: str) -> list[dict]:
    rows = [
        {"run_id": run_id, "pipeline_name": pipeline_name, "image_id": image_id, "artifact_type": key, "path": value}
        for key, value in artifacts.items()
    ]
    rows.append({"run_id": run_id, "pipeline_name": pipeline_name, "image_id": image_id, "artifact_type": "mask", "path": mask_path})
    return rows


def _write_summary(output_root: Path, run_id: str, records: list[ImageRecord], results: dict[str, dict[int, dict]]) -> None:
    rows = []
    for rec in records:
        for pipeline_name, by_image in results.items():
            item = by_image.get(rec.image_id, {})
            rows.append(
                {
                    "run_id": run_id,
                    "image_id": rec.image_id,
                    "image": rec.rel_path,
                    "pipeline": pipeline_name,
                    "detections": int(item.get("detections", 0)),
                    "segmentation_masks": int(item.get("segmentations", 0)),
                    "mask_area_px": int(item.get("mask_area_px", 0)),
                    "output_dir": str(item.get("output_dir", "")),
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(output_root / "summary.csv", "w", encoding="utf-8") as f:
        f.write("run_id,image_id,image,pipeline,detections,segmentation_masks,mask_area_px,output_dir\n")
        for row in rows:
            values = [
                row["run_id"],
                str(row["image_id"]),
                row["image"],
                row["pipeline"],
                str(row["detections"]),
                str(row["segmentation_masks"]),
                str(row["mask_area_px"]),
                row["output_dir"],
            ]
            f.write(",".join('"' + value.replace('"', '""') + '"' for value in values) + "\n")


def _sam_segmenter_name(model_dir: Path) -> str:
    config = load_inference_config(str(model_dir / "coarse_best_model.pth"))
    decoder = str(config.get("decoder_type", "baseline")).strip().lower() or "baseline"
    return "sam_finetune_hq" if decoder == "hq" else "sam_finetune_lora"


def run(args: argparse.Namespace) -> int:
    source_db = Path(args.source_db).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    db_path = output_root / "crack_test.sqlite3"
    image_ids = [int(v) for v in args.image_ids]
    if args.limit is not None:
        image_ids = image_ids[: max(0, int(args.limit))]
    records = load_requested_images(source_db, input_dir, image_ids)
    device_name = select_device_str(args.device)
    run_id = args.run_id or f"crack_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    stable_dino_dir = Path(args.stable_dino_dir).expanduser().resolve()
    yolo_model = Path(args.yolo_model).expanduser().resolve()
    sam_model_dir = Path(args.sam_model_dir).expanduser().resolve()
    unet_model = Path(args.unet_model).expanduser().resolve()

    conn = sqlite_store.connect(db_path)
    sqlite_store.insert_run(
        conn,
        {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_db": str(source_db),
            "input_dir": str(input_dir),
            "output_root": str(output_root),
            "device": device_name,
            "stable_dino_dir": str(stable_dino_dir),
            "yolo_model": str(yolo_model),
            "sam_model_dir": str(sam_model_dir),
            "unet_model": str(unet_model),
            "conf_threshold": float(args.conf),
            "nms_iou": float(args.nms_iou),
            "image_ids_json": image_ids,
        },
    )
    sqlite_store.insert_images(
        conn,
        [
            {
                "run_id": run_id,
                "image_id": rec.image_id,
                "rel_path": rec.rel_path,
                "image_path": str(rec.path),
                "width": rec.width,
                "height": rec.height,
            }
            for rec in records
        ],
    )

    pipelines = []
    selected = {str(item).strip().lower() for item in args.pipelines}
    if "stable_dino_sam_finetune" in selected or "stable" in selected or "all" in selected:
        pipelines.append(
            (
                "stable_dino_sam_finetune",
                "stable_dino",
                StableDinoDetector(stable_dino_dir, device_name, list(args.class_names)),
                _sam_segmenter_name(sam_model_dir),
                SamFineTuneSegmenter(sam_model_dir, device_name),
            )
        )
    if "yolo_unet" in selected or "yolo" in selected or "all" in selected:
        pipelines.append(("yolo_unet", "yolo", YoloDetector(yolo_model, device_name), "unet", UnetSegmenter(unet_model, device_name)))
    if not pipelines:
        raise ValueError(f"No pipeline selected from: {args.pipelines}")

    summary_results: dict[str, dict[int, dict]] = {pipeline[0]: {} for pipeline in pipelines}
    for rec in records:
        print(f"Image {rec.image_id}: {rec.rel_path}", flush=True)
        for pipeline_name, detector_name, detector, segmenter_name, segmenter in pipelines:
            print(f"  {pipeline_name}: detecting...", flush=True)
            boxes = detector.predict(
                rec.path,
                tile_size=int(args.tile_size),
                overlap=int(args.tile_overlap),
                conf=float(args.conf),
                nms_iou=float(args.nms_iou),
            )
            boxes = _filter_boxes_by_labels(boxes, set(args.keep_labels))
            boxes = _limit_boxes(boxes, int(args.max_boxes_per_image))
            sqlite_store.insert_detections(conn, _detection_rows(run_id, pipeline_name, rec.image_id, detector_name, boxes))
            print(f"  {pipeline_name}: {len(boxes)} boxes, segmenting...", flush=True)
            mask, seg_rows = segmenter.segment_boxes(rec.path, boxes)
            image_out_dir = output_root / pipeline_name / Path(rec.rel_path).stem
            mask_path = render.save_mask(mask, image_out_dir / "mask.png")
            artifacts = render.save_four_overlays(rec.path, boxes, mask, image_out_dir)
            sqlite_store.insert_segmentations(conn, _segmentation_rows(run_id, pipeline_name, rec.image_id, segmenter_name, seg_rows, mask_path))
            sqlite_store.insert_artifacts(conn, _artifact_rows(run_id, pipeline_name, rec.image_id, artifacts, mask_path))
            summary_results[pipeline_name][rec.image_id] = {
                "detections": len(boxes),
                "segmentations": len(seg_rows),
                "mask_area_px": int(np.asarray(mask).sum()),
                "output_dir": str(image_out_dir),
            }
            print(f"  {pipeline_name}: saved {image_out_dir}", flush=True)
    _write_summary(output_root, run_id, records, summary_results)
    conn.close()
    print(f"Saved SQLite: {db_path}", flush=True)
    print(f"Run ID: {run_id}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run crack object detection + box-guided segmentation on selected ROI images.")
    parser.add_argument("--source-db", default=str(WORKSPACE_ROOT / "data" / "HinhAnh" / "damage_scan.sqlite3"))
    parser.add_argument("--input-dir", default=str(WORKSPACE_ROOT / "data" / "HinhAnh"))
    parser.add_argument("--output-root", default=str(WORKSPACE_ROOT / "model_with_inference" / "crack_test"))
    parser.add_argument("--stable-dino-dir", default=str(WORKSPACE_ROOT / "model_with_inference" / "crack_object_detection" / "stable_dino_r50_img768"))
    parser.add_argument("--yolo-model", default=str(WORKSPACE_ROOT / "model_with_inference" / "crack_object_detection" / "yolo_26x_img768" / "model" / "best.pt"))
    parser.add_argument("--sam-model-dir", default=str(WORKSPACE_ROOT / "model_with_inference" / "crack_segmentation" / "sam_lora_hq_coarse_refine" / "model"))
    parser.add_argument("--unet-model", default=str(WORKSPACE_ROOT / "model_with_inference" / "crack_segmentation" / "unet_efficientnet_b4" / "model" / "best_model.pth"))
    parser.add_argument("--image-ids", nargs="+", type=int, default=DEFAULT_IMAGE_IDS)
    parser.add_argument("--pipelines", nargs="+", default=["all"], choices=["all", "stable", "yolo", "stable_dino_sam_finetune", "yolo_unet"])
    parser.add_argument("--class-names", nargs="+", default=["crack"])
    parser.add_argument("--keep-labels", nargs="+", default=["crack"])
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--max-boxes-per-image", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
