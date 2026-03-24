from __future__ import annotations

import inspect
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

from torch_runtime import describe_device_fallback, get_torch, select_device_str


@dataclass(frozen=True)
class DinoParams:
    gdino_checkpoint: str
    gdino_config_id: str = "auto"
    text_queries: Sequence[str] = ("crack",)
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    max_dets: int = 20
    device: str = "auto"
    output_dir: str = "results_dino"
    roi_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class Det:
    label: str
    box_xyxy: Any
    score: float


def normalize_queries(text_queries: Sequence[str]) -> List[str]:
    queries = [q.strip() for q in text_queries if q.strip()]
    seen = set()
    deduped: List[str] = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped


def label_matches(label: str, targets: Sequence[str]) -> bool:
    lowered = str(label or "").lower()
    for target in targets:
        token = str(target or "").strip().lower()
        if token and token in lowered:
            return True
    return False


def post_process_gdino(
    processor: Any,
    outputs: Any,
    input_ids: Any,
    box_threshold: float,
    text_threshold: float,
    target_sizes: Any,
) -> Any:
    fn = getattr(processor, "post_process_grounded_object_detection", None)
    if fn is None:
        raise RuntimeError(
            "Processor does not expose post_process_grounded_object_detection; "
            "use a compatible transformers version/model bundle."
        )
    signature = inspect.signature(fn)
    kwargs = {
        "outputs": outputs,
        "input_ids": input_ids,
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in signature.parameters:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
    return fn(**kwargs)


def run_text_boxes(
    *,
    processor: Any,
    gdino: Any,
    device: str,
    pil_image: Any,
    text_queries: Sequence[str],
    box_threshold: float,
    text_threshold: float,
) -> List[Det]:
    import numpy as np

    torch = get_torch()

    queries = normalize_queries(text_queries)
    if not queries:
        return []
    width, height = pil_image.size
    caption = ". ".join(queries)
    with torch.no_grad():
        inputs = processor(images=pil_image, text=caption, return_tensors="pt").to(device)
        outputs = gdino(**inputs)
        target_sizes = torch.tensor([[height, width]], device=device)
        processed = post_process_gdino(
            processor=processor,
            outputs=outputs,
            input_ids=inputs["input_ids"],
            target_sizes=target_sizes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    if not processed:
        return []
    p0 = processed[0]
    boxes = p0["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = p0["scores"].detach().cpu().numpy().astype(np.float32)
    labels = p0.get("labels", [])
    return [Det(label=str(label), box_xyxy=box.astype(np.float32), score=float(score)) for box, score, label in zip(boxes, scores, labels)]


def _box_iou(a: Any, b: Any) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _nms_boxes(boxes_info: list[tuple[Any, str, float]], iou_threshold: float = 0.5) -> list[tuple[Any, str, float]]:
    if not boxes_info:
        return []
    boxes_info = sorted(boxes_info, key=lambda x: x[2], reverse=True)
    used = [False] * len(boxes_info)
    kept = []
    for i, (box_i, label_i, score_i) in enumerate(boxes_info):
        if used[i]:
            continue
        kept.append((box_i, label_i, score_i))
        for j in range(i + 1, len(boxes_info)):
            if used[j]:
                continue
            if _box_iou(box_i, boxes_info[j][0]) > iou_threshold:
                used[j] = True
    return kept


def _filter_parent_boxes(boxes_info: list[tuple[Any, str, float]], contain_thresh: float = 0.7) -> list[tuple[Any, str, float]]:
    import numpy as np

    count = len(boxes_info)
    if count <= 1:
        return boxes_info

    def _area(box: Any) -> float:
        return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))

    def _intersection(a: Any, b: Any) -> float:
        ix1 = max(float(a[0]), float(b[0]))
        iy1 = max(float(a[1]), float(b[1]))
        ix2 = min(float(a[2]), float(b[2]))
        iy2 = min(float(a[3]), float(b[3]))
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    is_parent = [False] * count
    for i in range(count):
        box_i, _, _ = boxes_info[i]
        area_i = _area(box_i)
        if area_i <= 0:
            continue
        for j in range(count):
            if i == j or is_parent[i]:
                break
            box_j, _, _ = boxes_info[j]
            area_j = _area(box_j)
            if area_j <= 0 or area_j >= area_i:
                continue
            inter = _intersection(box_i, box_j)
            if area_j > 0 and inter / area_j >= contain_thresh:
                is_parent[i] = True
    return [boxes_info[i] for i in range(count) if not is_parent[i]]


def _recursive_zoom_detect(
    rgb: Any,
    crop_box: tuple[int, int, int, int],
    gdino_items: tuple[Any, Any, str],
    text_queries: Sequence[str],
    box_threshold: float,
    text_threshold: float,
    current_depth: int,
    max_depth: int,
    *,
    min_box_px: int = 48,
    stop_checker=None,
    log_fn=None,
) -> list[tuple[Any, str, float]]:
    import numpy as np
    from PIL import Image

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    width = crop_x2 - crop_x1
    height = crop_y2 - crop_y1
    if width < min_box_px or height < min_box_px:
        return []
    if stop_checker is not None and stop_checker():
        raise RuntimeError("Stopped")
    processor, gdino, device = gdino_items
    crop_rgb = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    pil_crop = Image.fromarray(crop_rgb)
    detections = run_text_boxes(
        processor=processor,
        gdino=gdino,
        device=device,
        pil_image=pil_crop,
        text_queries=text_queries,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if not detections:
        return []
    detections_full: list[tuple[Any, str, float]] = []
    for det in detections:
        rel_x1, rel_y1, rel_x2, rel_y2 = det.box_xyxy.tolist()
        mapped = np.array([rel_x1 + crop_x1, rel_y1 + crop_y1, rel_x2 + crop_x1, rel_y2 + crop_y1], dtype=np.float32)
        detections_full.append((mapped, det.label, float(det.score)))
    if log_fn is not None and detections_full:
        log_fn(f"  [depth {current_depth}] crop ({crop_x1},{crop_y1})-({crop_x2},{crop_y2}) -> {len(detections_full)} dets")
    if current_depth >= max_depth:
        return detections_full
    result: list[tuple[Any, str, float]] = []
    for box_full, label, score in detections_full:
        child_x1, child_y1, child_x2, child_y2 = (int(v) for v in box_full.tolist())
        children = _recursive_zoom_detect(
            rgb,
            (child_x1, child_y1, child_x2, child_y2),
            gdino_items,
            text_queries,
            box_threshold,
            text_threshold,
            current_depth + 1,
            max_depth,
            min_box_px=min_box_px,
            stop_checker=stop_checker,
            log_fn=log_fn,
        )
        if children:
            result.extend(children)
        else:
            result.append((box_full, label, score))
    return result


class DinoRunner:
    def __init__(self) -> None:
        self._device: str | None = None
        self._gdino_checkpoint: str | None = None
        self._gdino_config_id: str | None = None
        self._processor: Any | None = None
        self._gdino: Any | None = None

    def _ensure_import_paths(self) -> None:
        here = Path(__file__).resolve()
        repo_root = here.parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _load_gdino_state_dict(self, checkpoint_path: str) -> dict:
        torch = get_torch()

        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(checkpoint_path)
        try:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            if isinstance(raw.get("state_dict"), dict):
                state = raw["state_dict"]
            elif isinstance(raw.get("model"), dict):
                state = raw["model"]
            else:
                state = raw
        else:
            raise TypeError(f"Unsupported GroundingDINO checkpoint format: {type(raw)} ({checkpoint_path})")
        if not isinstance(state, dict):
            raise TypeError(f"GroundingDINO checkpoint state_dict invalid: {type(state)} ({checkpoint_path})")
        state = self._strip_prefix_if_present(state, "module.")
        state = self._strip_prefix_if_present(state, "model.")
        return state

    def _strip_prefix_if_present(self, state: dict, prefix: str) -> dict:
        keys = list(state.keys())
        if not keys:
            return state
        count = sum(1 for key in keys if str(key).startswith(prefix))
        if count >= int(len(keys) * 0.9):
            return {str(key)[len(prefix) :] if str(key).startswith(prefix) else key: value for key, value in state.items()}
        return state

    def _resolve_gdino_config_id(self, params: DinoParams) -> str:
        config_id = str(params.gdino_config_id or "").strip()
        if not config_id or config_id.lower() == "auto":
            name = os.path.basename(str(params.gdino_checkpoint or "")).lower()
            if "swint" in name or "tiny" in name:
                return "IDEA-Research/grounding-dino-tiny"
            return "IDEA-Research/grounding-dino-base"
        return config_id

    def ensure_model_loaded(self, params: DinoParams, *, log_fn=None) -> tuple[Any, Any, str]:
        self._ensure_import_paths()
        import transformers
        import transformers.models.grounding_dino.image_processing_grounding_dino
        from transformers import AutoProcessor, GroundingDinoConfig, GroundingDinoForObjectDetection
        import faulthandler

        def _dump_on_hang(label: str, seconds: float) -> threading.Event:
            stop = threading.Event()

            def _arm() -> None:
                if stop.wait(seconds):
                    return
                try:
                    if log_fn is not None:
                        log_fn(f"WARN: '{label}' still running after {int(seconds)}s. Dumping stack traces...")
                except Exception:
                    pass
                try:
                    faulthandler.dump_traceback(all_threads=True)
                except Exception:
                    pass

            threading.Thread(target=_arm, name=f"hang-dump:{label}", daemon=True).start()
            return stop

        device = select_device_str(params.device)
        fallback = describe_device_fallback(params.device, device)
        if fallback is not None and log_fn is not None:
            log_fn(fallback)

        checkpoint_path = str(params.gdino_checkpoint or "").strip()
        if not checkpoint_path:
            raise FileNotFoundError("GroundingDINO checkpoint path is required.")

        checkpoint_lower = checkpoint_path.lower()
        checkpoint_is_dir = os.path.isdir(checkpoint_path)
        checkpoint_is_file = os.path.isfile(checkpoint_path)
        checkpoint_is_explicit_file = checkpoint_lower.endswith((".pth", ".pt", ".safetensors", ".bin"))
        use_hf_id = (not checkpoint_is_dir) and (not checkpoint_is_file) and (not checkpoint_is_explicit_file)

        if checkpoint_is_file and not use_hf_id:
            parent_dir = os.path.dirname(checkpoint_path)
            if os.path.isfile(os.path.join(parent_dir, "config.json")):
                base = os.path.basename(checkpoint_path).lower()
                if base in {"model.safetensors", "pytorch_model.bin", "tf_model.h5"}:
                    if log_fn is not None:
                        log_fn(f"Detected valid HF model folder ({parent_dir}). Using folder mode instead of single file.")
                    checkpoint_path = parent_dir
                    checkpoint_is_dir = True
                    checkpoint_is_file = False
                    use_hf_id = False

        if use_hf_id or checkpoint_is_dir:
            config_id = checkpoint_path
        else:
            parent_dir = os.path.dirname(checkpoint_path)
            if os.path.isfile(os.path.join(parent_dir, "config.json")):
                config_id = parent_dir
            else:
                config_id = self._resolve_gdino_config_id(params)

        needs_reload = (
            self._gdino is None
            or self._processor is None
            or self._gdino_checkpoint != checkpoint_path
            or self._gdino_config_id != config_id
            or self._device != device
        )
        if not needs_reload:
            return self._processor, self._gdino, device

        if log_fn is not None:
            if checkpoint_is_dir:
                log_fn("Loading GroundingDINO from local folder...")
            elif use_hf_id:
                log_fn("Loading GroundingDINO (offline/cache)...")
            else:
                log_fn("Loading GroundingDINO (offline config/cache + local .pth weights)...")

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        tick_stop = threading.Event()

        def _tick() -> None:
            if log_fn is None:
                return
            start = time.time()
            while not tick_stop.wait(8.0):
                elapsed = int(time.time() - start)
                log_fn(f"Still loading GroundingDINO... ({elapsed}s)")

        tick_thread = None
        if log_fn is not None:
            tick_thread = threading.Thread(target=_tick, name="gdino-load-tick", daemon=True)
            tick_thread.start()

        try:
            if log_fn is not None:
                log_fn(f"GroundingDINO: device={device}")
                log_fn(f"GroundingDINO: checkpoint={checkpoint_path}")
                log_fn(f"GroundingDINO: config={config_id}")
                if checkpoint_is_file:
                    try:
                        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                        log_fn(f"GroundingDINO: weights_size={size_mb:.1f} MB")
                    except Exception:
                        pass
            if checkpoint_is_dir:
                if log_fn is not None:
                    log_fn("Step: load processor (AutoProcessor.from_pretrained folder)...")
                start = time.time()
                hang = _dump_on_hang("AutoProcessor.from_pretrained(folder)", 30.0)
                processor = AutoProcessor.from_pretrained(checkpoint_path, local_files_only=True)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded processor ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: load model weights (from_pretrained folder)...")
                start = time.time()
                use_safetensors = os.path.isfile(os.path.join(checkpoint_path, "model.safetensors"))
                hang = _dump_on_hang("GroundingDinoForObjectDetection.from_pretrained(folder)", 60.0)
                try:
                    gdino = GroundingDinoForObjectDetection.from_pretrained(
                        checkpoint_path,
                        local_files_only=True,
                        use_safetensors=use_safetensors,
                    )
                except TypeError:
                    gdino = GroundingDinoForObjectDetection.from_pretrained(checkpoint_path, local_files_only=True)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded model from folder ({time.time() - start:.1f}s)")
            elif use_hf_id:
                if log_fn is not None:
                    log_fn("Step: load processor (AutoProcessor.from_pretrained cache)...")
                start = time.time()
                hang = _dump_on_hang("AutoProcessor.from_pretrained(cache)", 30.0)
                processor = AutoProcessor.from_pretrained(checkpoint_path, local_files_only=True)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded processor ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: load model weights (from_pretrained cache)...")
                start = time.time()
                hang = _dump_on_hang("GroundingDinoForObjectDetection.from_pretrained(cache)", 60.0)
                try:
                    gdino = GroundingDinoForObjectDetection.from_pretrained(checkpoint_path, local_files_only=True)
                except TypeError:
                    gdino = GroundingDinoForObjectDetection.from_pretrained(checkpoint_path, local_files_only=True)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded model from cache ({time.time() - start:.1f}s)")
            else:
                if not checkpoint_is_file:
                    raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")
                if log_fn is not None:
                    log_fn("Step: load .pth state_dict (torch.load)...")
                start = time.time()
                state_dict = self._load_gdino_state_dict(checkpoint_path)
                if log_fn is not None:
                    log_fn(f"Loaded .pth state_dict ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: load processor (AutoProcessor.from_pretrained)...")
                start = time.time()
                hang = _dump_on_hang("AutoProcessor.from_pretrained(config_id)", 30.0)
                processor = AutoProcessor.from_pretrained(config_id, local_files_only=True)
                hang.set()
                if log_fn is not None:
                    log_fn(f"Loaded processor ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: load config (GroundingDinoConfig.from_pretrained)...")
                start = time.time()
                config = GroundingDinoConfig.from_pretrained(config_id, local_files_only=True)
                if log_fn is not None:
                    log_fn(f"Loaded config ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: build model (GroundingDinoForObjectDetection(config))...")
                start = time.time()
                gdino = GroundingDinoForObjectDetection(config)
                if log_fn is not None:
                    log_fn(f"Built model ({time.time() - start:.1f}s)")
                if log_fn is not None:
                    log_fn("Step: apply weights (load_state_dict)...")
                start = time.time()
                missing, unexpected = gdino.load_state_dict(state_dict, strict=False)
                if log_fn is not None:
                    log_fn(f"Applied weights ({time.time() - start:.1f}s)")
                    if missing or unexpected:
                        log_fn(f"WARN: GroundingDINO missing={len(missing)} unexpected={len(unexpected)} keys.")
        except Exception as exc:
            hint = (
                "Cannot load GroundingDINO config/processor locally.\n\n"
                "This app runs in offline mode (no internet), so HuggingFace downloads will not work.\n\n"
                "Fix options:\n"
                "1) Set DINO 'Checkpoint' to a local HF model folder (contains config.json + tokenizer/preprocessor files).\n"
                "2) Or keep a .pth checkpoint but point 'Config ID' to a local HF model folder or cached model id.\n"
                "3) Or pre-download the HuggingFace repo on another machine, then copy it here.\n\n"
                f"ckpt={checkpoint_path}\nconfig={config_id}"
            )
            raise RuntimeError(f"{exc}\n\n{hint}") from exc
        finally:
            tick_stop.set()

        if log_fn is not None:
            log_fn("Step: move model to device (gdino.to)...")
        start = time.time()
        gdino.to(device)
        if log_fn is not None:
            log_fn(f"Moved model to device ({time.time() - start:.1f}s)")
        gdino.eval()
        self._processor = processor
        self._gdino = gdino
        self._gdino_checkpoint = checkpoint_path
        self._gdino_config_id = config_id
        self._device = device
        if log_fn is not None:
            log_fn(f"GroundingDINO ready (ckpt={checkpoint_path}, config={config_id}).")
        return self._processor, self._gdino, device

    def _run_with_roi(self, func_name: str, image_path: str, params: DinoParams, **kwargs) -> dict:
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        roi_x1, roi_y1, roi_x2, roi_y2 = params.roi_box
        roi_x1 = max(0, min(image.shape[1] - 1, int(roi_x1)))
        roi_y1 = max(0, min(image.shape[0] - 1, int(roi_y1)))
        roi_x2 = max(0, min(image.shape[1], int(roi_x2)))
        roi_y2 = max(0, min(image.shape[0], int(roi_y2)))
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            raise ValueError("Invalid ROI box")
        crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
        ext = os.path.splitext(image_path)[1] or ".png"
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        cv2.imwrite(tmp_path, crop)
        try:
            sub_params = DinoParams(
                gdino_checkpoint=params.gdino_checkpoint,
                gdino_config_id=params.gdino_config_id,
                text_queries=params.text_queries,
                box_threshold=params.box_threshold,
                text_threshold=params.text_threshold,
                max_dets=params.max_dets,
                device=params.device,
                output_dir=params.output_dir,
                roi_box=None,
            )
            func = getattr(self, func_name)
            result = dict(func(tmp_path, sub_params, **kwargs) or {})
            detections = []
            for det in result.get("detections") or []:
                item = dict(det)
                box = item.get("box")
                if isinstance(box, list) and len(box) == 4:
                    item["box"] = [
                        float(box[0]) + roi_x1,
                        float(box[1]) + roi_y1,
                        float(box[2]) + roi_x1,
                        float(box[3]) + roi_y1,
                    ]
                detections.append(item)
            result["image_path"] = str(image_path)
            result["detections"] = detections
            return result
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def predict(self, image_path: str, params: DinoParams, *, stop_checker=None, log_fn=None) -> dict:
        import cv2
        from PIL import Image

        if params.roi_box is not None:
            return self._run_with_roi("predict", image_path, params, stop_checker=stop_checker, log_fn=log_fn)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        os.makedirs(params.output_dir, exist_ok=True)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processor, gdino, device = self.ensure_model_loaded(params, log_fn=log_fn)
        if stop_checker is not None and stop_checker():
            raise RuntimeError("Stopped")
        if log_fn is not None:
            log_fn("Running DINO detect-only...")
        pil_img = Image.fromarray(rgb)
        detections = run_text_boxes(
            processor=processor,
            gdino=gdino,
            device=device,
            pil_image=pil_img,
            text_queries=list(params.text_queries),
            box_threshold=float(params.box_threshold),
            text_threshold=float(params.text_threshold),
        )
        detections = detections[: max(0, int(params.max_dets))] if int(params.max_dets) > 0 else detections
        payload = [
            {
                "label": str(det.label),
                "score": float(det.score),
                "box": [float(det.box_xyxy[0]), float(det.box_xyxy[1]), float(det.box_xyxy[2]), float(det.box_xyxy[3])],
                "model_name": "Dino",
            }
            for det in detections
        ]
        if log_fn is not None:
            log_fn(f"DINO done. dets={len(payload)}")
        return {
            "image_path": str(image_path),
            "output_dir": params.output_dir,
            "dets": len(payload),
            "detections": payload,
        }

    def predict_recursive(
        self,
        image_path: str,
        params: DinoParams,
        *,
        target_labels: Sequence[str] = ("crack",),
        max_depth: int = 3,
        min_box_px: int = 48,
        nonblack_thresh: int = 10,
        stop_checker=None,
        log_fn=None,
    ) -> dict:
        import cv2
        import numpy as np

        if params.roi_box is not None:
            return self._run_with_roi(
                "predict_recursive",
                image_path,
                params,
                target_labels=target_labels,
                max_depth=max_depth,
                min_box_px=min_box_px,
                nonblack_thresh=nonblack_thresh,
                stop_checker=stop_checker,
                log_fn=log_fn,
            )
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        nonzero_y, nonzero_x = np.where(gray > nonblack_thresh)
        if len(nonzero_x) == 0:
            if log_fn is not None:
                log_fn("WARN: Image is entirely black. Nothing to scan.")
            return {"image_path": str(image_path), "output_dir": params.output_dir, "dets": 0, "detections": []}
        bridge_x1, bridge_y1 = int(nonzero_x.min()), int(nonzero_y.min())
        bridge_x2, bridge_y2 = int(nonzero_x.max()), int(nonzero_y.max())
        if log_fn is not None:
            log_fn(f"Bridge ROI: ({bridge_x1},{bridge_y1})-({bridge_x2},{bridge_y2}), size={bridge_x2-bridge_x1}x{bridge_y2-bridge_y1}. max_depth={max_depth}")
        processor, gdino, device = self.ensure_model_loaded(params, log_fn=log_fn)
        detections = _recursive_zoom_detect(
            rgb,
            (bridge_x1, bridge_y1, bridge_x2, bridge_y2),
            (processor, gdino, device),
            list(params.text_queries),
            float(params.box_threshold),
            float(params.text_threshold),
            current_depth=0,
            max_depth=int(max_depth),
            min_box_px=int(min_box_px),
            stop_checker=stop_checker,
            log_fn=log_fn,
        )
        if stop_checker is not None and stop_checker():
            raise RuntimeError("Stopped")
        targets = normalize_queries(target_labels) if target_labels else []
        kept = []
        for box, label, score in detections:
            if targets and not label_matches(label, targets):
                continue
            kept.append((box, label, score))
        before_cf = len(kept)
        kept = _filter_parent_boxes(kept, contain_thresh=0.7)
        if log_fn is not None and len(kept) < before_cf:
            log_fn(f"Containment filter: removed {before_cf - len(kept)} parent box(es), {len(kept)} boxes remain.")
        kept = _nms_boxes(kept, iou_threshold=0.5)
        if int(params.max_dets) > 0:
            kept = kept[: int(params.max_dets)]
        payload = [
            {
                "label": str(label),
                "score": float(score),
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "model_name": "DinoRecursive",
            }
            for box, label, score in kept
        ]
        if log_fn is not None:
            log_fn(f"Recursive DINO done. dets={len(payload)}")
        return {
            "image_path": str(image_path),
            "output_dir": params.output_dir,
            "dets": len(payload),
            "detections": payload,
        }
