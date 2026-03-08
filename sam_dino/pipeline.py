from __future__ import annotations

"""SAM + GroundingDINO pipeline utilities."""

import os
import re
import tempfile
from dataclasses import dataclass
from importlib import import_module
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoProcessor, GroundingDinoForObjectDetection


class StopRequested(RuntimeError):
    pass


@dataclass(frozen=True)
class Det:
    label: str
    box_xyxy: np.ndarray  # float32 (4,)
    score: float


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clip_box_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box.tolist()
    x1 = float(max(0, min(w - 1, x1)))
    y1 = float(max(0, min(h - 1, y1)))
    x2 = float(max(0, min(w - 1, x2)))
    y2 = float(max(0, min(h - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def safe_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def normalize_queries(text_queries: Sequence[str]) -> List[str]:
    queries = [q.strip() for q in text_queries if q.strip()]
    seen = set()
    deduped: List[str] = []
    for q in queries:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(q)
    return deduped


def label_matches(label: str, targets: Sequence[str]) -> bool:
    lbl = str(label or "").lower()
    for t in targets:
        tt = str(t or "").strip().lower()
        if not tt:
            continue
        if tt in lbl:
            return True
    return False


import inspect

def post_process_gdino(
    processor: AutoProcessor,
    outputs,
    input_ids,
    box_threshold: float,
    text_threshold: float,
    target_sizes,
):
    fn = getattr(processor, "post_process_grounded_object_detection", None)
    if fn is None:
        raise RuntimeError(
            "Processor khong co post_process_grounded_object_detection; "
            "hay nang transformers hoac dung model/processor tuong thich."
        )
    
    sig = inspect.signature(fn)
    kwargs = {
        "outputs": outputs,
        "input_ids": input_ids,
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in sig.parameters:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
        
    return fn(**kwargs)


def run_text_boxes(
    processor: AutoProcessor,
    gdino: GroundingDinoForObjectDetection,
    device: str,
    pil_image: Image.Image,
    text_queries: Sequence[str],
    box_threshold: float,
    text_threshold: float,
) -> List[Det]:
    if not text_queries or len(text_queries) == 0:
        return []

    w, h = pil_image.size
    caption = ". ".join(text_queries)
    with torch.no_grad():
        inputs = processor(images=pil_image, text=caption, return_tensors="pt").to(device)
        outputs = gdino(**inputs)
        target_sizes = torch.tensor([[h, w]], device=device)
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

    dets: List[Det] = []
    for box, score, label in zip(boxes, scores, labels):
        dets.append(Det(label=str(label), box_xyxy=box.astype(np.float32), score=float(score)))
    return dets


def overlay_mask(bgr: np.ndarray, mask01: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    out = bgr.copy()
    m = mask01.astype(bool)
    out[m] = (alpha * color + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def filter_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    if int(min_area) <= 0:
        return mask01
    m = (mask01 > 0).astype(np.uint8)
    if int(np.count_nonzero(m)) == 0:
        return m
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    keep = np.zeros(num, dtype=np.uint8)
    keep[0] = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            keep[i] = 1
    out = keep[labels].astype(np.uint8)
    return out


def load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)} ({checkpoint_path})")


def infer_sam_model_type_from_state_dict(state_dict: dict) -> Optional[str]:
    dim = None
    if "image_encoder.pos_embed" in state_dict:
        t = state_dict["image_encoder.pos_embed"]
        try:
            dim = int(t.shape[-1])
        except Exception:
            dim = None
    elif "image_encoder.patch_embed.proj.weight" in state_dict:
        t = state_dict["image_encoder.patch_embed.proj.weight"]
        try:
            dim = int(t.shape[0])
        except Exception:
            dim = None

    if dim == 768:
        return "vit_b"
    if dim == 1024:
        return "vit_l"
    if dim == 1280:
        return "vit_h"
    return None


def load_sam_model(checkpoint_path: str, requested_model_type: str) -> Tuple[torch.nn.Module, str]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint SAM: {checkpoint_path}")

    state_dict = load_checkpoint_state_dict(checkpoint_path)
    inferred = infer_sam_model_type_from_state_dict(state_dict)

    req = (requested_model_type or "auto").strip().lower()
    if req == "auto":
        if inferred is None:
            raise RuntimeError(
                "Không suy luận được SAM model type từ checkpoint. "
                "Hãy chọn vit_b|vit_l|vit_h đúng với checkpoint."
            )
        model_type = inferred
    else:
        model_type = req
        if inferred is not None and model_type != inferred:
            model_type = inferred

    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model type: {model_type!r} (use auto/vit_b/vit_l/vit_h)")

    sam = sam_model_registry[model_type](checkpoint=None)
    try:
        sam.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"SAM checkpoint/model-type không khớp. requested={requested_model_type!r}, inferred={inferred!r}.\n{e}"
        ) from e
    return sam, model_type


def resolve_best_delta_checkpoint(delta_type: str, delta_checkpoint: str) -> Optional[str]:
    if not delta_type or delta_type.lower() == "none":
        return None

    if delta_checkpoint and delta_checkpoint.lower() != "auto":
        if not os.path.isfile(delta_checkpoint):
            raise FileNotFoundError(f"Không tìm thấy delta checkpoint: {delta_checkpoint}")
        return delta_checkpoint

    dt = delta_type.lower().strip()
    candidates: List[str] = []
    search_dirs = [os.getcwd(), os.path.join(os.getcwd(), "checkpoints")]

    patterns = {
        "adapter": [r"(?i)adapter.*\\.pth$", r"(?i)delta.*adapter.*\\.pth$"],
        "lora": [r"(?i)lora.*\\.pth$", r"(?i)delta.*lora.*\\.pth$"],
        "both": [r"(?i)adapter.*lora.*\\.pth$", r"(?i)lora.*adapter.*\\.pth$"],
    }
    pats = patterns.get(dt, [])
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                continue
            if os.path.splitext(p)[1].lower() != ".pth":
                continue
            if re.search(r"(?i)^sam_.*\\.pth$", name):
                continue
            if any(re.search(pat, name) for pat in pats):
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy delta checkpoint phù hợp cho delta_type={delta_type!r} (auto search).")

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def apply_delta_to_sam(
    sam: torch.nn.Module,
    delta_type: str,
    delta_ckpt_path: str,
    middle_dim: int,
    scaling_factor: float,
    rank: int,
) -> None:
    try:
        delta_state = torch.load(delta_ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        delta_state = torch.load(delta_ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(delta_state, dict):
        raise TypeError(f"Unsupported delta checkpoint format: {type(delta_state)} ({delta_ckpt_path})")

    def _infer_delta_type_from_state(state: dict) -> str:
        keys = [str(k) for k in state.keys()]
        has_lora_suffix = any(k.startswith("w_a_") and k.endswith("_l") for k in keys) or any(
            k.startswith("w_b_") and k.endswith("_l") for k in keys
        )
        if has_lora_suffix:
            return "both"

        has_adapter_markers = any(k.startswith(("w_c_", "w_d_")) for k in keys) or any(k.endswith("_bia") for k in keys)
        if has_adapter_markers:
            return "adapter"

        has_lora_markers = any(k.startswith("w_a_") for k in keys) and any(k.startswith("w_b_") for k in keys)
        if has_lora_markers:
            return "lora"
        return "none"

    dt = delta_type.lower().strip()
    inferred = _infer_delta_type_from_state(delta_state)
    if inferred in {"adapter", "lora", "both"} and inferred != dt:
        print(
            f"WARN: delta_type={dt} but checkpoint looks like {inferred}. Auto-using {inferred}.",
            flush=True,
        )
        dt = inferred
    if dt == "adapter":
        ckpt_middle_dim = None
        for i in range(1000):
            k = f"w_a_{i:03d}"
            v = delta_state.get(k, None)
            if isinstance(v, torch.Tensor):
                ckpt_middle_dim = int(v.shape[0])
                break
        if ckpt_middle_dim is not None and int(middle_dim) != ckpt_middle_dim:
            middle_dim = ckpt_middle_dim
        mod = import_module("sam_finetune.delta.sam_adapter_image_encoder")
        wrapper = mod.Adapter_Sam(sam, int(middle_dim), float(scaling_factor))
    elif dt == "lora":
        mod = import_module("sam_finetune.delta.sam_lora_image_encoder")
        wrapper = mod.LoRA_Sam(sam, int(rank))
    elif dt == "both":
        ckpt_middle_dim = None
        for i in range(1000):
            k = f"w_a_{i:03d}"
            v = delta_state.get(k, None)
            if isinstance(v, torch.Tensor):
                ckpt_middle_dim = int(v.shape[0])
                break
        if ckpt_middle_dim is not None and int(middle_dim) != ckpt_middle_dim:
            middle_dim = ckpt_middle_dim
        mod = import_module("sam_finetune.delta.sam_adapter_lora_image_encoder")
        wrapper = mod.LoRA_Adapter_Sam(sam, int(middle_dim), int(rank))
    else:
        raise ValueError(f"Unknown delta_type: {delta_type!r} (use none/adapter/lora/both)")

    if dt in {"adapter", "both"}:
        tensor_specs = [
            ("w_down_attn", "w_a_{i:03d}", "w_a_{i:03d}_bia"),
            ("w_up_attn", "w_b_{i:03d}", "w_b_{i:03d}_bia"),
            ("w_down_mlp", "w_c_{i:03d}", "w_c_{i:03d}_bia"),
            ("w_up_mlp", "w_d_{i:03d}", "w_d_{i:03d}_bia"),
        ]
        for attr, w_tpl, b_tpl in tensor_specs:
            layers = getattr(wrapper, attr, None)
            if not layers:
                continue
            for i, layer in enumerate(layers):
                w_key = w_tpl.format(i=i)
                if w_key not in delta_state:
                    weight = getattr(layer, "weight", None)
                    if isinstance(weight, torch.Tensor):
                        delta_state[w_key] = torch.zeros_like(weight, device="cpu")

                b_key = b_tpl.format(i=i)
                if b_key not in delta_state:
                    bias = getattr(layer, "bias", None)
                    if isinstance(bias, torch.Tensor):
                        delta_state[b_key] = torch.zeros_like(bias, device="cpu")

    sam_state = sam.state_dict()
    for k, v in sam_state.items():
        if ("prompt_encoder" not in k) and ("mask_decoder" not in k):
            continue
        if k not in delta_state:
            delta_state[k] = v
            continue
        dv = delta_state[k]
        if isinstance(dv, torch.Tensor) and isinstance(v, torch.Tensor) and tuple(dv.shape) != tuple(v.shape):
            delta_state[k] = v

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(delta_state, tmp_path)
        wrapper.load_delta_parameters(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def process_one_image_text_box_sam_fullimage(
    image_path: str,
    out_dir: str,
    predictor: SamPredictor,
    processor: AutoProcessor,
    gdino: GroundingDinoForObjectDetection,
    device: str,
    text_queries: Sequence[str],
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    overlay_alpha: float,
    seed: int,
    invert_mask: bool,
    sam_min_component_area: int,
    sam_dilate_iters: int,
    *,
    stop_checker=None,
) -> Tuple[int, int, str, str]:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = bgr.shape[0], bgr.shape[1]

    pil_img = Image.fromarray(rgb)
    dets = run_text_boxes(
        processor=processor,
        gdino=gdino,
        device=device,
        pil_image=pil_img,
        text_queries=text_queries,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    dets = dets[: max(0, max_dets)] if max_dets > 0 else dets

    base = safe_basename(image_path)
    ensure_dir(out_dir)

    if stop_checker is not None and stop_checker():
        raise StopRequested("Stopped")

    overlay_path = os.path.join(out_dir, f"{base}_gdino_sam_overlay.png")
    mask_path = os.path.join(out_dir, f"{base}_crack_mask.png")

    if len(dets) == 0:
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return 0, 0, overlay_path, mask_path, []

    predictor.set_image(rgb)

    rng = np.random.default_rng(seed)
    disp = bgr.copy()
    merged = np.zeros((h_img, w_img), dtype=np.uint8)
    final_dets = []

    for det in dets:
        if stop_checker is not None and stop_checker():
            raise StopRequested("Stopped")

        x1, y1, x2, y2 = det.box_xyxy.tolist()
        label = det.label
        score = det.score

        x1 = float(max(0, min(w_img - 1, x1)))
        x2 = float(max(0, min(w_img - 1, x2)))
        y1 = float(max(0, min(h_img - 1, y1)))
        y2 = float(max(0, min(h_img - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        x1i = max(0, min(w_img, int(np.floor(x1))))
        y1i = max(0, min(h_img, int(np.floor(y1))))
        x2i = max(0, min(w_img, int(np.ceil(x2)) + 1))
        y2i = max(0, min(h_img, int(np.ceil(y2)) + 1))
        if x2i <= x1i or y2i <= y1i:
            continue

        cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
        cv2.putText(
            disp,
            f"{label} {score:.2f}",
            (int(x1), int(max(0, y1 - 5))),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            2,
        )

        input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        if masks is None or len(masks) == 0:
            continue
        idx = int(np.argmax(scores))
        chosen = masks[idx].astype(np.uint8)
        if invert_mask:
            chosen = (1 - chosen).astype(np.uint8)

        chosen_clip = np.zeros_like(chosen, dtype=np.uint8)
        chosen_clip[y1i:y2i, x1i:x2i] = chosen[y1i:y2i, x1i:x2i]
        chosen = chosen_clip

        chosen = filter_small_components(chosen, int(sam_min_component_area))
        if int(sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            chosen = cv2.dilate(chosen.astype(np.uint8), k, iterations=int(sam_dilate_iters)).astype(np.uint8)

        merged = np.maximum(merged, chosen)
        color = rng.integers(0, 255, (3,), dtype=np.uint8)
        disp = overlay_mask(disp, chosen, color=color, alpha=overlay_alpha)

        # Encode mask to png bytes for lightweight transport
        success, png_bytes = cv2.imencode(".png", chosen * 255)
        mask_b64 = None
        if success:
             import base64
             mask_b64 = base64.b64encode(png_bytes.tobytes()).decode('ascii')

        final_dets.append({
            "label": str(label),
            "score": float(score),
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "mask_b64": mask_b64,
            "model_name": "SamDino"
        })

    if stop_checker is not None and stop_checker():
        raise StopRequested("Stopped")

    cv2.imwrite(overlay_path, disp)
    cv2.imwrite(mask_path, merged * 255)
    masks_saved = 1 if int(np.count_nonzero(merged)) > 0 else 0
    return len(dets), masks_saved, overlay_path, mask_path, final_dets


def process_one_image_text_box_sam_isolate(
    image_path: str,
    out_dir: str,
    predictor: SamPredictor,
    processor: AutoProcessor,
    gdino: GroundingDinoForObjectDetection,
    device: str,
    text_queries: Sequence[str],
    target_labels: Sequence[str],
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    outside_value: int,
    crop_to_bbox: bool,
    overlay_alpha: float,
    seed: int,
    invert_mask: bool,
    sam_min_component_area: int,
    sam_dilate_iters: int,
    *,
    stop_checker=None,
    log_fn=None,
) -> Tuple[int, int, str, str, str]:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"KhÃ´ng Ä‘á»c Ä‘Æ°á»£c áº£nh: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = bgr.shape[0], bgr.shape[1]

    base = safe_basename(image_path)
    ensure_dir(out_dir)

    if stop_checker is not None and stop_checker():
        raise StopRequested("Stopped")

    overlay_path = os.path.join(out_dir, f"{base}_gdino_sam_isolate_overlay.png")
    mask_path = os.path.join(out_dir, f"{base}_gdino_sam_isolate_mask.png")
    isolate_path = os.path.join(out_dir, f"{base}_gdino_sam_isolate.png")

    pil_img = Image.fromarray(rgb)
    if log_fn:
        log_fn(f"Isolate: text_queries={list(text_queries)}, target_labels={list(target_labels)}")
        log_fn(f"Isolate: box_threshold={box_threshold}, text_threshold={text_threshold}")
    dets = run_text_boxes(
        processor=processor,
        gdino=gdino,
        device=device,
        pil_image=pil_img,
        text_queries=text_queries,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    dets = dets[: max(0, max_dets)] if max_dets > 0 else dets

    if log_fn:
        log_fn(f"GDINO found {len(dets)} detections total.")
        for d in dets:
            log_fn(f"  det: label={d.label!r}, score={d.score:.3f}, box={[round(x,1) for x in d.box_xyxy.tolist()]}")

    targets = normalize_queries(target_labels) if target_labels else []
    if len(targets) == 0:
        targets = normalize_queries(text_queries)

    dets_keep = [d for d in dets if label_matches(d.label, targets)]

    if log_fn:
        log_fn(f"Label filter targets={targets} -> kept {len(dets_keep)}/{len(dets)} detections.")

    if len(dets) == 0 or len(dets_keep) == 0:
        if log_fn:
            if len(dets) == 0:
                log_fn("WARN: GDINO found 0 detections. Returning blank image.")
            else:
                all_labels = [d.label for d in dets]
                log_fn(f"WARN: No detections matched target labels {targets}. All GDINO labels: {all_labels}")
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        blank = np.full_like(bgr, int(outside_value), dtype=np.uint8)
        cv2.imwrite(isolate_path, blank)
        return len(dets), 0, overlay_path, mask_path, isolate_path

    predictor.set_image(rgb)

    rng = np.random.default_rng(seed)
    disp = bgr.copy()
    merged = np.zeros((h_img, w_img), dtype=np.uint8)

    for det in dets_keep:
        if stop_checker is not None and stop_checker():
            raise StopRequested("Stopped")

        x1, y1, x2, y2 = det.box_xyxy.tolist()
        label = det.label
        score = det.score

        x1 = float(max(0, min(w_img - 1, x1)))
        x2 = float(max(0, min(w_img - 1, x2)))
        y1 = float(max(0, min(h_img - 1, y1)))
        y2 = float(max(0, min(h_img - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        x1i = max(0, min(w_img, int(np.floor(x1))))
        y1i = max(0, min(h_img, int(np.floor(y1))))
        x2i = max(0, min(w_img, int(np.ceil(x2)) + 1))
        y2i = max(0, min(h_img, int(np.ceil(y2)) + 1))
        if x2i <= x1i or y2i <= y1i:
            continue

        cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
        cv2.putText(
            disp,
            f"{label} {score:.2f}",
            (int(x1), int(max(0, y1 - 5))),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            2,
        )

        input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        if masks is None or len(masks) == 0:
            continue
        idx = int(np.argmax(scores))
        chosen = masks[idx].astype(np.uint8)
        if invert_mask:
            chosen = (1 - chosen).astype(np.uint8)

        chosen_clip = np.zeros_like(chosen, dtype=np.uint8)
        chosen_clip[y1i:y2i, x1i:x2i] = chosen[y1i:y2i, x1i:x2i]
        chosen = chosen_clip

        chosen = filter_small_components(chosen, int(sam_min_component_area))
        if int(sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            chosen = cv2.dilate(chosen.astype(np.uint8), k, iterations=int(sam_dilate_iters)).astype(np.uint8)

        merged = np.maximum(merged, chosen)
        color = rng.integers(0, 255, (3,), dtype=np.uint8)
        disp = overlay_mask(disp, chosen, color=color, alpha=overlay_alpha)

    if stop_checker is not None and stop_checker():
        raise StopRequested("Stopped")

    if int(np.count_nonzero(merged)) == 0:
        if log_fn:
            log_fn("WARN: SAM union mask is empty. Returning blank image.")
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        blank = np.full_like(bgr, int(outside_value), dtype=np.uint8)
        cv2.imwrite(isolate_path, blank)
        return len(dets), 0, overlay_path, mask_path, isolate_path

    mask01 = (merged > 0).astype(np.uint8)
    out = np.full_like(bgr, int(outside_value), dtype=np.uint8)
    out[mask01.astype(bool)] = bgr[mask01.astype(bool)]

    if crop_to_bbox:
        ys, xs = np.where(mask01 > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1c, x2c = int(xs.min()), int(xs.max())
            y1c, y2c = int(ys.min()), int(ys.max())
            out = out[y1c : y2c + 1, x1c : x2c + 1]

    cv2.imwrite(overlay_path, disp)
    cv2.imwrite(mask_path, merged * 255)
    cv2.imwrite(isolate_path, out)
    return len(dets), 1, overlay_path, mask_path, isolate_path


def process_one_image_sam_only(
    image_path: str,
    out_dir: str,
    predictor: SamPredictor,
    overlay_alpha: float,
    seed: int,
    invert_mask: bool,
    sam_min_component_area: int,
    sam_dilate_iters: int,
    *,
    stop_checker=None,
) -> Tuple[int, str, str, list]:
    """Run SAM only (no DINO) on the full image using a full-image bounding box prompt."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = bgr.shape[0], bgr.shape[1]

    base = safe_basename(image_path)
    ensure_dir(out_dir)

    overlay_path = os.path.join(out_dir, f"{base}_sam_only_overlay.png")
    mask_path = os.path.join(out_dir, f"{base}_crack_mask.png")

    if stop_checker is not None and stop_checker():
        raise StopRequested("Stopped")

    # Use a full-image box as the prompt (x1, y1, x2, y2)
    full_box = np.array([[0.0, 0.0, float(w_img - 1), float(h_img - 1)]], dtype=np.float32)

    predictor.set_image(rgb)

    if stop_checker is not None and stop_checker():
        raise StopRequested("Stopped")

    masks, scores, _ = predictor.predict(box=full_box, multimask_output=True)

    if masks is None or len(masks) == 0:
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return 0, overlay_path, mask_path, []

    idx = int(np.argmax(scores))
    chosen = masks[idx].astype(np.uint8)
    if invert_mask:
        chosen = (1 - chosen).astype(np.uint8)

    chosen = filter_small_components(chosen, int(sam_min_component_area))
    if int(sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        chosen = cv2.dilate(chosen.astype(np.uint8), k, iterations=int(sam_dilate_iters)).astype(np.uint8)

    rng = np.random.default_rng(seed)
    disp = bgr.copy()
    color = rng.integers(0, 255, (3,), dtype=np.uint8)
    disp = overlay_mask(disp, chosen, color=color, alpha=overlay_alpha)

    cv2.imwrite(overlay_path, disp)
    cv2.imwrite(mask_path, chosen * 255)

    import base64
    mask_b64 = None
    success, png_bytes = cv2.imencode(".png", chosen * 255)
    if success:
        mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii")

    final_dets = [{
        "label": "Mask",
        "score": float(scores[idx]),
        "box": [0.0, 0.0, float(w_img - 1), float(h_img - 1)],
        "mask_b64": mask_b64,
        "model_name": "SamOnly",
    }]

    masks_saved = 1 if int(np.count_nonzero(chosen)) > 0 else 0
    return masks_saved, overlay_path, mask_path, final_dets


# ──────────────────────────────────────────────────────────────────────────────
# Tiled crack detection
# ──────────────────────────────────────────────────────────────────────────────

def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _nms_boxes(boxes_info: list, iou_threshold: float = 0.5) -> list:
    """boxes_info: list of (box_xyxy np.ndarray, label str, score float)"""
    if not boxes_info:
        return []
    boxes_info = sorted(boxes_info, key=lambda x: x[2], reverse=True)
    used = [False] * len(boxes_info)
    keep = []
    for i, (bi, li, si) in enumerate(boxes_info):
        if used[i]:
            continue
        keep.append((bi, li, si))
        for j in range(i + 1, len(boxes_info)):
            if used[j]:
                continue
            if _box_iou(bi, boxes_info[j][0]) > iou_threshold:
                used[j] = True
    return keep


def _filter_parent_boxes(boxes_info: list, contain_thresh: float = 0.7) -> list:
    """
    Remove any box that 'contains' at least one smaller box.

    A box A is considered a PARENT of B if:
        intersection(A, B) / area(B) >= contain_thresh   AND   area(A) > area(B)

    If A is a parent, it is removed (the smaller children are kept).
    This prevents large coarse detections from surviving alongside their
    finer, more precise children.
    """
    n = len(boxes_info)
    if n <= 1:
        return boxes_info

    def _area(b: np.ndarray) -> float:
        return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))

    def _intersection(a: np.ndarray, b: np.ndarray) -> float:
        ix1 = max(float(a[0]), float(b[0]));  iy1 = max(float(a[1]), float(b[1]))
        ix2 = min(float(a[2]), float(b[2]));  iy2 = min(float(a[3]), float(b[3]))
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    is_parent = [False] * n
    for i in range(n):
        bi, _, _ = boxes_info[i]
        ai = _area(bi)
        if ai <= 0:
            continue
        for j in range(n):
            if i == j or is_parent[i]:
                break
            bj, _, _ = boxes_info[j]
            aj = _area(bj)
            if aj <= 0 or aj >= ai:          # j not strictly smaller than i
                continue
            inter = _intersection(bi, bj)
            if aj > 0 and inter / aj >= contain_thresh:
                is_parent[i] = True          # i contains j -> i is a parent -> remove i

    return [boxes_info[i] for i in range(n) if not is_parent[i]]


def process_one_image_tiled_crack_sam(
    image_path: str,
    out_dir: str,
    predictor: SamPredictor,
    processor: AutoProcessor,
    gdino: GroundingDinoForObjectDetection,
    device: str,
    text_queries: Sequence[str],
    target_labels: Sequence[str],   # only keep boxes whose label matches these
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    overlay_alpha: float,
    seed: int,
    invert_mask: bool,
    sam_min_component_area: int,
    sam_dilate_iters: int,
    tile_size: int = 640,
    tile_overlap: float = 0.25,
    nonblack_thresh: int = 10,
    *,
    stop_checker=None,
    log_fn=None,
) -> Tuple[int, int, str, str, list]:
    """
    1. Find non-black bounding box (bridge mask region).
    2. Tile that region with overlap.
    3. Run GDINO on each tile.
    4. Keep ONLY boxes whose label matches target_labels (e.g. 'crack').
    5. NMS across tiles.
    6. Run SAM on each kept box -> union mask.
    """
    import base64

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = bgr.shape[:2]

    base = safe_basename(image_path)
    ensure_dir(out_dir)

    overlay_path = os.path.join(out_dir, f"{base}_tiled_crack_overlay.png")
    mask_path    = os.path.join(out_dir, f"{base}_crack_mask.png")

    # ── 1. Find bridge (non-black) bounding box ──────────────────────────────
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    nz_ys, nz_xs = np.where(gray > nonblack_thresh)
    if len(nz_xs) == 0:
        if log_fn:
            log_fn("WARN: Image is entirely black. Nothing to scan.")
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return 0, 0, overlay_path, mask_path, []

    roi_x1, roi_y1 = int(nz_xs.min()), int(nz_ys.min())
    roi_x2, roi_y2 = int(nz_xs.max()), int(nz_ys.max())

    if log_fn:
        log_fn(f"Bridge ROI: ({roi_x1},{roi_y1})-({roi_x2},{roi_y2}), "
               f"size=({roi_x2-roi_x1} x {roi_y2-roi_y1})")

    # ── 2. Generate tiles ─────────────────────────────────────────────────────
    step = max(1, int(tile_size * (1.0 - tile_overlap)))
    tiles: List[Tuple[int, int, int, int]] = []
    y = roi_y1
    while y < roi_y2:
        x = roi_x1
        while x < roi_x2:
            tx2 = min(x + tile_size, roi_x2)
            ty2 = min(y + tile_size, roi_y2)
            if tx2 > x and ty2 > y:
                tiles.append((x, y, tx2, ty2))
            x += step
        y += step

    targets = normalize_queries(target_labels) if target_labels else normalize_queries(text_queries)
    if log_fn:
        log_fn(f"Generated {len(tiles)} tiles (size={tile_size}px, overlap={tile_overlap:.0%}). "
               f"Label filter: {targets}")

    # ── 3-4. GDINO on each tile, filter labels ───────────────────────────────
    accepted: list = []  # (box_full np.ndarray float32, label, score)
    for idx_t, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        if stop_checker is not None and stop_checker():
            raise StopRequested("Stopped")

        tile_bgr = bgr[ty1:ty2, tx1:tx2]
        tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
        if np.count_nonzero(tile_gray > nonblack_thresh) / max(1, tile_gray.size) < 0.05:
            continue  # skip mostly-black tiles

        tile_rgb = rgb[ty1:ty2, tx1:tx2]
        pil_tile = Image.fromarray(tile_rgb)
        dets = run_text_boxes(
            processor=processor, gdino=gdino, device=device,
            pil_image=pil_tile,
            text_queries=text_queries,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        for d in dets:
            if not label_matches(d.label, targets):
                continue
            # Map tile coords -> full image coords
            rx1, ry1, rx2, ry2 = d.box_xyxy.tolist()
            full_box = np.array([rx1 + tx1, ry1 + ty1, rx2 + tx1, ry2 + ty1], dtype=np.float32)
            accepted.append((full_box, d.label, float(d.score)))

    if log_fn:
        log_fn(f"Total label-filtered boxes from all tiles: {len(accepted)}")

    if not accepted:
        if log_fn:
            log_fn("WARN: No crack boxes found. Try lowering box_threshold or adding more queries.")
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return len(tiles), 0, overlay_path, mask_path, []

    # ── 5. NMS ────────────────────────────────────────────────────────────────
    accepted = _nms_boxes(accepted, iou_threshold=0.5)
    if max_dets > 0:
        accepted = accepted[:max_dets]
    if log_fn:
        log_fn(f"After NMS & cap: {len(accepted)} boxes -> SAM")

    # ── 6. SAM on each kept box ───────────────────────────────────────────────
    predictor.set_image(rgb)
    rng  = np.random.default_rng(seed)
    disp = bgr.copy()
    merged   = np.zeros((h_img, w_img), dtype=np.uint8)
    final_dets = []

    for full_box, label, score in accepted:
        if stop_checker is not None and stop_checker():
            raise StopRequested("Stopped")

        x1, y1f, x2, y2f = full_box.tolist()
        x1  = float(max(0, min(w_img - 1, x1)))
        x2  = float(max(0, min(w_img - 1, x2)))
        y1f = float(max(0, min(h_img - 1, y1f)))
        y2f = float(max(0, min(h_img - 1, y2f)))
        if x2 <= x1 or y2f <= y1f:
            continue

        x1i = max(0, int(np.floor(x1)));  x2i = min(w_img, int(np.ceil(x2)) + 1)
        y1i = max(0, int(np.floor(y1f))); y2i = min(h_img, int(np.ceil(y2f)) + 1)

        input_box = np.array([[x1, y1f, x2, y2f]], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        if masks is None or len(masks) == 0:
            continue

        best_idx = int(np.argmax(scores))
        chosen = masks[best_idx].astype(np.uint8)
        if invert_mask:
            chosen = (1 - chosen).astype(np.uint8)

        clip = np.zeros_like(chosen, dtype=np.uint8)
        clip[y1i:y2i, x1i:x2i] = chosen[y1i:y2i, x1i:x2i]
        chosen = clip

        chosen = filter_small_components(chosen, int(sam_min_component_area))
        if int(sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            chosen = cv2.dilate(chosen.astype(np.uint8), k, iterations=int(sam_dilate_iters)).astype(np.uint8)

        merged = np.maximum(merged, chosen)

        color = rng.integers(0, 255, (3,), dtype=np.uint8)
        cv2.rectangle(disp, (int(x1), int(y1f)), (int(x2), int(y2f)), (0, 200, 255), 2)
        cv2.putText(disp, f"{label} {score:.2f}", (int(x1), int(max(0, y1f - 5))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        disp = overlay_mask(disp, chosen, color=color, alpha=overlay_alpha)

        mask_b64 = None
        ok, png_bytes = cv2.imencode(".png", chosen * 255)
        if ok:
            mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii")
        final_dets.append({
            "label": str(label), "score": float(score),
            "box": [float(x1), float(y1f), float(x2), float(y2f)],
            "mask_b64": mask_b64, "model_name": "SamDinoTiled",
        })

    cv2.imwrite(overlay_path, disp)
    cv2.imwrite(mask_path, merged * 255)
    masks_saved = 1 if int(np.count_nonzero(merged)) > 0 else 0
    if log_fn:
        log_fn(f"Done. {len(final_dets)} crack boxes segmented. masks_saved={masks_saved}")
    return len(accepted), masks_saved, overlay_path, mask_path, final_dets


# ──────────────────────────────────────────────────────────────────────────────
# Recursive zoom-in detection
# ──────────────────────────────────────────────────────────────────────────────

def _recursive_zoom_detect(
    rgb: np.ndarray,
    crop_box: Tuple[int, int, int, int],  # (x1,y1,x2,y2) in full-image coords
    gdino_items: tuple,                   # (processor, gdino, device)
    text_queries: Sequence[str],
    box_threshold: float,
    text_threshold: float,
    current_depth: int,
    max_depth: int,
    min_box_px: int = 48,
    stop_checker=None,
    log_fn=None,
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Recursively zoom into `crop_box` and return FINEST-level detections.

    For each box found at this level:
      - If zooming IN yields more boxes -> use those (finer), discard parent
      - If zooming IN yields nothing    -> keep parent as a leaf

    Returns list of (box_xyxy_full_image, label, score).
    """
    cx1, cy1, cx2, cy2 = crop_box
    w, h = cx2 - cx1, cy2 - cy1

    # Skip degenerate / too-small crops
    if w < min_box_px or h < min_box_px:
        return []

    if stop_checker is not None and stop_checker():
        raise StopRequested("Stopped")

    processor, gdino, device = gdino_items
    crop_rgb = rgb[cy1:cy2, cx1:cx2]
    pil_crop = Image.fromarray(crop_rgb)

    dets = run_text_boxes(
        processor=processor, gdino=gdino, device=device,
        pil_image=pil_crop,
        text_queries=text_queries,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if not dets:
        return []  # nothing found in this region

    # Map detections from crop-coords -> full-image coords
    dets_full: List[Tuple[np.ndarray, str, float]] = []
    for d in dets:
        rx1, ry1, rx2, ry2 = d.box_xyxy.tolist()
        fb = np.array([rx1 + cx1, ry1 + cy1, rx2 + cx1, ry2 + cy1], dtype=np.float32)
        dets_full.append((fb, d.label, float(d.score)))

    if log_fn and dets_full:
        log_fn(f"  [depth {current_depth}] crop ({cx1},{cy1})-({cx2},{cy2}) "
               f"-> {len(dets_full)} dets")

    if current_depth >= max_depth:
        return dets_full  # max depth reached, return what we have

    # For each detection, try to zoom in further
    result: List[Tuple[np.ndarray, str, float]] = []
    for box_full, label, score in dets_full:
        bx1, by1, bx2, by2 = (int(v) for v in box_full.tolist())
        children = _recursive_zoom_detect(
            rgb, (bx1, by1, bx2, by2),
            gdino_items, text_queries,
            box_threshold, text_threshold,
            current_depth + 1, max_depth,
            min_box_px=min_box_px,
            stop_checker=stop_checker,
            log_fn=log_fn,
        )
        if children:
            result.extend(children)   # finer results -> use them, discard parent
        else:
            result.append((box_full, label, score))  # leaf -> keep

    return result


def process_one_image_recursive_crack_sam(
    image_path: str,
    out_dir: str,
    predictor: SamPredictor,
    processor: AutoProcessor,
    gdino: GroundingDinoForObjectDetection,
    device: str,
    text_queries: Sequence[str],
    target_labels: Sequence[str],   # SAM only on boxes whose label matches these
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    overlay_alpha: float,
    seed: int,
    invert_mask: bool,
    sam_min_component_area: int,
    sam_dilate_iters: int,
    max_depth: int = 3,
    min_box_px: int = 48,
    nonblack_thresh: int = 10,
    *,
    stop_checker=None,
    log_fn=None,
) -> Tuple[int, int, str, str, list]:
    """
    Recursive zoom-in detection + SAM segmentation.

    Algorithm:
      1. Find non-black (bridge) bounding box.
      2. Run GDINO on the bridge region.
      3. For each found box: zoom in and run GDINO again.
         - If zoom yields boxes  -> replace parent with children (finer)
         - If zoom yields nothing -> keep parent (leaf)
      4. Repeat until no new boxes or max_depth reached.
      5. Filter final boxes by target_labels (e.g. 'crack').
      6. NMS + SAM segmentation.
    """
    import base64

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = bgr.shape[:2]

    base_name = safe_basename(image_path)
    ensure_dir(out_dir)

    overlay_path = os.path.join(out_dir, f"{base_name}_recursive_crack_overlay.png")
    mask_path    = os.path.join(out_dir, f"{base_name}_crack_mask.png")

    # ── 1. Find bridge (non-black) bounding box ───────────────────────────────
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    nz_ys, nz_xs = np.where(gray > nonblack_thresh)
    if len(nz_xs) == 0:
        if log_fn:
            log_fn("WARN: Image is entirely black. Nothing to scan.")
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return 0, 0, overlay_path, mask_path, []

    bx1, by1 = int(nz_xs.min()), int(nz_ys.min())
    bx2, by2 = int(nz_xs.max()), int(nz_ys.max())

    if log_fn:
        log_fn(f"Bridge ROI: ({bx1},{by1})-({bx2},{by2}), "
               f"size={bx2-bx1}x{by2-by1}. max_depth={max_depth}")

    # ── 2-4. Recursive detection ──────────────────────────────────────────────
    gdino_items = (processor, gdino, device)
    all_boxes = _recursive_zoom_detect(
        rgb, (bx1, by1, bx2, by2),
        gdino_items, text_queries,
        box_threshold, text_threshold,
        current_depth=0, max_depth=max_depth,
        min_box_px=min_box_px,
        stop_checker=stop_checker,
        log_fn=log_fn,
    )

    if log_fn:
        log_fn(f"Recursive detection total leaf boxes: {len(all_boxes)}")
        all_labels = sorted(set(l for _, l, _ in all_boxes))
        log_fn(f"DINO labels found: {all_labels} -> ALL will be segmented by SAM as 'crack'")

    # ── No label filter: keep ALL boxes DINO found ────────────────────────────
    # SAM segments using the bounding BOX only (not the text label).
    # All outputs will be labeled 'crack' regardless of DINO's label.
    kept = list(all_boxes)

    if not kept:
        if log_fn:
            log_fn("WARN: DINO found no boxes at all. Returning empty mask.")
        cv2.imwrite(overlay_path, bgr)
        cv2.imwrite(mask_path, np.zeros((h_img, w_img), dtype=np.uint8))
        return 0, 0, overlay_path, mask_path, []

    # ── Containment filter + NMS + cap ────────────────────────────────────────
    before_cf = len(kept)
    kept = _filter_parent_boxes(kept, contain_thresh=0.7)
    if log_fn and len(kept) < before_cf:
        log_fn(f"Containment filter: removed {before_cf - len(kept)} parent box(es), "
               f"{len(kept)} boxes remain.")

    kept = _nms_boxes(kept, iou_threshold=0.5)
    if max_dets > 0:
        kept = kept[:max_dets]
    if log_fn:
        log_fn(f"After NMS & cap: {len(kept)} boxes -> SAM")

    # ── 6. SAM segmentation ───────────────────────────────────────────────────
    predictor.set_image(rgb)
    rng = np.random.default_rng(seed)
    disp = bgr.copy()
    merged = np.zeros((h_img, w_img), dtype=np.uint8)
    final_dets = []

    for box, label, score in kept:
        if stop_checker is not None and stop_checker():
            raise StopRequested("Stopped")

        x1, y1f, x2, y2f = box.tolist()
        x1  = float(max(0, min(w_img - 1, x1)));  x2  = float(max(0, min(w_img - 1, x2)))
        y1f = float(max(0, min(h_img - 1, y1f)));  y2f = float(max(0, min(h_img - 1, y2f)))
        if x2 <= x1 or y2f <= y1f:
            continue

        x1i = max(0, int(np.floor(x1)));   x2i = min(w_img, int(np.ceil(x2)) + 1)
        y1i = max(0, int(np.floor(y1f)));  y2i = min(h_img, int(np.ceil(y2f)) + 1)

        masks_sam, scores_sam, _ = predictor.predict(
            box=np.array([[x1, y1f, x2, y2f]], dtype=np.float32),
            multimask_output=True,
        )
        if masks_sam is None or len(masks_sam) == 0:
            continue

        best = int(np.argmax(scores_sam))
        chosen = masks_sam[best].astype(np.uint8)
        if invert_mask:
            chosen = (1 - chosen).astype(np.uint8)

        clip = np.zeros_like(chosen, dtype=np.uint8)
        clip[y1i:y2i, x1i:x2i] = chosen[y1i:y2i, x1i:x2i]
        chosen = clip

        chosen = filter_small_components(chosen, int(sam_min_component_area))
        if int(sam_dilate_iters) > 0 and int(np.count_nonzero(chosen)) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            chosen = cv2.dilate(chosen.astype(np.uint8), k, iterations=int(sam_dilate_iters)).astype(np.uint8)

        merged = np.maximum(merged, chosen)

        color = rng.integers(0, 255, (3,), dtype=np.uint8)
        # Always label as 'crack' in the overlay regardless of DINO's original label
        cv2.rectangle(disp, (int(x1), int(y1f)), (int(x2), int(y2f)), (0, 200, 255), 2)
        cv2.putText(disp, f"crack {score:.2f} ({label})",
                    (int(x1), int(max(0, y1f - 5))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        disp = overlay_mask(disp, chosen, color=color, alpha=overlay_alpha)

        mask_b64 = None
        ok, png_bytes = cv2.imencode(".png", chosen * 255)
        if ok:
            mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii")
        final_dets.append({
            "label": "crack",          # force to 'crack' regardless of DINO label
            "score": float(score),
            "box": [float(x1), float(y1f), float(x2), float(y2f)],
            "mask_b64": mask_b64, "model_name": "SamDinoRecursive",
            "dino_label": str(label),  # keep original DINO label for reference
        })

    cv2.imwrite(overlay_path, disp)
    cv2.imwrite(mask_path, merged * 255)
    masks_saved = 1 if int(np.count_nonzero(merged)) > 0 else 0
    if log_fn:
        log_fn(f"Done. {len(final_dets)} boxes -> SAM. masks_saved={masks_saved}")
    return len(kept), masks_saved, overlay_path, mask_path, final_dets
