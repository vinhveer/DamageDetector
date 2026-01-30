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
            "Processor kh??ng c?? post_process_grounded_object_detection; "
            "h??y n??ng transformers ho???c d??ng model/processor t????ng th??ch."
        )
    return fn(
        outputs=outputs,
        input_ids=input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )


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
    delta_state = torch.load(delta_ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(delta_state, dict):
        raise TypeError(f"Unsupported delta checkpoint format: {type(delta_state)} ({delta_ckpt_path})")

    dt = delta_type.lower().strip()
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
        return 0, 0, overlay_path, mask_path

    predictor.set_image(rgb)

    rng = np.random.default_rng(seed)
    disp = bgr.copy()
    merged = np.zeros((h_img, w_img), dtype=np.uint8)

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

    if stop_checker is not None and stop_checker():
        raise StopRequested("Stopped")

    cv2.imwrite(overlay_path, disp)
    cv2.imwrite(mask_path, merged * 255)
    masks_saved = 1 if int(np.count_nonzero(merged)) > 0 else 0
    return len(dets), masks_saved, overlay_path, mask_path


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

    targets = normalize_queries(target_labels) if target_labels else []
    if len(targets) == 0:
        targets = normalize_queries(text_queries)

    dets_keep = [d for d in dets if label_matches(d.label, targets)]

    if len(dets) == 0 or len(dets_keep) == 0:
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
