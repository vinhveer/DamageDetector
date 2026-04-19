from __future__ import annotations

import base64
import os
from typing import Any, Callable

import cv2

from object_detection.dino import get_dino_service
from inference_api.contracts import InferenceRequest, InferenceResult, JobEvent
from segmentation.sam.runtime import get_sam_service
from segmentation.sam.finetune import get_sam_finetune_service
from segmentation.unet import get_unet_service


class WorkflowContext:
    def __init__(
        self,
        *,
        job_id: str,
        request: InferenceRequest,
        emit_event: Callable[[JobEvent], None],
        stop_checker: Callable[[], bool],
        register_service: Callable[[str], None],
    ) -> None:
        self.job_id = job_id
        self.request = request
        self._emit_event = emit_event
        self.stop_checker = stop_checker
        self._register_service = register_service

    def log(self, text: str) -> None:
        self._emit_event(JobEvent(type="progress", job_id=self.job_id, workflow=self.request.workflow, message=str(text)))

    def partial(self, payload: dict[str, Any], *, message: str | None = None) -> None:
        self._emit_event(
            JobEvent(
                type="partial_result",
                job_id=self.job_id,
                workflow=self.request.workflow,
                message=message,
                result=InferenceResult(job_id=self.job_id, workflow=self.request.workflow, payload=dict(payload)),
            )
        )

    def completed(self, payload: dict[str, Any]) -> InferenceResult:
        return InferenceResult(job_id=self.job_id, workflow=self.request.workflow, payload=dict(payload))

    def call_service(
        self,
        service_name: str,
        service_getter: Callable[[], Any],
        method: str,
        params: dict[str, Any],
    ) -> Any:
        self._register_service(str(service_name))
        service = service_getter()
        return service.call(method, params, log_fn=self.log, stop_checker=self.stop_checker)


def _with_b64_mask(det: dict[str, Any]) -> dict[str, Any]:
    out = dict(det)
    mask_path = str(out.get("mask_path") or "")
    if not mask_path or not os.path.isfile(mask_path) or out.get("mask_b64"):
        return out
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return out
    success, png_bytes = cv2.imencode(".png", mask_img)
    if not success:
        return out
    out["mask_b64"] = base64.b64encode(png_bytes.tobytes()).decode("ascii")
    return out


def _merge_detection_boxes(detections: list[dict[str, Any]], *, pad: int = 50) -> tuple[int, int, int, int] | None:
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")
    has_roi = False
    for det in detections:
        box = det.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
        has_roi = True
    if not has_roi:
        return None
    return (
        int(min_x - pad),
        int(min_y - pad),
        int(max_x + pad),
        int(max_y + pad),
    )


def _filter_detections_by_labels(detections: list[dict[str, Any]], target_labels: list[str]) -> list[dict[str, Any]]:
    labels = [str(label).strip().lower() for label in target_labels if str(label).strip()]
    if not labels:
        return list(detections)
    filtered = []
    for det in detections:
        label = str(det.get("label") or "").lower()
        if any(target in label for target in labels):
            filtered.append(det)
    return filtered


def _set_detection_model_name(payload: dict[str, Any], model_name: str) -> dict[str, Any]:
    result = dict(payload or {})
    detections = []
    for det in result.get("detections") or []:
        item = dict(det)
        item["model_name"] = model_name
        detections.append(item)
    if detections:
        result["detections"] = detections
    return result


def _set_detection_label(payload: dict[str, Any], label_text: str) -> dict[str, Any]:
    cleaned = str(label_text or "").strip()
    if not cleaned:
        return dict(payload or {})
    result = dict(payload or {})
    detections = []
    source = list(result.get("detections") or [])
    total = len(source)
    for idx, det in enumerate(source, start=1):
        item = dict(det)
        item["label"] = f"{cleaned} #{idx}" if total > 1 else cleaned
        detections.append(item)
    if detections:
        result["detections"] = detections
    return result


def _is_crack_detection(det: dict[str, Any]) -> bool:
    return "crack" in str(det.get("label") or "").strip().lower()


def _decode_detection_mask(det: dict[str, Any], *, image_shape: tuple[int, int]) -> Any | None:
    import numpy as np

    mask_img = None
    mask_b64 = det.get("mask_b64")
    if mask_b64:
        try:
            raw = base64.b64decode(str(mask_b64))
            encoded = np.frombuffer(raw, dtype=np.uint8)
            mask_img = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        except Exception:
            mask_img = None
    if mask_img is None:
        mask_path = str(det.get("mask_path") or "")
        if mask_path and os.path.isfile(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None
    height, width = image_shape
    if mask_img.shape[:2] != (height, width):
        mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask_img > 0).astype("uint8")


def _compose_segmented_output(
    *,
    image_path: str,
    output_dir: str,
    detections: list[dict[str, Any]],
    overlay_alpha: float,
    output_tag: str,
) -> dict[str, Any]:
    import numpy as np

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    height, width = image.shape[:2]
    os.makedirs(output_dir, exist_ok=True)
    overlay = image.copy()
    merged = np.zeros((height, width), dtype=np.uint8)
    final_detections: list[dict[str, Any]] = []
    rng = np.random.default_rng(1337)

    for det in detections:
        item = _with_b64_mask(det)
        mask01 = _decode_detection_mask(item, image_shape=(height, width))
        if mask01 is None or int(mask01.sum()) <= 0:
            continue
        final_detections.append(item)
        merged = np.maximum(merged, mask01)
        color = rng.integers(32, 255, (3,), dtype=np.uint8)
        mask_pixels = mask01 > 0
        blended = (
            overlay[mask_pixels].astype(np.float32) * (1.0 - float(overlay_alpha))
            + color.astype(np.float32) * float(overlay_alpha)
        )
        overlay[mask_pixels] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        box = item.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(v) for v in color.tolist()), 2)
            label = str(item.get("label") or "object")
            cv2.putText(
                overlay,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                tuple(int(v) for v in color.tolist()),
                2,
            )

    base = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{base}_{output_tag}_mask.png")
    overlay_path = os.path.join(output_dir, f"{base}_{output_tag}_overlay.png")
    cv2.imwrite(mask_path, merged * 255)
    cv2.imwrite(overlay_path, overlay)
    return {
        "image_path": str(image_path),
        "output_dir": output_dir,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "masks_saved": 1 if int(merged.sum()) > 0 else 0,
        "detections": final_detections,
    }


def _segment_boxes_with_unet_override(
    ctx: WorkflowContext,
    *,
    image_path: str,
    unet_params: dict[str, Any],
    boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    height, width = image.shape[:2]
    output_root = str(unet_params.get("output_dir") or "results_unet")
    detections: list[dict[str, Any]] = []

    for index, entry in enumerate(boxes, start=1):
        box = entry.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        roi = (
            max(0, min(width, int(x1))),
            max(0, min(height, int(y1))),
            max(0, min(width, int(x2) + 1)),
            max(0, min(height, int(y2) + 1)),
        )
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            continue
        roi_params = dict(unet_params)
        roi_params["output_dir"] = os.path.join(output_root, "crack_unet", f"roi_{index:03d}")
        unet_res = ctx.call_service(
            "unet",
            get_unet_service,
            "run_rois",
            {"image_path": image_path, "params": roi_params, "rois": [roi]},
        )
        if isinstance(unet_res, dict) and unet_res.get("stopped"):
            return {"stopped": True}
        mask_path = str(unet_res.get("mask_path") or "")
        if not mask_path or not os.path.isfile(mask_path):
            continue
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        if mask_img.shape[:2] != (height, width):
            mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
        if int(np.count_nonzero(mask_img)) <= 0:
            continue
        success, png_bytes = cv2.imencode(".png", mask_img)
        detections.append(
            {
                "label": str(entry.get("label") or "crack"),
                "score": float(entry.get("score") or 0.0),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "mask_path": mask_path,
                "mask_b64": base64.b64encode(png_bytes.tobytes()).decode("ascii") if success else None,
                "model_name": "UnetCrack",
            }
        )
    return {
        "image_path": str(image_path),
        "output_dir": output_root,
        "detections": detections,
    }


def _write_isolate_from_mask(
    *,
    image_path: str,
    mask_path: str,
    output_dir: str,
    outside_value: int,
    crop_to_bbox: bool,
    action: str,
) -> str | None:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    keep = mask > 0
    action_name = str(action or "keep").strip().lower()
    isolate = image.copy()
    if action_name == "erase":
        isolate[keep] = int(outside_value)
    else:
        isolate[~keep] = int(outside_value)
    if crop_to_bbox and action_name != "erase":
        ys, xs = keep.nonzero()
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            isolate = isolate[y1:y2, x1:x2]
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    if action_name == "erase":
        suffix = "erase"
    else:
        suffix = "crop_isolate" if crop_to_bbox else "isolate"
    isolate_path = os.path.join(output_dir, f"{base}_{suffix}.png")
    if not cv2.imwrite(isolate_path, isolate):
        return None
    return isolate_path


def _image_max_dim(image_path: str) -> int:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
            return max(int(width), int(height))
    except Exception:
        return 0


def _run_single_or_batch(
    *,
    ctx: WorkflowContext,
    service_name: str,
    service_getter: Callable[[], Any],
    params: dict[str, Any],
) -> InferenceResult:
    if ctx.request.image_paths:
        payload = ctx.call_service(service_name, service_getter, "predict_batch", {"image_paths": list(ctx.request.image_paths), "params": params})
        return ctx.completed(dict(payload or {}))
    payload = ctx.call_service(service_name, service_getter, "predict", {"image_path": str(ctx.request.image_path or ""), "params": params})
    return ctx.completed(dict(payload or {}))


def _run_sam_dino_single(
    ctx: WorkflowContext,
    *,
    image_path: str,
    dino_params: dict[str, Any],
    sam_params: dict[str, Any],
    sam_service_name: str,
    sam_getter: Callable[[], Any],
    use_tiled_dino: bool,
    tile_trigger_px: int,
    target_labels: list[str],
) -> dict[str, Any]:
    max_dim = _image_max_dim(image_path) if use_tiled_dino else 0
    if use_tiled_dino and max_dim > tile_trigger_px:
        ctx.log(f"Large image detected ({max_dim}px). Using tiled DINO before SAM.")
        dino_res = ctx.call_service(
            "dino",
            get_dino_service,
            "recursive_detect",
            {
                "image_path": image_path,
                "params": dino_params,
                "target_labels": list(target_labels or dino_params.get("text_queries") or ["object"]),
                "max_depth": int(dino_params.get("recursive_max_depth") or 3),
                "min_box_px": int(dino_params.get("recursive_min_box_px") or 48),
            },
        )
        partial_message = "Recursive DINO detections ready"
        empty_message = "No tiled DINO detections. Skipping SAM box prompting."
    else:
        dino_res = ctx.call_service("dino", get_dino_service, "predict", {"image_path": image_path, "params": dino_params})
        partial_message = "DINO detections ready"
        empty_message = "No DINO detections. Skipping SAM box prompting."
    if isinstance(dino_res, dict) and dino_res.get("stopped"):
        return {"stopped": True}
    display_detections = list(dino_res.get("display_detections") or dino_res.get("detections") or [])
    detections = list(dino_res.get("detections") or [])
    if display_detections:
        ctx.partial({"image_path": image_path, "detections": display_detections}, message=partial_message)
    if not detections:
        ctx.log(empty_message)
        return {"image_path": image_path, "detections": [], "display_detections": display_detections}
    sam_boxes = display_detections or detections
    ctx.log(f"SAM box prompting on {len(sam_boxes)} DINO boxes...")
    crack_mask_model = str(ctx.request.params.get("crack_mask_model") or "off").strip().lower()
    crack_boxes = [det for det in sam_boxes if _is_crack_detection(det)]
    non_crack_boxes = [det for det in sam_boxes if not _is_crack_detection(det)]

    if crack_mask_model == "off" or not crack_boxes:
        sam_res = ctx.call_service(
            sam_service_name,
            sam_getter,
            "segment_boxes",
            {"image_path": image_path, "params": sam_params, "boxes": sam_boxes},
        )
        payload = dict(sam_res or {})
        if display_detections:
            payload["display_detections"] = display_detections
        if use_tiled_dino and max_dim > tile_trigger_px:
            model_name = "SamTiled" if sam_service_name == "sam" else "SamFinetuneTiled"
            payload = _set_detection_model_name(payload, model_name)
        return payload

    ctx.log(
        f"Crack override active: {len(crack_boxes)} crack box(es) -> {crack_mask_model}. "
        f"Other boxes stay on {sam_service_name}."
    )
    segmented_detections: list[dict[str, Any]] = []

    if non_crack_boxes:
        main_res = ctx.call_service(
            sam_service_name,
            sam_getter,
            "segment_boxes",
            {"image_path": image_path, "params": sam_params, "boxes": non_crack_boxes},
        )
        if isinstance(main_res, dict) and main_res.get("stopped"):
            return {"stopped": True}
        segmented_detections.extend(list((main_res or {}).get("detections") or []))

    if crack_mask_model == "sam_lora":
        crack_params = dict(ctx.request.params.get("crack_sam") or {})
        crack_res = ctx.call_service(
            "sam_finetune",
            get_sam_finetune_service,
            "segment_boxes",
            {"image_path": image_path, "params": crack_params, "boxes": crack_boxes},
        )
        if isinstance(crack_res, dict) and crack_res.get("stopped"):
            return {"stopped": True}
        crack_res = _set_detection_model_name(dict(crack_res or {}), "SamFinetuneCrack")
        segmented_detections.extend(list(crack_res.get("detections") or []))
    elif crack_mask_model == "unet":
        crack_params = dict(ctx.request.params.get("crack_unet") or {})
        crack_res = _segment_boxes_with_unet_override(
            ctx,
            image_path=image_path,
            unet_params=crack_params,
            boxes=crack_boxes,
        )
        if isinstance(crack_res, dict) and crack_res.get("stopped"):
            return {"stopped": True}
        segmented_detections.extend(list((crack_res or {}).get("detections") or []))
    else:
        raise ValueError(f"Unsupported crack mask model: {crack_mask_model}")

    payload = _compose_segmented_output(
        image_path=image_path,
        output_dir=str(sam_params.get("output_dir") or "results_sam"),
        detections=segmented_detections,
        overlay_alpha=float(sam_params.get("overlay_alpha") or 0.45),
        output_tag="hybrid_boxes",
    )
    if display_detections:
        payload["display_detections"] = display_detections
    return payload


def _run_sam_dino_workflow(
    ctx: WorkflowContext,
    *,
    sam_service_name: str,
    sam_getter: Callable[[], Any],
) -> InferenceResult:
    sam_params = dict(ctx.request.params.get("sam") or {})
    dino_params = dict(ctx.request.params.get("dino") or {})
    use_tiled_dino = bool(ctx.request.params.get("use_tiled_dino") if ctx.request.params.get("use_tiled_dino") is not None else True)
    tile_trigger_px = int(ctx.request.params.get("tile_trigger_px") or 512)
    target_labels = [str(label).strip() for label in (ctx.request.params.get("target_labels") or dino_params.get("text_queries") or []) if str(label).strip()]
    crack_mask_model = str(ctx.request.params.get("crack_mask_model") or "off").strip().lower()
    ctx.log("Warming up DINO...")
    ctx.call_service("dino", get_dino_service, "warmup", {"params": dino_params})
    ctx.log(f"Warming up {sam_service_name}...")
    ctx.call_service(sam_service_name, sam_getter, "warmup", {"params": sam_params})
    if crack_mask_model == "sam_lora" and sam_service_name != "sam_finetune":
        ctx.log("Warming up crack sam_finetune...")
        ctx.call_service("sam_finetune", get_sam_finetune_service, "warmup", {"params": dict(ctx.request.params.get("crack_sam") or {})})
    elif crack_mask_model == "unet":
        ctx.log("Warming up crack unet...")
        ctx.call_service("unet", get_unet_service, "warmup", {"params": dict(ctx.request.params.get("crack_unet") or {})})

    if ctx.request.image_paths:
        results = []
        total = len(ctx.request.image_paths)
        ctx.log(f"Batch {ctx.request.workflow}: {total} images")
        for idx, path in enumerate(ctx.request.image_paths):
            if ctx.stop_checker():
                return ctx.completed({"stopped": True})
            path = str(path)
            ctx.log(f"[{idx+1}/{total}] {path}")
            res = _run_sam_dino_single(
                ctx,
                image_path=path,
                dino_params=dino_params,
                sam_params=sam_params,
                sam_service_name=sam_service_name,
                sam_getter=sam_getter,
                use_tiled_dino=use_tiled_dino,
                tile_trigger_px=tile_trigger_px,
                target_labels=target_labels,
            )
            if res.get("stopped"):
                return ctx.completed({"stopped": True})
            if "image_path" not in res:
                res["image_path"] = path
            results.append(res)
        return ctx.completed({"batch_done": True, "results": results})

    payload = _run_sam_dino_single(
        ctx,
        image_path=str(ctx.request.image_path or ""),
        dino_params=dino_params,
        sam_params=sam_params,
        sam_service_name=sam_service_name,
        sam_getter=sam_getter,
        use_tiled_dino=use_tiled_dino,
        tile_trigger_px=tile_trigger_px,
        target_labels=target_labels,
    )
    return ctx.completed(payload)


def _run_unet_dino_single(
    ctx: WorkflowContext,
    *,
    image_path: str,
    unet_params: dict[str, Any],
    dino_params: dict[str, Any],
) -> dict[str, Any]:
    dino_res = ctx.call_service("dino", get_dino_service, "predict", {"image_path": image_path, "params": dino_params})
    if isinstance(dino_res, dict) and dino_res.get("stopped"):
        return {"stopped": True}
    detections = list(dino_res.get("detections") or [])
    if detections:
        ctx.partial({"image_path": image_path, "detections": detections}, message="DINO detections ready")
    if not detections:
        ctx.log("No objects detected. Skipping UNet.")
        return {"image_path": image_path, "detections": []}
    roi = _merge_detection_boxes(detections, pad=50)
    if roi is None:
        ctx.log("No valid boxes found from DINO.")
        return {"image_path": image_path, "detections": detections}
    ctx.log(f"Unified ROI: {roi}")
    unet_res = ctx.call_service("unet", get_unet_service, "run_rois", {"image_path": image_path, "params": unet_params, "rois": [roi]})
    final_det = _with_b64_mask(
        {
            "label": "Merged",
            "score": 1.0,
            "model_name": "UnetDino",
            "mask_path": unet_res.get("mask_path"),
            "box": list(roi),
        }
    )
    return {
        "image_path": image_path,
        "mask_path": unet_res.get("mask_path"),
        "overlay_path": unet_res.get("overlay_path"),
        "detections": [final_det],
    }


def _run_sam_tiled_single(
    ctx: WorkflowContext,
    *,
    image_path: str,
    dino_params: dict[str, Any],
    sam_params: dict[str, Any],
    target_labels: list[str],
    max_depth: int,
    min_box_px: int,
) -> dict[str, Any]:
    dino_res = ctx.call_service(
        "dino",
        get_dino_service,
        "recursive_detect",
        {
            "image_path": image_path,
            "params": dino_params,
            "target_labels": target_labels,
            "max_depth": max_depth,
            "min_box_px": min_box_px,
        },
    )
    if isinstance(dino_res, dict) and dino_res.get("stopped"):
        return {"stopped": True}
    display_detections = list(dino_res.get("display_detections") or dino_res.get("detections") or [])
    detections = list(dino_res.get("detections") or [])
    if display_detections:
        ctx.partial({"image_path": image_path, "detections": display_detections}, message="Recursive DINO detections ready")
    if not detections:
        ctx.log("No tiled DINO detections. Skipping SAM box prompting.")
        return {"image_path": image_path, "detections": [], "display_detections": display_detections}
    sam_boxes = display_detections or detections
    ctx.log(f"SAM box prompting on {len(sam_boxes)} DINO boxes...")
    sam_res = ctx.call_service(
        "sam",
        get_sam_service,
        "segment_boxes",
        {"image_path": image_path, "params": sam_params, "boxes": sam_boxes},
    )
    payload = _set_detection_model_name(dict(sam_res or {}), "SamTiled")
    if display_detections:
        payload["display_detections"] = display_detections
    return payload


def _run_isolate_single(
    ctx: WorkflowContext,
    *,
    image_path: str,
    sam_params: dict[str, Any],
    prompt: str,
    outside_value: int,
    crop_to_bbox: bool,
    action: str,
) -> dict[str, Any]:
    sam_res = ctx.call_service(
        "sam",
        get_sam_service,
        "predict",
        {"image_path": image_path, "params": sam_params},
    )
    if isinstance(sam_res, dict) and sam_res.get("stopped"):
        return {"stopped": True}
    payload = dict(sam_res or {})
    payload["prompt"] = str(prompt or "").strip()
    payload["isolate_action"] = str(action or "keep").strip().lower()
    payload = _set_detection_model_name(payload, "Isolate")
    payload = _set_detection_label(payload, prompt)
    mask_path = str(payload.get("mask_path") or "")
    isolate_path = _write_isolate_from_mask(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=str(payload.get("output_dir") or sam_params.get("output_dir") or os.path.dirname(mask_path)),
        outside_value=outside_value,
        crop_to_bbox=crop_to_bbox,
        action=action,
    )
    if isolate_path:
        payload["isolate_path"] = isolate_path
    return payload


def _run_isolate_dino_single(
    ctx: WorkflowContext,
    *,
    image_path: str,
    dino_params: dict[str, Any],
    sam_params: dict[str, Any],
    prompt: str,
    outside_value: int,
    crop_to_bbox: bool,
    action: str,
) -> dict[str, Any]:
    dino_res = ctx.call_service("dino", get_dino_service, "predict", {"image_path": image_path, "params": dino_params})
    if isinstance(dino_res, dict) and dino_res.get("stopped"):
        return {"stopped": True}
    detections = list(dino_res.get("detections") or [])
    if detections:
        ctx.partial({"image_path": image_path, "detections": detections}, message="DINO detections ready")
    if not detections:
        ctx.log("No DINO detections for isolate prompt.")
        return {"image_path": image_path, "prompt": str(prompt or "").strip(), "detections": []}
    sam_res = ctx.call_service(
        "sam",
        get_sam_service,
        "segment_boxes",
        {"image_path": image_path, "params": sam_params, "boxes": detections},
    )
    if isinstance(sam_res, dict) and sam_res.get("stopped"):
        return {"stopped": True}
    payload = dict(sam_res or {})
    payload["prompt"] = str(prompt or "").strip()
    payload["isolate_action"] = str(action or "keep").strip().lower()
    payload = _set_detection_model_name(payload, "Isolate")
    payload = _set_detection_label(payload, prompt)
    mask_path = str(payload.get("mask_path") or "")
    isolate_path = _write_isolate_from_mask(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=str(payload.get("output_dir") or sam_params.get("output_dir") or os.path.dirname(mask_path)),
        outside_value=outside_value,
        crop_to_bbox=crop_to_bbox,
        action=action,
    )
    if isolate_path:
        payload["isolate_path"] = isolate_path
    return payload


def _run_isolate_dino_tiled_single(
    ctx: WorkflowContext,
    *,
    image_path: str,
    dino_params: dict[str, Any],
    sam_params: dict[str, Any],
    prompt: str,
    outside_value: int,
    crop_to_bbox: bool,
    action: str,
) -> dict[str, Any]:
    target_labels = [part.strip() for part in str(prompt or "").split(",") if part.strip()]
    max_depth = int(dino_params.get("recursive_max_depth") or 3)
    min_box_px = int(dino_params.get("recursive_min_box_px") or 48)
    dino_res = ctx.call_service(
        "dino",
        get_dino_service,
        "recursive_detect",
        {
            "image_path": image_path,
            "params": dino_params,
            "target_labels": target_labels or ["object"],
            "max_depth": max_depth,
            "min_box_px": min_box_px,
        },
    )
    if isinstance(dino_res, dict) and dino_res.get("stopped"):
        return {"stopped": True}
    detections = list(dino_res.get("detections") or [])
    if detections:
        ctx.partial({"image_path": image_path, "detections": detections}, message="Recursive DINO detections ready")
    if not detections:
        ctx.log("No tiled DINO detections for isolate prompt.")
        return {"image_path": image_path, "prompt": str(prompt or "").strip(), "detections": []}
    sam_res = ctx.call_service(
        "sam",
        get_sam_service,
        "segment_boxes",
        {"image_path": image_path, "params": sam_params, "boxes": detections},
    )
    if isinstance(sam_res, dict) and sam_res.get("stopped"):
        return {"stopped": True}
    payload = dict(sam_res or {})
    payload["prompt"] = str(prompt or "").strip()
    payload["isolate_action"] = str(action or "keep").strip().lower()
    payload = _set_detection_model_name(payload, "Isolate")
    payload = _set_detection_label(payload, prompt)
    mask_path = str(payload.get("mask_path") or "")
    isolate_path = _write_isolate_from_mask(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=str(payload.get("output_dir") or sam_params.get("output_dir") or os.path.dirname(mask_path)),
        outside_value=outside_value,
        crop_to_bbox=crop_to_bbox,
        action=action,
    )
    if isolate_path:
        payload["isolate_path"] = isolate_path
    return payload


def _run_unet_dino(ctx: WorkflowContext) -> InferenceResult:
    unet_params = dict(ctx.request.params.get("unet") or {})
    dino_params = dict(ctx.request.params.get("dino") or {})
    ctx.log("Warming up DINO...")
    ctx.call_service("dino", get_dino_service, "warmup", {"params": dino_params})
    ctx.log("Warming up UNet...")
    ctx.call_service("unet", get_unet_service, "warmup", {"params": unet_params})

    if ctx.request.image_paths:
        results = []
        total = len(ctx.request.image_paths)
        ctx.log(f"Batch UNet + DINO: {total} images")
        for idx, path in enumerate(ctx.request.image_paths):
            if ctx.stop_checker():
                return ctx.completed({"stopped": True})
            path = str(path)
            ctx.log(f"[{idx+1}/{total}] {path}")
            res = _run_unet_dino_single(ctx, image_path=path, unet_params=unet_params, dino_params=dino_params)
            if res.get("stopped"):
                return ctx.completed({"stopped": True})
            if "image_path" not in res:
                res["image_path"] = path
            results.append(res)
        return ctx.completed({"batch_done": True, "results": results})

    payload = _run_unet_dino_single(
        ctx,
        image_path=str(ctx.request.image_path or ""),
        unet_params=unet_params,
        dino_params=dino_params,
    )
    return ctx.completed(payload)


def _run_sam_only(ctx: WorkflowContext) -> InferenceResult:
    params = dict(ctx.request.params.get("sam") or {})
    return _run_single_or_batch(ctx=ctx, service_name="sam", service_getter=get_sam_service, params=params)


def _run_sam_only_ft(ctx: WorkflowContext) -> InferenceResult:
    params = dict(ctx.request.params.get("sam") or {})
    return _run_single_or_batch(ctx=ctx, service_name="sam_finetune", service_getter=get_sam_finetune_service, params=params)


def _run_sam_tiled(ctx: WorkflowContext) -> InferenceResult:
    sam_params = dict(ctx.request.params.get("sam") or {})
    dino_params = dict(ctx.request.params.get("dino") or {})
    target_labels = list(ctx.request.params.get("target_labels") or ["crack"])
    max_depth = int(ctx.request.params.get("max_depth") or 3)
    min_box_px = int(ctx.request.params.get("min_box_px") or 48)
    ctx.log("Warming up DINO...")
    ctx.call_service("dino", get_dino_service, "warmup", {"params": dino_params})
    ctx.log("Warming up sam...")
    ctx.call_service("sam", get_sam_service, "warmup", {"params": sam_params})
    if ctx.request.image_paths:
        results = []
        total = len(ctx.request.image_paths)
        ctx.log(f"Batch SAM+DINO Tiled: {total} images")
        for idx, path in enumerate(ctx.request.image_paths):
            if ctx.stop_checker():
                return ctx.completed({"stopped": True})
            ctx.log(f"[{idx+1}/{total}] {path}")
            res = _run_sam_tiled_single(
                ctx,
                image_path=str(path),
                dino_params=dino_params,
                sam_params=sam_params,
                target_labels=target_labels,
                max_depth=max_depth,
                min_box_px=min_box_px,
            )
            if isinstance(res, dict) and "image_path" not in res:
                res["image_path"] = str(path)
            results.append(res)
        return ctx.completed({"batch_done": True, "results": results})
    payload = _run_sam_tiled_single(
        ctx,
        image_path=str(ctx.request.image_path or ""),
        dino_params=dino_params,
        sam_params=sam_params,
        target_labels=target_labels,
        max_depth=max_depth,
        min_box_px=min_box_px,
    )
    return ctx.completed(dict(payload or {}))


def _run_isolate(ctx: WorkflowContext) -> InferenceResult:
    sam_params = dict(ctx.request.params.get("sam") or {})
    dino_params = dict(ctx.request.params.get("dino") or {})
    isolate_mode = str(ctx.request.params.get("mode") or "sam_only").strip().lower()
    prompt = str(ctx.request.params.get("prompt") or "").strip()
    outside_value = int(ctx.request.params.get("outside_value") or 0)
    crop_to_bbox = bool(ctx.request.params.get("crop_to_bbox") or False)
    action = str(ctx.request.params.get("action") or "keep").strip().lower()
    use_tiled_dino = bool(ctx.request.params.get("use_tiled_dino") or False)
    tile_trigger_px = int(ctx.request.params.get("tile_trigger_px") or 512)
    if prompt:
        ctx.log(f'Isolate prompt: "{prompt}"')
    ctx.log(f"Isolate action: {action}")
    if isolate_mode == "dino_sam":
        ctx.log("Warming up dino...")
        ctx.call_service("dino", get_dino_service, "warmup", {"params": dino_params})
    ctx.log("Warming up sam...")
    ctx.call_service("sam", get_sam_service, "warmup", {"params": sam_params})
    if isolate_mode == "dino_sam":
        image_path = str(ctx.request.image_path or "")
        max_dim = _image_max_dim(image_path) if use_tiled_dino else 0
        if use_tiled_dino and max_dim > tile_trigger_px:
            ctx.log(f"Large image detected ({max_dim}px). Using tiled DINO before SAM.")
            payload = _run_isolate_dino_tiled_single(
                ctx,
                image_path=image_path,
                dino_params=dino_params,
                sam_params=sam_params,
                prompt=prompt,
                outside_value=outside_value,
                crop_to_bbox=crop_to_bbox,
                action=action,
            )
        else:
            payload = _run_isolate_dino_single(
                ctx,
                image_path=image_path,
                dino_params=dino_params,
                sam_params=sam_params,
                prompt=prompt,
                outside_value=outside_value,
                crop_to_bbox=crop_to_bbox,
                action=action,
            )
    else:
        payload = _run_isolate_single(
            ctx,
            image_path=str(ctx.request.image_path or ""),
            sam_params=sam_params,
            prompt=prompt,
            outside_value=outside_value,
            crop_to_bbox=crop_to_bbox,
            action=action,
        )
    return ctx.completed(dict(payload or {}))


_WORKFLOW_HANDLERS: dict[str, Callable[[WorkflowContext], InferenceResult]] = {
    "sam_dino": lambda ctx: _run_sam_dino_workflow(ctx, sam_service_name="sam", sam_getter=get_sam_service),
    "sam_dino_ft": lambda ctx: _run_sam_dino_workflow(ctx, sam_service_name="sam_finetune", sam_getter=get_sam_finetune_service),
    "sam_only": _run_sam_only,
    "sam_only_ft": _run_sam_only_ft,
    "sam_tiled": _run_sam_tiled,
    "isolate": _run_isolate,
    "unet_only": lambda ctx: _run_single_or_batch(
        ctx=ctx,
        service_name="unet",
        service_getter=get_unet_service,
        params=dict(ctx.request.params.get("unet") or {}),
    ),
    "unet_dino": _run_unet_dino,
}


def run_workflow(ctx: WorkflowContext) -> InferenceResult:
    workflow = str(ctx.request.workflow or "").strip().lower()
    handler = _WORKFLOW_HANDLERS.get(workflow)
    if handler is None:
        raise ValueError(f"Unknown workflow: {ctx.request.workflow}")
    return handler(ctx)
