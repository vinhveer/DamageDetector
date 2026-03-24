from __future__ import annotations

import base64
import os
from typing import Any, Callable

import cv2

from dino import get_dino_service
from inference_api.contracts import InferenceRequest, InferenceResult, JobEvent
from sam import get_sam_service
from sam_finetune import get_sam_finetune_service
from unet import get_unet_service


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


def _write_isolate_from_mask(
    *,
    image_path: str,
    mask_path: str,
    output_dir: str,
    outside_value: int,
    crop_to_bbox: bool,
) -> str | None:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    keep = mask > 0
    isolate = image.copy()
    isolate[~keep] = int(outside_value)
    if crop_to_bbox:
        ys, xs = keep.nonzero()
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            isolate = isolate[y1:y2, x1:x2]
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "crop_isolate" if crop_to_bbox else "isolate"
    isolate_path = os.path.join(output_dir, f"{base}_{suffix}.png")
    if not cv2.imwrite(isolate_path, isolate):
        return None
    return isolate_path


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
) -> dict[str, Any]:
    dino_res = ctx.call_service("dino", get_dino_service, "predict", {"image_path": image_path, "params": dino_params})
    if isinstance(dino_res, dict) and dino_res.get("stopped"):
        return {"stopped": True}
    detections = list(dino_res.get("detections") or [])
    if detections:
        ctx.partial({"image_path": image_path, "detections": detections}, message="DINO detections ready")
    if not detections:
        ctx.log("No DINO detections. Skipping SAM box prompting.")
        return {"image_path": image_path, "detections": []}
    sam_res = ctx.call_service(
        sam_service_name,
        sam_getter,
        "segment_boxes",
        {"image_path": image_path, "params": sam_params, "boxes": detections},
    )
    return dict(sam_res or {})


def _run_sam_dino_workflow(
    ctx: WorkflowContext,
    *,
    sam_service_name: str,
    sam_getter: Callable[[], Any],
) -> InferenceResult:
    sam_params = dict(ctx.request.params.get("sam") or {})
    dino_params = dict(ctx.request.params.get("dino") or {})
    ctx.log("Warming up DINO...")
    ctx.call_service("dino", get_dino_service, "warmup", {"params": dino_params})
    ctx.log(f"Warming up {sam_service_name}...")
    ctx.call_service(sam_service_name, sam_getter, "warmup", {"params": sam_params})

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
    detections = list(dino_res.get("detections") or [])
    if detections:
        ctx.partial({"image_path": image_path, "detections": detections}, message="Recursive DINO detections ready")
    if not detections:
        ctx.log("No tiled DINO detections. Skipping SAM box prompting.")
        return {"image_path": image_path, "detections": []}
    sam_res = ctx.call_service(
        "sam",
        get_sam_service,
        "segment_boxes",
        {"image_path": image_path, "params": sam_params, "boxes": detections},
    )
    return _set_detection_model_name(dict(sam_res or {}), "SamTiled")


def _run_isolate_single(
    ctx: WorkflowContext,
    *,
    image_path: str,
    dino_params: dict[str, Any],
    sam_params: dict[str, Any],
    target_labels: list[str],
    outside_value: int,
    crop_to_bbox: bool,
) -> dict[str, Any]:
    dino_res = ctx.call_service("dino", get_dino_service, "predict", {"image_path": image_path, "params": dino_params})
    if isinstance(dino_res, dict) and dino_res.get("stopped"):
        return {"stopped": True}
    detections = _filter_detections_by_labels(list(dino_res.get("detections") or []), target_labels)
    if detections:
        ctx.partial({"image_path": image_path, "detections": detections}, message="DINO detections ready")
    if not detections:
        ctx.log("No matching DINO detections for isolate.")
        return {"image_path": image_path, "detections": []}
    sam_res = ctx.call_service(
        "sam",
        get_sam_service,
        "segment_boxes",
        {"image_path": image_path, "params": sam_params, "boxes": detections},
    )
    payload = dict(sam_res or {})
    payload = _set_detection_model_name(payload, "Isolate")
    mask_path = str(payload.get("mask_path") or "")
    isolate_path = _write_isolate_from_mask(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=str(payload.get("output_dir") or sam_params.get("output_dir") or os.path.dirname(mask_path)),
        outside_value=outside_value,
        crop_to_bbox=crop_to_bbox,
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


def run_workflow(ctx: WorkflowContext) -> InferenceResult:
    workflow = str(ctx.request.workflow or "").strip().lower()
    if workflow == "sam_dino":
        return _run_sam_dino_workflow(ctx, sam_service_name="sam", sam_getter=get_sam_service)
    if workflow == "sam_dino_ft":
        return _run_sam_dino_workflow(ctx, sam_service_name="sam_finetune", sam_getter=get_sam_finetune_service)
    if workflow == "sam_only":
        params = dict(ctx.request.params.get("sam") or {})
        return _run_single_or_batch(ctx=ctx, service_name="sam", service_getter=get_sam_service, params=params)
    if workflow == "sam_only_ft":
        params = dict(ctx.request.params.get("sam") or {})
        return _run_single_or_batch(ctx=ctx, service_name="sam_finetune", service_getter=get_sam_finetune_service, params=params)
    if workflow == "sam_tiled":
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
    if workflow == "isolate":
        sam_params = dict(ctx.request.params.get("sam") or {})
        dino_params = dict(ctx.request.params.get("dino") or {})
        target_labels = list(ctx.request.params.get("target_labels") or [])
        outside_value = int(ctx.request.params.get("outside_value") or 0)
        crop_to_bbox = bool(ctx.request.params.get("crop_to_bbox") or False)
        ctx.log("Warming up DINO...")
        ctx.call_service("dino", get_dino_service, "warmup", {"params": dino_params})
        ctx.log("Warming up sam...")
        ctx.call_service("sam", get_sam_service, "warmup", {"params": sam_params})
        payload = _run_isolate_single(
            ctx,
            image_path=str(ctx.request.image_path or ""),
            dino_params=dino_params,
            sam_params=sam_params,
            target_labels=target_labels,
            outside_value=outside_value,
            crop_to_bbox=crop_to_bbox,
        )
        return ctx.completed(dict(payload or {}))
    if workflow == "unet_only":
        params = dict(ctx.request.params.get("unet") or {})
        return _run_single_or_batch(ctx=ctx, service_name="unet", service_getter=get_unet_service, params=params)
    if workflow == "unet_dino":
        return _run_unet_dino(ctx)
    raise ValueError(f"Unknown workflow: {ctx.request.workflow}")
