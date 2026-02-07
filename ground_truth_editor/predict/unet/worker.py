from __future__ import annotations

from typing import Any

from predict.process_worker import WorkerProtocol, run_worker
from predict.unet.runner import UnetParams, UnetRunner
import torch
import numpy as np
import PIL
import PIL.Image


_runner = UnetRunner()


def _unet_params_from(obj: dict[str, Any]) -> UnetParams:
    data: dict[str, Any] = dict(obj or {})
    roi = data.get("roi_box")
    if isinstance(roi, list) and len(roi) == 4:
        data["roi_box"] = tuple(int(x) for x in roi)
    return UnetParams(**data)


def _dispatch(proto: WorkerProtocol, call_id: int, method: str, params: dict[str, Any]) -> None:
    m = str(method or "").strip().lower()

    if m == "warmup":
        def _job():
            p = _unet_params_from(params.get("unet") or params)
            _runner.ensure_model_loaded(p, log_fn=proto.log_fn(call_id))
            return {"ok": True}

        proto.spawn_job(call_id, _job)
        return

    if m == "run":
        def _job():
            image_path = str(params.get("image_path") or "").strip()
            p = _unet_params_from(params.get("params") or {})
            return _runner.run(
                image_path,
                p,
                stop_checker=proto.stop_checker(call_id),
                log_fn=proto.log_fn(call_id),
            )

        proto.spawn_job(call_id, _job)
        return

    if m == "batch_run":
        def _job():
            image_paths = params.get("image_paths") or []
            if not isinstance(image_paths, list):
                raise TypeError("image_paths must be a list")
            p = _unet_params_from(params.get("params") or {})
            total = len(image_paths)
            log = proto.log_fn(call_id)
            results = []
            log(f"Batch UNet: {total} images")
            for i, path in enumerate(image_paths):
                if proto.stop_checker(call_id)():
                    return {"stopped": True}
                log(f"[{i+1}/{total}] {path}")
                try:
                    res = _runner.run(
                        str(path),
                        p,
                        stop_checker=proto.stop_checker(call_id),
                        log_fn=None,
                    )
                    if isinstance(res, dict) and "image_path" not in res:
                        res["image_path"] = str(path)
                    results.append(res)
                except Exception as e:
                    log(f"Error processing {path}: {e}")
            return {"batch_done": True, "results": results}

        proto.spawn_job(call_id, _job)
        return

    if m == "run_rois":
        def _job():
            image_path = str(params.get("image_path") or "").strip()
            p = _unet_params_from(params.get("params") or {})
            rois_raw = params.get("rois") or []
            rois = []
            for r in rois_raw:
                if len(r) == 4:
                    rois.append(tuple(int(x) for x in r))
            
            return _runner.run_rois(
                image_path,
                p,
                rois,
                stop_checker=proto.stop_checker(call_id),
                log_fn=proto.log_fn(call_id),
            )
        proto.spawn_job(call_id, _job)
        return

    proto.spawn_job(call_id, lambda: {"error": f"Unknown method: {method}"})


def main() -> int:
    return run_worker(_dispatch)


if __name__ == "__main__":
    raise SystemExit(main())
