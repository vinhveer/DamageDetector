from __future__ import annotations

from dataclasses import fields
from typing import Any

from inference_api.process_worker import WorkerProtocol, run_worker
from .engine import SamParams, SamRunner


_runner = SamRunner()


def _params_from(obj: dict[str, Any]) -> SamParams:
    data = dict(obj or {})
    roi = data.get("roi_box")
    if isinstance(roi, list) and len(roi) == 4:
        data["roi_box"] = tuple(int(x) for x in roi)
    allowed = {field.name for field in fields(SamParams)}
    data = {key: value for key, value in data.items() if key in allowed}
    return SamParams(**data)


def _dispatch(proto: WorkerProtocol, call_id: int, method: str, params: dict[str, Any]) -> None:
    normalized = str(method or "").strip().lower()

    if normalized == "warmup":
        def _job():
            p = _params_from(params.get("params") or params)
            _runner.ensure_model_loaded(p, log_fn=proto.log_fn(call_id))
            return {"ok": True}
        proto.spawn_job(call_id, _job)
        return

    if normalized == "predict":
        def _job():
            image_path = str(params.get("image_path") or "").strip()
            p = _params_from(params.get("params") or {})
            return _runner.predict(image_path, p, stop_checker=proto.stop_checker(call_id), log_fn=proto.log_fn(call_id))
        proto.spawn_job(call_id, _job)
        return

    if normalized == "predict_batch":
        def _job():
            image_paths = params.get("image_paths") or []
            if not isinstance(image_paths, list):
                raise TypeError("image_paths must be a list")
            p = _params_from(params.get("params") or {})
            log = proto.log_fn(call_id)
            total = len(image_paths)
            results = []
            log(f"Batch SAM Only: {total} images")
            for idx, path in enumerate(image_paths):
                if proto.stop_checker(call_id)():
                    return {"stopped": True}
                log(f"[{idx+1}/{total}] {path}")
                try:
                    result = _runner.predict(str(path), p, stop_checker=proto.stop_checker(call_id), log_fn=None)
                    if isinstance(result, dict) and "image_path" not in result:
                        result["image_path"] = str(path)
                    results.append(result)
                except Exception as exc:
                    log(f"Error processing {path}: {exc}")
            return {"batch_done": True, "results": results}
        proto.spawn_job(call_id, _job)
        return

    if normalized == "segment_boxes":
        def _job():
            image_path = str(params.get("image_path") or "").strip()
            boxes = params.get("boxes") or []
            if not isinstance(boxes, list):
                raise TypeError("boxes must be a list")
            p = _params_from(params.get("params") or {})
            return _runner.segment_boxes(image_path, p, boxes, stop_checker=proto.stop_checker(call_id), log_fn=proto.log_fn(call_id))
        proto.spawn_job(call_id, _job)
        return

    proto.spawn_job(call_id, lambda: {"error": f"Unknown method: {method}"})


def main() -> int:
    return run_worker(_dispatch)


if __name__ == "__main__":
    raise SystemExit(main())
