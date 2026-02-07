from __future__ import annotations

from typing import Any

from predict.process_worker import WorkerProtocol, run_worker
import scipy
import scipy.signal
import scipy.spatial


def _dispatch(proto: WorkerProtocol, call_id: int, method: str, params: dict[str, Any]) -> None:
    m = str(method or "").strip().lower()

    from sam_dino.runner import SamDinoParams, SamDinoRunner

    # singleton inside process
    global _runner  # type: ignore[declared-but-unused]
    try:
        _runner
    except NameError:
        _runner = SamDinoRunner()

    if m == "warmup":
        def _job():
            p = SamDinoParams(**(params.get("params") or params))
            _runner._ensure_models(p, log_fn=proto.log_fn(call_id))  # type: ignore[attr-defined]
            return {"ok": True}

        proto.spawn_job(call_id, _job)
        return

    if m == "run":
        def _job():
            image_path = str(params.get("image_path") or "").strip()
            p = SamDinoParams(**(params.get("params") or {}))
            return _runner.run(
                image_path,
                p,
                stop_checker=proto.stop_checker(call_id),
                log_fn=proto.log_fn(call_id),
            )

        proto.spawn_job(call_id, _job)
        return

    if m == "isolate":
        def _job():
            image_path = str(params.get("image_path") or "").strip()
            p = SamDinoParams(**(params.get("params") or {}))
            target_labels = params.get("target_labels") or []
            if not isinstance(target_labels, list):
                raise TypeError("target_labels must be a list")
            outside_value = int(params.get("outside_value") or 0)
            crop_to_bbox = bool(params.get("crop_to_bbox") or False)
            return _runner.run_isolate(
                image_path,
                p,
                target_labels=[str(x) for x in target_labels],
                outside_value=outside_value,
                crop_to_bbox=crop_to_bbox,
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
            p = SamDinoParams(**(params.get("params") or {}))
            total = len(image_paths)
            log = proto.log_fn(call_id)
            results = []
            log(f"Batch SAM+DINO: {total} images")
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

    proto.spawn_job(call_id, lambda: {"error": f"Unknown method: {method}"})


def main() -> int:
    return run_worker(_dispatch)


if __name__ == "__main__":
    raise SystemExit(main())
