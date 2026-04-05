from __future__ import annotations

from typing import Any

from dino.engine import DinoParams, DinoRunner
from inference_api.process_worker import WorkerProtocol, run_worker


_runner = DinoRunner()


def _params_from(obj: dict[str, Any]) -> DinoParams:
    data = dict(obj or {})
    roi = data.get("roi_box")
    if isinstance(roi, list) and len(roi) == 4:
        data["roi_box"] = tuple(int(x) for x in roi)
    queries = data.get("text_queries")
    if isinstance(queries, str):
        data["text_queries"] = [part.strip() for part in queries.split(",") if part.strip()]
    background_labels = data.get("prototype_background_labels")
    if isinstance(background_labels, str):
        data["prototype_background_labels"] = [part.strip() for part in background_labels.split(",") if part.strip()]
    return DinoParams(**data)


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
            log(f"Batch DINO: {total} images")
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

    if normalized == "recursive_detect":
        def _job():
            image_path = str(params.get("image_path") or "").strip()
            p = _params_from(params.get("params") or {})
            target_labels = [str(x) for x in (params.get("target_labels") or ["crack"])]
            max_depth = int(params.get("max_depth") or 3)
            min_box_px = int(params.get("min_box_px") or 48)
            return _runner.predict_recursive(
                image_path,
                p,
                target_labels=target_labels,
                max_depth=max_depth,
                min_box_px=min_box_px,
                stop_checker=proto.stop_checker(call_id),
                log_fn=proto.log_fn(call_id),
            )
        proto.spawn_job(call_id, _job)
        return

    if normalized == "rank_boxes":
        def _job():
            image_path = str(params.get("image_path") or "").strip()
            p = _params_from(params.get("params") or {})
            return _runner.rank_boxes(
                image_path,
                p,
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
