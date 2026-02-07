from __future__ import annotations

from typing import Any

from predict.process_worker import WorkerProtocol, run_worker


def _dispatch(proto: WorkerProtocol, call_id: int, method: str, params: dict[str, Any]) -> None:
    m = str(method or "").strip().lower()

    if m == "ping":
        proto.spawn_job(call_id, lambda: {"ok": True})
        return

    # Minimal SAM singleton: warmup/load only. (Not used by UI yet)
    if m == "warmup":
        def _job():
            import torch
            from segment_anything import SamPredictor  # type: ignore
            from sam_dino.pipeline import apply_delta_to_sam, load_sam_model, resolve_best_delta_checkpoint

            sam_ckpt = str(params.get("sam_checkpoint") or "").strip()
            sam_type = str(params.get("sam_model_type") or "auto")
            delta_type = str(params.get("delta_type") or "none").strip().lower()
            delta_ckpt = str(params.get("delta_checkpoint") or "auto").strip()
            middle_dim = int(params.get("middle_dim") or 32)
            scaling_factor = float(params.get("scaling_factor") or 0.2)
            rank = int(params.get("rank") or 4)
            device = str(params.get("device") or "auto").strip().lower()

            from device_utils import select_device_str

            dev = select_device_str(device, torch=torch)
            log = proto.log_fn(call_id)
            log("Loading SAM checkpoint...")
            sam, used = load_sam_model(sam_ckpt, sam_type)
            if delta_type != "none":
                delta_path = resolve_best_delta_checkpoint(delta_type, delta_ckpt)
                if delta_path is not None:
                    log("Applying delta to SAM...")
                    apply_delta_to_sam(
                        sam=sam,
                        delta_type=delta_type,
                        delta_ckpt_path=str(delta_path),
                        middle_dim=middle_dim,
                        scaling_factor=scaling_factor,
                        rank=rank,
                    )
            sam.to(device=dev)

            global _predictor  # type: ignore[declared-but-unused]
            _predictor = SamPredictor(sam)
            return {"ok": True, "device": dev}

        proto.spawn_job(call_id, _job)
        return

    proto.spawn_job(call_id, lambda: {"error": f"Unknown method: {method}"})


def main() -> int:
    return run_worker(_dispatch)


if __name__ == "__main__":
    raise SystemExit(main())

