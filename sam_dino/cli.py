from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import json
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SAM + GroundingDINO from a JSON payload.")
    p.add_argument("--payload", required=True, help="Path to payload JSON.")
    p.add_argument("--output", required=True, help="Path to write result JSON.")
    return p


def _load_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError("Payload must be a JSON object.")
    return obj


def _print(msg: str) -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()

    args = _build_parser().parse_args(argv)
    payload = _load_payload(args.payload)

    mode = str(payload.get("mode") or "run").strip().lower()
    image_path = str(payload.get("image_path") or "").strip()
    params_dict = payload.get("params") or {}

    if mode not in {"run", "isolate"}:
        raise ValueError("payload.mode must be run|isolate")
    if not image_path:
        raise ValueError("payload.image_path is required")
    if not isinstance(params_dict, dict):
        raise TypeError("payload.params must be a JSON object")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from ground_truth_editor.predict_sam_dino import SamDinoParams, SamDinoRunner

    params = SamDinoParams(**params_dict)
    runner = SamDinoRunner()

    try:
        if mode == "run":
            result = runner.run(image_path, params, log_fn=_print)
        else:
            target_labels = payload.get("target_labels") or []
            outside_value = int(payload.get("outside_value") or 0)
            crop_to_bbox = bool(payload.get("crop_to_bbox") or False)
            if not isinstance(target_labels, list):
                raise TypeError("payload.target_labels must be a list of strings")
            result = runner.run_isolate(
                image_path,
                params,
                target_labels=[str(x) for x in target_labels],
                outside_value=outside_value,
                crop_to_bbox=crop_to_bbox,
                log_fn=_print,
            )
    except Exception as e:
        result = {
            "error": str(e),
            "error_type": e.__class__.__name__,
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

