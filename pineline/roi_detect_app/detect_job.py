"""Standalone ROI detection job.

Runs as an independent OS process (launched by the app via subprocess in its
own session) so the GUI can hard-cancel it: killing this process group also
kills the GroundingDINO worker subprocesses it spawns.

Protocol:
  stdin  : nothing
  stdout : human-readable log lines (one per line, flushed)
  result : written to --out-json as a JSON list of detection rows
  exit   : 0 on success, non-zero on failure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _become_session_leader() -> None:
    """Detach into a new session so the GUI can kill the whole process group.

    Killing this group also terminates the GroundingDINO worker subprocesses
    spawned later by MultiDetector.
    """
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass


def _log(message: str) -> None:
    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    _become_session_leader()
    parser = argparse.ArgumentParser(description="ROI detection job (standalone process).")
    parser.add_argument("--image", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--rois-json", required=True, help="JSON file: list of [x1,y1,x2,y2] in image coords.")
    parser.add_argument("--tmp-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--stabledino-checkpoint", default="")
    args = parser.parse_args(argv)

    from PIL import Image

    from pineline.common.detection import MultiDetector, default_detection_config
    from pineline.lib.step_gdino_detect.prompts import (
        combined_queries,
        default_prompt_groups,
        match_group_for_label,
    )

    rois = json.loads(Path(args.rois_json).read_text(encoding="utf-8"))
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Start detect: {args.detector}, conf={args.conf:.3f}, rois={len(rois)}")
    cfg = default_detection_config(
        models=args.detector,
        box_threshold=args.conf,
        text_threshold=args.conf,
        stabledino_conf=args.conf,
        stabledino_checkpoint=(args.stabledino_checkpoint or None),
        max_dets=500,
        tiled_threshold=256,
        tile_scales=("small", "medium", "large"),
        min_box_px=4,
        device="auto",
    )
    detector = MultiDetector(cfg, log=_log)
    groups = default_prompt_groups()
    queries = combined_queries(groups)
    names = [g.name for g in groups]

    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    rows: list[dict] = []

    try:
        for roi_index, roi in enumerate(rois, start=1):
            x1 = max(0, int(roi[0]))
            y1 = max(0, int(roi[1]))
            x2 = min(width, int(roi[2]))
            y2 = min(height, int(roi[3]))
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            _log(f"ROI {roi_index}/{len(rois)}: crop x={x1} y={y1} w={w} h={h}")
            crop_path = tmp_dir / f"roi_{roi_index:03d}.png"
            image.crop((x1, y1, x1 + w, y1 + h)).save(str(crop_path))
            detections = detector.detect(crop_path, width=w, height=h, queries=queries, names=names)
            matched = 0
            for det in detections:
                box = det.get("box") or []
                if len(box) != 4:
                    continue
                match = match_group_for_label(str(det.get("label") or ""), groups)
                if match is None:
                    continue
                _, group_name = match
                bx1, by1, bx2, by2 = [float(v) for v in box]
                rows.append({
                    "roi_index": roi_index,
                    "detector_name": str(det.get("detector_name") or args.detector),
                    "group_name": group_name,
                    "label": group_name,
                    "score": float(det.get("score") or 0.0),
                    "x1": bx1 + x1,
                    "y1": by1 + y1,
                    "x2": bx2 + x1,
                    "y2": by2 + y1,
                })
                matched += 1
            _log(f"ROI {roi_index}: raw={len(detections)} matched={matched}")
            _log(f"PROGRESS {roi_index}/{len(rois)}")
    finally:
        try:
            detector.close()
        except Exception:
            pass

    Path(args.out_json).write_text(json.dumps(rows), encoding="utf-8")
    _log(f"Done: {len(rows)} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
