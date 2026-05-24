from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image

from object_detection.dino.client import get_dino_service
from object_detection.dino.engine import default_gdino_checkpoint

from pineline.sam_gdino.step3_gdino_damage.overlay import write_overlay
from pineline.sam_gdino.step3_gdino_damage.prompts import (
    PromptGroup,
    combined_queries,
    match_group_for_label,
)
from pineline.sam_gdino.step3_gdino_damage.rgb_export import rgba_to_rgb_on_black
from pineline.sam_gdino.step3_gdino_damage.store import (
    ensure_schema,
    insert_damage_detections,
    insert_prompt_groups,
    insert_run_info,
    load_step2_crops,
)


def resolve_checkpoint(raw: str | None) -> str:
    ckpt = str(raw or "").strip() or str(default_gdino_checkpoint() or "").strip()
    if not ckpt:
        raise RuntimeError("No GroundingDINO checkpoint available.")
    return ckpt


def _detect_one(
    service,
    image_path: Path,
    *,
    checkpoint: str,
    queries: list[str],
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    device: str,
    tiled_threshold: int,
    tile_scales: list[str],
    recursive_max_depth: int,
    recursive_min_box_px: int,
) -> list[dict]:
    with Image.open(image_path) as im:
        w, h = im.size
    max_dim = max(int(w), int(h))
    params = {
        "gdino_checkpoint": checkpoint,
        "gdino_config_id": "auto",
        "text_queries": queries,
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "max_dets": int(max_dets),
        "device": device,
        "recursive_tile_scales": tile_scales,
    }
    if max_dim > tiled_threshold:
        result = service.call(
            "recursive_detect",
            {
                "image_path": str(image_path),
                "params": params,
                "target_labels": queries,
                "max_depth": int(recursive_max_depth),
                "min_box_px": int(recursive_min_box_px),
            },
        )
    else:
        result = service.call("predict", {"image_path": str(image_path), "params": params})
    return list(result.get("detections") or [])


def run_step3(
    *,
    step2_db: Path,
    db_path: Path,
    rgb_dir: Path,
    overlay_dir: Path | None,
    summary_csv: Path | None,
    prompt_groups: list[PromptGroup],
    source_run_id: str | None = "latest",
    checkpoint: str | None = None,
    device: str = "auto",
    box_threshold: float = 0.16,
    text_threshold: float = 0.16,
    max_dets: int = 80,
    tiled_threshold: int = 1600,
    tile_scales: list[str] | None = None,
    recursive_max_depth: int = 2,
    recursive_min_box_px: int = 32,
    max_box_area_ratio: float = 0.50,
    write_overlays: bool = True,
    log: Callable[[str], None] | None = None,
    stop_checker: Callable[[], bool] | None = None,
) -> dict:
    log = log or (lambda s: print(s, flush=True))
    if not prompt_groups:
        raise ValueError("Step 3 requires at least one prompt group.")
    tile_scales = tile_scales or ["small", "medium", "large"]
    ckpt = resolve_checkpoint(checkpoint)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if write_overlays and overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(str(step2_db))
    crops = load_step2_crops(src_conn, source_run_id=source_run_id)
    src_conn.close()
    if not crops:
        raise RuntimeError(f"No bridge crops found in {step2_db} run={source_run_id}")
    step2_run_id = crops[0]["step2_run_id"]
    log(f"Loaded {len(crops)} crops from step2 run {step2_run_id}.")

    queries = combined_queries(prompt_groups)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_rows: list[dict] = []
    detection_count = 0

    conn = sqlite3.connect(str(db_path))
    service = get_dino_service()
    try:
        ensure_schema(conn)
        insert_prompt_groups(conn, run_id, prompt_groups)

        for idx, c in enumerate(crops, start=1):
            if stop_checker is not None and stop_checker():
                log("Stop requested; aborting step3.")
                break

            crop_rgba = Path(c["crop_path"])
            parent_image_id = c["parent_image_id"]
            rgb_path = rgb_dir / f"{parent_image_id}.png"
            try:
                w, h = rgba_to_rgb_on_black(crop_rgba, rgb_path)
            except Exception as exc:
                log(f"[{idx}/{len(crops)}] {parent_image_id}: rgb export error: {exc}")
                continue

            try:
                detections = _detect_one(
                    service, rgb_path,
                    checkpoint=ckpt, queries=queries,
                    box_threshold=box_threshold, text_threshold=text_threshold,
                    max_dets=max_dets, device=device,
                    tiled_threshold=tiled_threshold, tile_scales=tile_scales,
                    recursive_max_depth=recursive_max_depth,
                    recursive_min_box_px=recursive_min_box_px,
                )
            except Exception as exc:
                log(f"[{idx}/{len(crops)}] {parent_image_id}: detect error: {exc}")
                continue

            rows: list[dict] = []
            tagged_detections: list[dict] = []
            image_area = max(1.0, float(w) * float(h))
            drop_too_large = 0
            for det_idx, det in enumerate(detections):
                box = det.get("box")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in box]
                except (TypeError, ValueError):
                    continue
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                if (bw * bh) / image_area > float(max_box_area_ratio):
                    drop_too_large += 1
                    continue
                label = str(det.get("label") or "")
                score = float(det.get("score") or 0.0)
                match = match_group_for_label(label, prompt_groups)
                if match is None:
                    continue
                group_id, group_name = match
                rows.append(
                    {
                        "run_id": run_id,
                        "parent_image_id": parent_image_id,
                        "det_idx": det_idx,
                        "group_id": group_id,
                        "group_name": group_name,
                        "label": label,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score,
                    }
                )
                tagged_detections.append(
                    {"box": [x1, y1, x2, y2], "score": score,
                     "label": label, "group_name": group_name}
                )

            if rows:
                insert_damage_detections(conn, rows)
                detection_count += len(rows)
                if write_overlays and overlay_dir is not None:
                    try:
                        write_overlay(
                            image_path=rgb_path,
                            detections=tagged_detections,
                            out_path=overlay_dir / f"{parent_image_id}.png",
                        )
                    except Exception as exc:
                        log(f"  overlay failed: {exc}")
                for row in rows:
                    summary_rows.append(
                        {
                            "parent_image_id": parent_image_id,
                            "det_idx": row["det_idx"],
                            "group_name": row["group_name"],
                            "label": row["label"],
                            "x1": row["x1"], "y1": row["y1"],
                            "x2": row["x2"], "y2": row["y2"],
                            "score": row["score"],
                        }
                    )
            conn.commit()
            log(
                f"[{idx}/{len(crops)}] {parent_image_id}: raw={len(detections)} "
                f"matched={len(rows)} drop_large={drop_too_large} (image size {w}x{h})"
            )

        insert_run_info(
            conn,
            {
                "run_id": run_id,
                "created_at_utc": run_id,
                "step2_db": str(step2_db),
                "step2_run_id": step2_run_id,
                "checkpoint": ckpt,
                "device": device,
                "box_threshold": float(box_threshold),
                "text_threshold": float(text_threshold),
                "max_dets": int(max_dets),
                "crop_count": len(crops),
                "detection_count": detection_count,
            },
        )
    finally:
        try:
            service.close()
        except Exception:
            pass
        conn.close()

    if summary_csv is not None and summary_rows:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "parent_image_id", "det_idx", "group_name", "label",
                    "x1", "y1", "x2", "y2", "score",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    log(
        f"step3 done. run_id={run_id} crops={len(crops)} dets={detection_count} db={db_path}"
    )
    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "rgb_dir": str(rgb_dir),
        "overlay_dir": str(overlay_dir) if (write_overlays and overlay_dir) else None,
        "crop_count": len(crops),
        "detection_count": detection_count,
    }
