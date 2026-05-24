from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

from object_detection.dino.client import get_dino_service
from object_detection.dino.engine import default_gdino_checkpoint

from pineline.sam_gdino.step1_gdino_bridge.overlay import (
    overlay_path_for,
    write_overlay,
)
from pineline.sam_gdino.step1_gdino_bridge.store import (
    ensure_schema,
    insert_bridge_detections,
    insert_image,
    insert_run_info,
)
from pineline.sam_gdino.step1_gdino_bridge.threshold import (
    apply_dynamic_threshold,
    relabel,
)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(input_dir: Path, *, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTS]
    else:
        files = [p for p in input_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
    return sorted(files)


def image_id_for(input_dir: Path, image_path: Path) -> str:
    rel = image_path.resolve().relative_to(input_dir.resolve()).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    return digest


def resolve_checkpoint(raw: str | None) -> str:
    ckpt = str(raw or "").strip() or str(default_gdino_checkpoint() or "").strip()
    if not ckpt:
        raise RuntimeError(
            "No GroundingDINO checkpoint available. "
            "Pass --checkpoint or run `python setup.py download_models --name grounding_dino_base`."
        )
    return ckpt


def detect_bridge_one(
    service,
    image_path: Path,
    *,
    checkpoint: str,
    box_threshold: float,
    text_threshold: float,
    max_dets: int,
    device: str,
    tiled_threshold: int,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[dict], int, int]:
    """Run GDINO with prompt 'bridge.' Returns (detections, width, height)."""
    with Image.open(image_path) as pil_img:
        width, height = pil_img.size
    max_dim = max(width, height)

    params = {
        "gdino_checkpoint": checkpoint,
        "gdino_config_id": "auto",
        "text_queries": ["bridge."],
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "max_dets": int(max_dets),
        "device": device,
    }

    if max_dim > tiled_threshold:
        result = service.call(
            "recursive_detect",
            {
                "image_path": str(image_path),
                "params": params,
                "target_labels": ["bridge"],
                "max_depth": 1,
                "min_box_px": 32,
            },
            log_fn=log_fn,
        )
    else:
        result = service.call(
            "predict",
            {"image_path": str(image_path), "params": params},
            log_fn=log_fn,
        )

    detections = list(result.get("detections") or [])
    return detections, int(width), int(height)


def _box_to_xyxy(box) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def run_step1(
    *,
    input_dir: Path,
    db_path: Path,
    overlay_dir: Path | None = None,
    summary_csv: Path | None = None,
    checkpoint: str | None = None,
    device: str = "auto",
    box_threshold: float = 0.10,
    text_threshold: float = 0.20,
    score_floor: float = 0.20,
    top_k: int = 3,
    max_dets: int = 25,
    tiled_threshold: int = 1600,
    recursive_find: bool = True,
    limit: int = 0,
    write_overlays: bool = True,
    log: Callable[[str], None] | None = None,
    stop_checker: Callable[[], bool] | None = None,
) -> dict:
    log = log or (lambda s: print(s, flush=True))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if write_overlays and overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    ckpt = resolve_checkpoint(checkpoint)
    images = iter_images(input_dir, recursive=recursive_find)
    if int(limit or 0) > 0:
        images = images[: int(limit)]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    conn = sqlite3.connect(str(db_path))
    service = get_dino_service()
    kept_image_count = 0
    try:
        ensure_schema(conn)
        for idx, image_path in enumerate(images, start=1):
            if stop_checker is not None and stop_checker():
                log("Stop requested; aborting step1.")
                break
            image_id = image_id_for(input_dir, image_path)
            rel = image_path.resolve().relative_to(input_dir.resolve()).as_posix()
            try:
                detections, width, height = detect_bridge_one(
                    service,
                    image_path,
                    checkpoint=ckpt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    max_dets=max_dets,
                    device=device,
                    tiled_threshold=tiled_threshold,
                    log_fn=None,
                )
            except Exception as exc:
                log(f"[{idx}/{len(images)}] error {rel}: {exc}")
                continue

            kept, best_score = apply_dynamic_threshold(
                detections, score_floor=score_floor, top_k=top_k,
            )
            kept = relabel(kept, label="bridge")

            insert_image(
                conn,
                {
                    "run_id": run_id,
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "image_rel_path": rel,
                    "width": width,
                    "height": height,
                    "kept": 1 if kept else 0,
                    "best_score": float(best_score),
                    "raw_box_count": len(detections),
                    "kept_box_count": len(kept),
                },
            )

            rows: list[dict] = []
            for box_idx, det in enumerate(kept):
                xyxy = _box_to_xyxy(det.get("box"))
                if xyxy is None:
                    continue
                x1, y1, x2, y2 = xyxy
                rows.append(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "box_idx": box_idx,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": float(det.get("score") or 0.0),
                        "label": str(det.get("label") or "bridge"),
                    }
                )
            if rows:
                insert_bridge_detections(conn, rows)
                kept_image_count += 1
                if write_overlays and overlay_dir is not None:
                    try:
                        write_overlay(
                            image_path=image_path,
                            detections=kept,
                            out_path=overlay_path_for(overlay_dir, image_id, rel),
                        )
                    except Exception as exc:
                        log(f"  overlay failed for {rel}: {exc}")
                for row in rows:
                    summary_rows.append(
                        {
                            "image_rel_path": rel,
                            "image_id": image_id,
                            "box_idx": row["box_idx"],
                            "x1": row["x1"], "y1": row["y1"],
                            "x2": row["x2"], "y2": row["y2"],
                            "score": row["score"], "label": row["label"],
                        }
                    )
            conn.commit()
            log(
                f"[{idx}/{len(images)}] {rel}: raw={len(detections)} kept={len(kept)} "
                f"best={best_score:.3f}"
            )
        insert_run_info(
            conn,
            {
                "run_id": run_id,
                "created_at_utc": run_id,
                "input_dir": str(input_dir),
                "checkpoint": ckpt,
                "device": device,
                "box_threshold": float(box_threshold),
                "text_threshold": float(text_threshold),
                "score_floor": float(score_floor),
                "top_k": int(top_k),
                "image_count": len(images),
                "kept_image_count": kept_image_count,
            },
        )
    finally:
        try:
            service.close()
        except Exception:
            pass
        conn.close()

    if summary_csv is not None and summary_rows:
        import csv
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_rel_path", "image_id", "box_idx",
                    "x1", "y1", "x2", "y2", "score", "label",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    log(
        f"step1 done. run_id={run_id} images={len(images)} "
        f"kept_images={kept_image_count} db={db_path}"
        + (f" overlays={overlay_dir}" if (write_overlays and overlay_dir) else "")
        + (f" csv={summary_csv}" if summary_csv else "")
    )
    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "overlay_dir": str(overlay_dir) if (write_overlays and overlay_dir) else None,
        "summary_csv": str(summary_csv) if summary_csv else None,
        "image_count": len(images),
        "kept_image_count": kept_image_count,
    }
