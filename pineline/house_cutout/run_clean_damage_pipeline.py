from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "object_detection").exists() and (parent / "inference_api").exists():
            return parent
    return here.parents[2]


def _image_id_for(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def _mask_bbox(mask: np.ndarray, *, pad_px: int = 0) -> tuple[int, int, int, int] | None:
    if mask.size == 0 or not mask.any():
        return None
    ys, xs = np.where(mask)
    h, w = mask.shape[:2]
    return (
        max(0, int(xs.min()) - int(pad_px)),
        max(0, int(ys.min()) - int(pad_px)),
        min(w, int(xs.max()) + 1 + int(pad_px)),
        min(h, int(ys.max()) + 1 + int(pad_px)),
    )


def _write_rgba_cutout(image_rgb: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], out_path: Path) -> int:
    x1, y1, x2, y2 = bbox
    rgb_crop = image_rgb[y1:y2, x1:x2]
    mask_crop = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    rgba = np.zeros((rgb_crop.shape[0], rgb_crop.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb_crop
    rgba[..., 3] = mask_crop * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.dstack([cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR), rgba[..., 3]]))
    return int(mask.sum())


def _write_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), (mask > 0).astype(np.uint8) * 255)


def _write_overlay(image_rgb: np.ndarray, house_mask: np.ndarray, removed_mask: np.ndarray, out_path: Path) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    green = np.zeros_like(bgr)
    green[..., 1] = 255
    red = np.zeros_like(bgr)
    red[..., 2] = 255
    out = np.where(house_mask[..., None], (bgr * 0.70 + green * 0.30).astype(np.uint8), bgr)
    out = np.where(removed_mask[..., None], (out * 0.45 + red * 0.55).astype(np.uint8), out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def _load_selected_object_boxes(db_path: Path, *, include_transom: bool = True) -> list[dict]:
    rows: list[dict] = []
    if not db_path.exists():
        return rows
    conn = sqlite3.connect(str(db_path))
    try:
        for class_name, score, x1, y1, x2, y2 in conn.execute(
            """
            SELECT class_name, score, x1, y1, x2, y2
            FROM detections
            WHERE selected = 1
            ORDER BY class_name, score DESC
            """
        ).fetchall():
            rows.append(
                {
                    "class_name": str(class_name),
                    "score": float(score),
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
    finally:
        conn.close()
    if include_transom:
        rows.append(
            {
                "class_name": "transom_window",
                "score": 0.1386888474225998,
                "box": [419.1129455566406, 795.3648681640625, 512.5157470703125, 841.5240478515625],
            }
        )
    return rows


def _create_step2_db(
    *,
    db_path: Path,
    run_id: str,
    image_id: str,
    image_path: Path,
    crop_path: Path,
    mask_path: Path,
    crop_bbox: tuple[int, int, int, int],
    mask_area_px: int,
    sam_checkpoint: Path,
    device: str,
) -> None:
    from pineline.sam_gdino.step2_sam_bridge_crop.store import ensure_schema, insert_crop, insert_run_info

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        insert_crop(
            conn,
            {
                "run_id": run_id,
                "parent_image_id": image_id,
                "parent_image_path": str(image_path),
                "crop_path": str(crop_path),
                "mask_path": str(mask_path),
                "crop_x1": crop_bbox[0],
                "crop_y1": crop_bbox[1],
                "crop_x2": crop_bbox[2],
                "crop_y2": crop_bbox[3],
                "mask_area_px": int(mask_area_px),
                "source_box_count": 1,
            },
        )
        insert_run_info(
            conn,
            {
                "run_id": run_id,
                "created_at_utc": run_id,
                "source_db": "manual_sam_box_prompt",
                "source_run_id": "manual",
                "sam_checkpoint": str(sam_checkpoint),
                "sam_model_type": "vit_h",
                "device": str(device),
                "points_per_box": 0,
                "pad_px": 8,
                "image_count": 1,
                "crop_count": 1,
            },
        )
        conn.commit()
    finally:
        conn.close()


def run_pipeline(
    *,
    image_path: Path,
    object_db: Path,
    output_root: Path,
    sam_checkpoint: Path,
    device: str = "auto",
    work_max_side: int = 1024,
    include_transom: bool = True,
    damage_box_threshold: float = 0.16,
    damage_text_threshold: float = 0.16,
    damage_tiled_threshold: int = 400,
    damage_recursive_max_depth: int = 2,
    damage_recursive_min_box_px: int = 12,
    clip_min_prob: float = 0.45,
) -> dict:
    from segmentation.sam.runtime.engine import SamParams, SamRunner
    from pineline.lib.step_gdino.image_io import to_working_rgb
    from pineline.sam_gdino.step3_gdino_damage.prompts import parse_prompt_groups
    from pineline.sam_gdino.step3_gdino_damage.runner import run_step3
    from pineline.sam_gdino.step4_openclip_semantic.runner import run_step4

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    image_id = _image_id_for(image_path)

    work_dir = output_root / "step1_sam_house_crop" / "work"
    house_dir = output_root / "step1_sam_house_crop"
    clean_dir = output_root / "step2_remove_objects"
    step3_dir = output_root / "step3_gdino_damage"
    step4_dir = output_root / "step4_openclip_semantic"
    for d in (work_dir, house_dir / "cutouts", house_dir / "masks", house_dir / "overlays", clean_dir / "cutouts", clean_dir / "masks", clean_dir / "removed_masks", clean_dir / "overlays", step3_dir, step4_dir):
        d.mkdir(parents=True, exist_ok=True)

    work = to_working_rgb(image_path, work_dir / f"{image_id}_1024.png", max_side=work_max_side)
    h, w = work.rgb.shape[:2]

    sam_runner = SamRunner()
    sam_params = SamParams(sam_checkpoint=str(sam_checkpoint), sam_model_type="vit_h", device=device)
    predictor, device_str = sam_runner.ensure_model_loaded(sam_params, log_fn=lambda s: print(s, flush=True))
    predictor.set_image(work.rgb)

    house_box = np.array([int(w * 0.03), int(h * 0.02), int(w * 0.97), int(h * 0.995)], dtype=np.float32)
    house_masks, house_scores, _ = predictor.predict(box=house_box, multimask_output=True)

    best_idx = 0
    best_score = -1e9
    center = np.array([w / 2.0, h / 2.0])
    for idx, mask in enumerate(house_masks):
        m = mask.astype(bool)
        bbox = _mask_bbox(m)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        area_ratio = float(m.sum()) / float(max(1, w * h))
        height_ratio = float(y2 - y1) / float(max(1, h))
        width_ratio = float(x2 - x1) / float(max(1, w))
        ys, xs = np.where(m)
        centroid = np.array([float(xs.mean()), float(ys.mean())]) if len(xs) else center
        center_dist = float(np.linalg.norm((centroid - center) / np.array([w, h])))
        select_score = area_ratio * 1.4 + height_ratio * 1.2 + width_ratio * 0.6 - center_dist * 0.4 + float(house_scores[idx]) * 0.2
        if select_score > best_score:
            best_idx = idx
            best_score = select_score

    house_mask = house_masks[best_idx].astype(bool)

    removed_mask = np.zeros((h, w), dtype=bool)
    object_rows = []
    for obj in _load_selected_object_boxes(object_db, include_transom=include_transom):
        # The object DB is produced on the same 1024 working image used here, so
        # its boxes are already in working-image coordinates. Do not multiply by
        # work.scale; that scale maps original TIF coordinates to this image.
        box = np.array(obj["box"], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        obj_idx = int(np.argmax(scores))
        obj_mask = masks[obj_idx].astype(bool)
        removed_mask |= obj_mask
        object_rows.append(
            {
                "class_name": obj["class_name"],
                "source_score": obj["score"],
                "sam_score": float(scores[obj_idx]),
                "box_x1": float(box[0]), "box_y1": float(box[1]),
                "box_x2": float(box[2]), "box_y2": float(box[3]),
                "mask_area_px": int(obj_mask.sum()),
            }
        )

    clean_mask = house_mask & ~removed_mask
    clean_bbox = _mask_bbox(clean_mask, pad_px=8)
    if clean_bbox is None:
        raise RuntimeError("SAM produced an empty cleaned house mask.")

    house_bbox = _mask_bbox(house_mask, pad_px=8) or clean_bbox
    house_cutout = house_dir / "cutouts" / f"{image_id}.png"
    house_mask_path = house_dir / "masks" / f"{image_id}.png"
    house_area = _write_rgba_cutout(work.rgb, house_mask, house_bbox, house_cutout)
    _write_mask(house_mask, house_mask_path)

    clean_cutout = clean_dir / "cutouts" / f"{image_id}.png"
    clean_mask_path = clean_dir / "masks" / f"{image_id}.png"
    removed_mask_path = clean_dir / "removed_masks" / f"{image_id}.png"
    clean_area = _write_rgba_cutout(work.rgb, clean_mask, clean_bbox, clean_cutout)
    _write_mask(clean_mask, clean_mask_path)
    _write_mask(removed_mask, removed_mask_path)
    _write_overlay(work.rgb, house_mask, removed_mask, clean_dir / "overlays" / f"{image_id}.png")

    with (clean_dir / "removed_objects.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["class_name", "source_score", "sam_score", "box_x1", "box_y1", "box_x2", "box_y2", "mask_area_px"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(object_rows)

    step2_db = clean_dir / "clean_house_crops.sqlite3"
    _create_step2_db(
        db_path=step2_db,
        run_id=run_id,
        image_id=image_id,
        image_path=image_path,
        crop_path=clean_cutout,
        mask_path=clean_mask_path,
        crop_bbox=clean_bbox,
        mask_area_px=clean_area,
        sam_checkpoint=sam_checkpoint,
        device=device_str,
    )

    step3_res = run_step3(
        step2_db=step2_db,
        db_path=step3_dir / "damage_detections.sqlite3",
        rgb_dir=step3_dir / "rgb",
        overlay_dir=step3_dir / "overlays",
        summary_csv=step3_dir / "summary.csv",
        prompt_groups=parse_prompt_groups([]),
        source_run_id=run_id,
        device=device,
        box_threshold=damage_box_threshold,
        text_threshold=damage_text_threshold,
        tiled_threshold=damage_tiled_threshold,
        tile_scales=["small", "medium"],
        recursive_max_depth=damage_recursive_max_depth,
        recursive_min_box_px=damage_recursive_min_box_px,
    )

    step4_res = None
    if int(step3_res.get("detection_count") or 0) > 0:
        step4_res = run_step4(
            step3_db=step3_dir / "damage_detections.sqlite3",
            rgb_dir=step3_dir / "rgb",
            db_path=step4_dir / "semantic_labels.sqlite3",
            crops_dir=step4_dir / "crops",
            summary_csv=step4_dir / "summary.csv",
            source_run_id=step3_res["run_id"],
            device=device,
            save_crops=True,
            min_prob=clip_min_prob,
        )

    return {
        "run_id": run_id,
        "work_path": str(work.work_path),
        "house_cutout": str(house_cutout),
        "clean_cutout": str(clean_cutout),
        "removed_objects_csv": str(clean_dir / "removed_objects.csv"),
        "clean_overlay": str(clean_dir / "overlays" / f"{image_id}.png"),
        "step2_db": str(step2_db),
        "step3": step3_res,
        "step4": step4_res,
        "house_area_px": int(house_area),
        "clean_area_px": int(clean_area),
    }


def main() -> int:
    repo_root = _resolve_repo_root()
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pineline.house_cutout.common.paths import default_sam_checkpoint

    lab_root = repo_root.parent
    parser = argparse.ArgumentParser(description="Crop house with SAM, remove objects, then run GDINO damage + OpenCLIP semantic.")
    parser.add_argument("--image", type=Path, default=lab_root / "data" / "HinhAnhThucTe" / "NTT - 16m Lan 3.tif")
    parser.add_argument("--object-db", type=Path, default=lab_root / "model_with_inference" / "semi_labeling_training" / "results" / "nha_truyen_thong" / "step_gdino_window_door_star_precise" / "detections.sqlite3")
    parser.add_argument("--output-root", type=Path, default=lab_root / "model_with_inference" / "semi_labeling_training" / "results" / "nha_truyen_thong" / "clean_damage_pipeline")
    parser.add_argument("--sam-checkpoint", type=Path, default=default_sam_checkpoint())
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--work-max-side", type=int, default=1024)
    parser.add_argument("--no-transom", dest="include_transom", action="store_false")
    parser.set_defaults(include_transom=True)
    parser.add_argument("--damage-box-threshold", type=float, default=0.16)
    parser.add_argument("--damage-text-threshold", type=float, default=0.16)
    parser.add_argument("--damage-tiled-threshold", type=int, default=400,
                        help="Force recursive GDINO damage detect when max image side is above this value.")
    parser.add_argument("--damage-recursive-max-depth", type=int, default=2,
                        help="Recursive depth for patch detection. 2 gives coarse split then smaller tiles.")
    parser.add_argument("--damage-recursive-min-box-px", type=int, default=12,
                        help="Keep small damage boxes during tiled detection.")
    parser.add_argument("--clip-min-prob", type=float, default=0.45)
    args = parser.parse_args()

    result = run_pipeline(
        image_path=args.image,
        object_db=args.object_db,
        output_root=args.output_root,
        sam_checkpoint=args.sam_checkpoint,
        device=args.device,
        work_max_side=args.work_max_side,
        include_transom=args.include_transom,
        damage_box_threshold=args.damage_box_threshold,
        damage_text_threshold=args.damage_text_threshold,
        damage_tiled_threshold=args.damage_tiled_threshold,
        damage_recursive_max_depth=args.damage_recursive_max_depth,
        damage_recursive_min_box_px=args.damage_recursive_min_box_px,
        clip_min_prob=args.clip_min_prob,
    )
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
