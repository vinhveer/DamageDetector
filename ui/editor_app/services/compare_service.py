from __future__ import annotations

from pathlib import Path

import cv2

from ui.editor_app.services.metrics import compute_dice_iou
from ui.editor_app.services.run_storage import RunStorageService


class CompareService:
    def __init__(self, run_storage: RunStorageService) -> None:
        self._run_storage = run_storage

    def compare_run(self, *, run_dir: Path, gt_dir: Path, affix: str = "") -> list[dict]:
        items = self._run_storage.list_result_items(run_dir)
        gt_files = [
            *gt_dir.glob("*.png"),
            *gt_dir.glob("*.jpg"),
            *gt_dir.glob("*.jpeg"),
            *gt_dir.glob("*.bmp"),
            *gt_dir.glob("*.tif"),
            *gt_dir.glob("*.tiff"),
        ]
        results: list[dict] = []
        for item in items:
            image_path = Path(str(item.get("image_path") or ""))
            mask_path = Path(str(item.get("mask_path") or ""))
            if not image_path.is_file() or not mask_path.is_file():
                continue
            gt_path = self._match_gt_file(image_path, gt_files, affix)
            if gt_path is None or not gt_path.is_file():
                continue
            pred_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if pred_mask is None or gt_mask is None:
                continue
            if pred_mask.shape != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            dice, iou = compute_dice_iou(pred_mask, gt_mask)
            results.append(
                {
                    "image": image_path.name,
                    "gt_mask": gt_path.name,
                    "dice": float(dice),
                    "iou": float(iou),
                    "mask_path": str(mask_path),
                    "gt_path": str(gt_path),
                }
            )
        return results

    def _match_gt_file(self, image_path: Path, gt_files: list[Path], affix: str) -> Path | None:
        affix = str(affix or "").strip()
        for gt_file in gt_files:
            if gt_file.name == image_path.name or gt_file.stem == image_path.stem:
                return gt_file
            if affix and (
                gt_file.stem == f"{affix}{image_path.stem}" or gt_file.stem == f"{image_path.stem}{affix}"
            ):
                return gt_file
        return None
