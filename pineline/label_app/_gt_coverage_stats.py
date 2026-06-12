"""Per image, compute how much of the ground-truth damage area is covered by the
predicted boxes (coverage / recall-of-area).

coverage = area(union(GT) AND union(pred)) / area(union(GT))

Union areas are computed by rasterizing boxes onto a boolean mask so overlapping
boxes are not double counted. Score threshold = 0.05.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\Dell Precision 7810\Desktop\quangvinh_workspace")
GT_DIR = BASE / "DamageDetector" / "pineline" / "label_app" / "selected_boxes"
SD_DIR = BASE / "model_with_inference" / "semi_labeling_training" / "stabledino_final_predict_test"
YOLO_PRED = BASE / "model_with_inference" / "semi_labeling_training" / "myrun_yolo26x_img768_b16_100ep" / "val_test" / "predictions.json"

CONF = 0.05


def load_gt() -> dict[str, list[list[float]]]:
    gt: dict[str, list[list[float]]] = {}
    for f in sorted(GT_DIR.glob("*.jsonm")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        boxes = []
        for r in rows:
            b = r.get("box")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                x1, y1, x2, y2 = (float(v) for v in b)
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        gt[f.stem] = boxes
    return gt


def load_stabledino() -> dict[str, list[list[float]]]:
    coco = json.loads((SD_DIR / "dataset_cache" / "test_coco.json").read_text(encoding="utf-8"))
    preds = json.loads((SD_DIR / "coco_instances_results.json").read_text(encoding="utf-8"))
    id_to_stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    out: dict[str, list[list[float]]] = defaultdict(list)
    for p in preds:
        if p["score"] < CONF:
            continue
        stem = id_to_stem.get(p["image_id"])
        if stem is None:
            continue
        x, y, w, h = p["bbox"]
        out[stem].append([x, y, x + w, y + h])
    return out


def load_yolo() -> dict[str, list[list[float]]]:
    preds = json.loads(YOLO_PRED.read_text(encoding="utf-8"))
    out: dict[str, list[list[float]]] = defaultdict(list)
    for p in preds:
        if p["score"] < CONF:
            continue
        out[Path(p["file_name"]).stem].append([p["bbox"][0], p["bbox"][1],
                                                p["bbox"][0] + p["bbox"][2],
                                                p["bbox"][1] + p["bbox"][3]])
    return out


def rasterize(boxes: list[list[float]], W: int, H: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        ix1 = max(0, int(np.floor(x1))); iy1 = max(0, int(np.floor(y1)))
        ix2 = min(W, int(np.ceil(x2)));  iy2 = min(H, int(np.ceil(y2)))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = True
    return mask


def report(name: str, preds_by_stem: dict[str, list[list[float]]], gt: dict[str, list[list[float]]]):
    print(f"\n=== {name} (conf >= {CONF}) ===")
    print(f"{'image':24} {'GT_area':>10} {'covered':>10} {'coverage%':>10}")
    tot_gt = tot_cov = 0
    for stem in sorted(gt):
        gts = gt[stem]
        if not gts:
            print(f"{stem:24} {'-':>10} {'-':>10} {'(no GT)':>10}")
            continue
        preds = preds_by_stem.get(stem, [])
        allb = gts + preds
        W = int(np.ceil(max(b[2] for b in allb))) + 1
        H = int(np.ceil(max(b[3] for b in allb))) + 1
        gt_mask = rasterize(gts, W, H)
        pr_mask = rasterize(preds, W, H)
        gt_area = int(gt_mask.sum())
        cov = int((gt_mask & pr_mask).sum())
        pct = (cov / gt_area * 100) if gt_area else 0.0
        tot_gt += gt_area
        tot_cov += cov
        print(f"{stem:24} {gt_area:>10} {cov:>10} {pct:>9.1f}%")
    tot_pct = (tot_cov / tot_gt * 100) if tot_gt else 0.0
    print(f"{'TOTAL':24} {tot_gt:>10} {tot_cov:>10} {tot_pct:>9.1f}%")


def main():
    gt = load_gt()
    report("StableDINO", load_stabledino(), gt)
    report("YOLO", load_yolo(), gt)


if __name__ == "__main__":
    main()
