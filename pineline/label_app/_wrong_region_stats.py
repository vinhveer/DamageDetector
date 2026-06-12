"""Compute, per image, the percentage of predicted boxes that fall COMPLETELY
outside every ground-truth damage region.

Ground truth = the hand-edited .jsonm files in selected_boxes/ (label_app schema,
box = [x1, y1, x2, y2] absolute pixels).

Predictions:
    StableDINO -> stabledino_final_predict_test/coco_instances_results.json (COCO xywh)
    YOLO       -> myrun_yolo26x_img768_b16_100ep/val_test/predictions.json (xywh + file_name)

A predicted box is "completely wrong" if it has ZERO overlap area with all GT boxes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

BASE = Path(r"C:\Users\Dell Precision 7810\Desktop\quangvinh_workspace")
GT_DIR = BASE / "DamageDetector" / "pineline" / "label_app" / "selected_boxes"
SD_DIR = BASE / "model_with_inference" / "semi_labeling_training" / "stabledino_final_predict_test"
YOLO_PRED = BASE / "model_with_inference" / "semi_labeling_training" / "myrun_yolo26x_img768_b16_100ep" / "val_test" / "predictions.json"

CONF = 0.05


def load_gt() -> dict[str, list[list[float]]]:
    """stem (without ext) -> list of xyxy GT boxes."""
    gt: dict[str, list[list[float]]] = {}
    for f in sorted(GT_DIR.glob("*.jsonm")):
        stem = f.stem  # e.g. DSC01279__roi54
        rows = json.loads(f.read_text(encoding="utf-8"))
        boxes = []
        for r in rows:
            b = r.get("box")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                x1, y1, x2, y2 = (float(v) for v in b)
                boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        gt[stem] = boxes
    return gt


def overlaps_any(pred: list[float], gts: list[list[float]]) -> bool:
    px1, py1, px2, py2 = pred
    for gx1, gy1, gx2, gy2 in gts:
        iw = min(px2, gx2) - max(px1, gx1)
        ih = min(py2, gy2) - max(py1, gy1)
        if iw > 0 and ih > 0:
            return True
    return False


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
        stem = Path(p["file_name"]).stem
        x, y, w, h = p["bbox"]
        out[stem].append([x, y, x + w, y + h])
    return out


def report(name: str, preds_by_stem: dict[str, list[list[float]]], gt: dict[str, list[list[float]]]):
    print(f"\n=== {name} (conf >= {CONF}) ===")
    print(f"{'image':24} {'#pred':>6} {'#wrong':>7} {'wrong%':>8}  note")
    tot_pred = tot_wrong = 0
    for stem in sorted(gt):
        gts = gt[stem]
        preds = preds_by_stem.get(stem, [])
        n = len(preds)
        if not gts:
            wrong = n  # no damage region at all -> every pred is outside
            note = "no GT region"
        else:
            wrong = sum(1 for p in preds if not overlaps_any(p, gts))
            note = ""
        pct = (wrong / n * 100) if n else 0.0
        tot_pred += n
        tot_wrong += wrong
        print(f"{stem:24} {n:>6} {wrong:>7} {pct:>7.1f}%  {note}")
    tot_pct = (tot_wrong / tot_pred * 100) if tot_pred else 0.0
    print(f"{'TOTAL':24} {tot_pred:>6} {tot_wrong:>7} {tot_pct:>7.1f}%")


def main():
    gt = load_gt()
    report("StableDINO", load_stabledino(), gt)
    report("YOLO", load_yolo(), gt)


if __name__ == "__main__":
    main()
