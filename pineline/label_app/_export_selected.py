"""One-off: copy selected ROI images into a subfolder of label_app and emit
matching .json box files (label_app schema) from StableDINO predictions.

ROI selection (filename stem -> image to use):
    roi54  -> DSC01279__roi54
    roi56  -> DSC01279__roi56
    roi228 -> DSC01310__roi228
    roi245 -> DSC01312__roi245
    roi246 -> DSC01312__roi246
    roi424 -> DSC01336__roi424
    roi788 -> DSC01376__roi788
    roi791 -> DSC01376__roi791
    roi790 -> DSC01376__roi790
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

BASE = Path(r"C:\Users\Dell Precision 7810\Desktop\quangvinh_workspace")
IMAGES_DIR = BASE / "model_with_inference" / "semi_labeling" / "dataset" / "myrun" / "images" / "test"
SD_DIR = BASE / "model_with_inference" / "semi_labeling_training" / "stabledino_final_predict_test"
RESULTS = SD_DIR / "coco_instances_results.json"
META = SD_DIR / "dataset_cache" / "test_coco.json"

OUT_DIR = Path(__file__).resolve().parent / "selected_boxes"

CATEGORIES = {1: "crack", 2: "mold", 3: "spall"}
CONF_THRESH = 0.05

SELECTED = [
    "DSC01279__roi54.png",
    "DSC01279__roi56.png",
    "DSC01310__roi228.png",
    "DSC01312__roi245.png",
    "DSC01312__roi246.png",
    "DSC01336__roi424.png",
    "DSC01376__roi788.png",
    "DSC01376__roi791.png",
    "DSC01376__roi790.png",
]


def main() -> None:
    predictions = json.loads(RESULTS.read_text(encoding="utf-8"))
    coco = json.loads(META.read_text(encoding="utf-8"))

    fname_to_id = {Path(im["file_name"]).name: im["id"] for im in coco["images"]}

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for p in predictions:
        preds_by_image[p["image_id"]].append(p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname in SELECTED:
        src = IMAGES_DIR / fname
        if not src.exists():
            print(f"[SKIP] image not found: {fname}")
            continue

        # copy image
        shutil.copy2(src, OUT_DIR / fname)

        # build boxes json (xyxy, label, score) filtered by conf
        rows = []
        img_id = fname_to_id.get(fname)
        if img_id is not None:
            for p in preds_by_image.get(img_id, []):
                if p["score"] < CONF_THRESH:
                    continue
                x, y, w, h = p["bbox"]  # COCO xywh
                rows.append({
                    "box": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
                    "label": CATEGORIES.get(p["category_id"], str(p["category_id"])),
                    "score": round(float(p["score"]), 4),
                })

        json_path = (OUT_DIR / fname).with_suffix(".json")
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {fname}: {len(rows)} boxes -> {json_path.name}")

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
