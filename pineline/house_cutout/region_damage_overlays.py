from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

LABEL_COLORS = {
    "crack": (36, 114, 255),
    "mold": (48, 180, 75),
    "stain": (180, 80, 255),
    "spall": (0, 190, 255),
    "exposed_rebar": (255, 120, 30),
}

CRACK_QUERIES = [
    "crack", "surface crack", "concrete crack", "wall crack",
    "thin crack", "long crack", "hairline crack", "fracture line",
]
MOLD_QUERIES = [
    "mold", "mildew", "moss", "algae", "algae stain",
    "biological growth on wall", "green stain on concrete",
    "fungal growth", "lichen on surface",
]
STAIN_QUERIES = [
    "water stain", "damp stain", "dark stain on wall",
    "discoloration patch", "dirty stain", "rust stain",
    "surface contamination", "blackish patch on concrete",
]
GROUPS = [("crack", CRACK_QUERIES), ("mold", MOLD_QUERIES), ("stain", STAIN_QUERIES)]


def image_to_bgr_respecting_alpha(image_path, *, background=(255, 255, 255)):
    """Load an image for overlays without exposing hidden RGB under alpha=0.

    PIL's RGBA -> RGB conversion drops alpha and keeps the original RGB values in
    transparent pixels. Cutouts often store the full original image RGB behind a
    mask, so blindly converting makes the output look like the uncut image. This
    composites RGBA onto a solid background first.
    """
    with Image.open(image_path) as im:
        if im.mode == "RGBA":
            bg = Image.new("RGBA", im.size, (*background, 255))
            rgb = Image.alpha_composite(bg, im).convert("RGB")
        else:
            rgb = im.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def color_for(label):
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    h = abs(hash(label))
    return (40 + h % 200, 40 + (h // 7) % 200, 40 + (h // 13) % 200)


def match_group(label):
    text = str(label or "").lower()
    for name, queries in GROUPS:
        for q in queries:
            if q.lower() in text:
                return name
    return None


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def nms(dets, thr=0.35):
    kept = []
    for d in sorted(dets, key=lambda x: x["score"], reverse=True):
        if all(iou(d["box"], k["box"]) <= thr for k in kept):
            kept.append(d)
    return kept


def merge_same_class(dets, *, gap_ratio=0.15, min_gap=8, max_span_ratio=0.45):
    """Merge same-class boxes that overlap or are very close into one box.

    A long crack tiled at 768px comes back as many small boxes following the crack.
    Expand each box by a small margin and union touching boxes per class, so adjacent
    fragments of one streak join into a single box, without bridging unrelated areas
    across the whole facade. Clusters whose union would exceed max_span_ratio of the
    image are left unmerged to avoid mega-boxes that make segmentation flood.
    """
    by_group = {}
    for d in dets:
        by_group.setdefault(d["group"], []).append(d)

    all_boxes = [d["box"] for d in dets]
    image_w = max((b[2] for b in all_boxes), default=0) - min((b[0] for b in all_boxes), default=0)
    image_h = max((b[3] for b in all_boxes), default=0) - min((b[1] for b in all_boxes), default=0)

    def expanded(b):
        bw = b[2] - b[0]; bh = b[3] - b[1]
        gx = max(min_gap, bw * gap_ratio)
        gy = max(min_gap, bh * gap_ratio)
        return [b[0] - gx, b[1] - gy, b[2] + gx, b[3] + gy]

    merged_all = []
    for group, items in by_group.items():
        n = len(items)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        exp = [expanded(it["box"]) for it in items]
        for i in range(n):
            for j in range(i + 1, n):
                if iou(exp[i], exp[j]) > 0 or _boxes_touch(exp[i], items[j]["box"]) or _boxes_touch(exp[j], items[i]["box"]):
                    union(i, j)
        clusters = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)
        for members in clusters.values():
            xs1 = min(items[m]["box"][0] for m in members)
            ys1 = min(items[m]["box"][1] for m in members)
            xs2 = max(items[m]["box"][2] for m in members)
            ys2 = max(items[m]["box"][3] for m in members)

            span_w = xs2 - xs1
            span_h = ys2 - ys1
            if len(members) > 1 and (
                (image_w > 0 and span_w > image_w * max_span_ratio)
                or (image_h > 0 and span_h > image_h * max_span_ratio)
            ):
                merged_all.extend(items[m] for m in members)
                continue

            best = max(members, key=lambda m: items[m]["score"])
            merged_all.append({
                "box": [xs1, ys1, xs2, ys2],
                "score": float(items[best]["score"]),
                "group": group,
                "label": group,
                "members": len(members),
            })
    return merged_all


def _boxes_touch(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

def square_tiles(w, h, tile=768, overlap=192):
    """Square tiles with overlap. Keeps aspect ratio ~1 so GDINO is not distorted."""
    stride = max(1, tile - overlap)
    def starts(length):
        if length <= tile:
            return [0]
        vals = list(range(0, length - tile + 1, stride))
        if vals[-1] != length - tile:
            vals.append(length - tile)
        return vals
    out = []
    for y in starts(h):
        for x in starts(w):
            out.append((x, y, min(w, x + tile), min(h, y + tile)))
    return out


def detect_cracks_and_damage(region_bgr, *, edge_pad=10, score_min=0.22,
                             tile=768, overlap=192, max_area_ratio=0.20):
    """Tile region into square tiles, GDINO detect damage, drop edge/oversized boxes.

    Square tiling avoids the distortion of a fixed 3x4 grid on tall/narrow regions
    (e.g. a column), which otherwise produces hallucinated boxes. score_min is kept
    higher to suppress low-confidence noise on smooth surfaces.
    """
    from object_detection.dino.client import get_dino_service
    from object_detection.dino.engine import default_gdino_checkpoint

    h, w = region_bgr.shape[:2]
    queries = []
    for _, qs in GROUPS:
        queries.extend(qs)
    ckpt = str(default_gdino_checkpoint())
    service = get_dino_service()
    boxes = []
    dropped_edge = 0
    dropped_large = 0
    tmp = Path("/tmp/_region_damage_patch.png")
    tiles = square_tiles(w, h, tile=tile, overlap=overlap)
    try:
        for (px1, py1, px2, py2) in tiles:
            patch = region_bgr[py1:py2, px1:px2]
            if patch.size == 0:
                continue
            cv2.imwrite(str(tmp), patch)
            res = service.call("predict", {
                "image_path": str(tmp),
                "params": {
                    "gdino_checkpoint": ckpt,
                    "gdino_config_id": "auto",
                    "text_queries": queries,
                    "box_threshold": 0.18,
                    "text_threshold": 0.18,
                    "max_dets": 60,
                    "device": "auto",
                },
            })
            for det in list(res.get("detections") or []):
                score = float(det.get("score") or 0.0)
                if score < score_min:
                    continue
                b = det.get("box")
                if not isinstance(b, (list, tuple)) or len(b) != 4:
                    continue
                bx1, by1, bx2, by2 = [float(v) for v in b]
                if bx2 <= bx1 or by2 <= by1:
                    continue
                gx1, gy1, gx2, gy2 = bx1 + px1, by1 + py1, bx2 + px1, by2 + py1
                group = match_group(det.get("label"))
                if group is None:
                    continue
                if gx1 <= edge_pad or gy1 <= edge_pad or gx2 >= w - edge_pad or gy2 >= h - edge_pad:
                    dropped_edge += 1
                    continue
                if ((gx2 - gx1) * (gy2 - gy1)) / (w * h) > max_area_ratio:
                    dropped_large += 1
                    continue
                boxes.append({"box": [gx1, gy1, gx2, gy2], "score": score,
                              "group": group, "label": str(det.get("label") or group)})
    finally:
        try:
            service.close()
        except Exception:
            pass
    kept = nms(boxes, thr=0.35)
    return kept, {"tiles": len(tiles), "raw_inside": len(boxes), "dropped_edge": dropped_edge,
                  "dropped_large": dropped_large, "kept": len(kept)}

TRAINED_CLASSES = ["crack", "mold", "spall"]


def detect_damage_trained(full_bgr, image_path, *, detector, conf=0.1,
                          edge_pad=2, max_area_ratio=0.20, device="auto",
                          no_filter=False, max_dets=300, nms_thr=0.45, merge=False,
                          scale_to_original=1.0, offset_to_original=(0.0, 0.0),
                          classes=None):
    """Detect damage with a trained detector (yolo or stabledino) instead of GDINO.

    The detector runs on the full image; MultiDetector handles 768px tiling for
    large images. Labels are the trained classes crack/mold/spall, so group == label.

    no_filter=True keeps every detection above conf: no NMS, no area cap, no edge
    drop, and a very high max_dets, to maximise coverage like a raw detector dump.
    """
    from pineline.common.detection import MultiDetector, default_detection_config
    from pineline.common.model_defaults import default_stabledino_checkpoint, default_yolo_model

    detect_classes = list(classes or TRAINED_CLASSES)
    if no_filter:
        edge_pad = 0
        max_area_ratio = 1.01
        max_dets = max(max_dets, 5000)

    h, w = full_bgr.shape[:2]
    det_scale = float(scale_to_original or 1.0)
    off_x, off_y = [float(v) for v in offset_to_original]
    cfg = default_detection_config(
        models=detector,
        gdino_checkpoint=None,
        yolo_model=str(default_yolo_model()),
        stabledino_checkpoint=str(default_stabledino_checkpoint()),
        yolo_conf=conf,
        stabledino_conf=conf,
        max_dets=int(max_dets),
        device=device,
        tiled_threshold=400,
        stabledino_output_dir=str(Path("/tmp") / f"_stabledino_{detector}"),
        disable_tiled_nms=no_filter,
    )
    detector_obj = MultiDetector(cfg)
    boxes = []
    dropped_edge = 0
    dropped_large = 0
    try:
        dets = detector_obj.detect(Path(image_path), width=w, height=h,
                                   queries=detect_classes, names=detect_classes)
    finally:
        detector_obj.close()
    for det in dets:
        score = float(det.get("score") or 0.0)
        if score < conf:
            continue
        b = det.get("box")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        bx1, by1, bx2, by2 = [((float(b[0]) - off_x) * det_scale),
                              ((float(b[1]) - off_y) * det_scale),
                              ((float(b[2]) - off_x) * det_scale),
                              ((float(b[3]) - off_y) * det_scale)]
        if bx2 <= bx1 or by2 <= by1:
            continue
        group = str(det.get("label") or "").strip().lower()
        if group not in detect_classes:
            continue
        if not no_filter:
            if bx1 <= edge_pad or by1 <= edge_pad or bx2 >= w - edge_pad or by2 >= h - edge_pad:
                dropped_edge += 1
                continue
            if ((bx2 - bx1) * (by2 - by1)) / (w * h) > max_area_ratio:
                dropped_large += 1
                continue
        boxes.append({"box": [bx1, by1, bx2, by2], "score": score,
                      "group": group, "label": group})
    kept = boxes if no_filter else nms(boxes, thr=nms_thr)
    merged_count = None
    if merge:
        kept = merge_same_class(kept)
        merged_count = len(kept)
    return kept, {"detector": detector, "raw": len(dets), "kept_pre_nms": len(boxes),
                  "dropped_edge": dropped_edge, "dropped_large": dropped_large,
                  "no_filter": no_filter, "merged": merged_count, "kept": len(kept)}


def render_text_x3(out_dir, full_bgr, seg_results, crop_box, *, extra_box=None):
    """Render a single report overlay (box+label+segment) with ~3x larger text."""
    canvas = full_bgr.copy()
    h, w = canvas.shape[:2]
    labels = sorted({r["group"] for r in seg_results})
    for r in seg_results:
        if r.get("segmenter") == "box_only":
            continue
        rx1, ry1, rx2, ry2 = r["roi"]
        mp = out_dir / "masks" / f"det_{r['idx']:04d}_{r['group']}.png"
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (ry2 - ry1, rx2 - rx1):
            m = cv2.resize(m, (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_NEAREST)
        mb = m > 127
        c = np.array(color_for(r["group"]), dtype=np.uint8)
        roi = canvas[ry1:ry2, rx1:rx2]
        ci = np.zeros_like(roi); ci[:] = c
        canvas[ry1:ry2, rx1:rx2] = np.where(mb[..., None], (roi * 0.5 + ci * 0.5).astype(np.uint8), roi)
    if extra_box is not None:
        ex1, ey1, ex2, ey2, etext = extra_box
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), color_for("column"), 12)
        _draw_label_big(canvas, (ex1, ey1), etext, color_for("column"))
    for r in seg_results:
        c = color_for(r["group"])
        bx1, by1, bx2, by2 = [int(round(v)) for v in r["box"]]
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), c, 10)
        _draw_label_big(canvas, (bx1, by1), f"{r['group']} {r['score']:.2f}", c)
    if crop_box is not None:
        cx1, cy1, cx2, cy2 = [int(round(v)) for v in crop_box]
        canvas = canvas[cy1:cy2, cx1:cx2]
    _legend_big(canvas, labels, with_column=extra_box is not None)
    cv2.imwrite(str(out_dir / "5_box_label_segment_text_x3.png"), canvas)


def _draw_label_big(img, p1, text, c, font=2.7, thick=6):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
    ty = max(th + bl + 20, p1[1])
    cv2.rectangle(img, (p1[0], ty - th - bl - 20), (p1[0] + tw + 30, ty + 14), c, -1)
    cv2.putText(img, text, (p1[0] + 14, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick, cv2.LINE_AA)


def _legend_big(img, labels, *, with_column=False):
    y = 130
    items = list(labels) + (["column"] if with_column else [])
    for g in items:
        c = color_for(g)
        cv2.rectangle(img, (40, y - 70), (130, y + 20), c, -1)
        cv2.putText(img, g, (155, y), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 11, cv2.LINE_AA)
        cv2.putText(img, g, (155, y), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 5, cv2.LINE_AA)
        y += 130


def pad_box(box, w, h, pad=0.25, minpad=32):
    x1, y1, x2, y2 = box
    p = max(minpad, int(max(x2 - x1, y2 - y1) * pad))
    return (max(0, int(x1 - p)), max(0, int(y1 - p)),
            min(w, int(x2 + p)), min(h, int(y2 + p)))


def segment_region(region_bgr, kept, out_dir, *, sam_vit_h_ckpt, lora_model_dir):
    """Crack -> SAM finetune coarse_refine. Other -> SAM ViT-H box prompt.

    Segmentation runs on original-resolution region crops.
    """
    from segmentation.sam.runtime.engine import SamParams, SamRunner
    from segmentation.sam.finetune.engine import SamFinetuneParams, SamFinetuneRunner
    import base64

    h, w = region_bgr.shape[:2]
    region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    mask_dir = out_dir / "masks"; roi_dir = out_dir / "roi_images"; tmp_dir = out_dir / "tmp"
    for d in (mask_dir, roi_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    sam_runner = None
    ft_runner = None
    ft_params = None
    lora = Path(lora_model_dir)
    results = []
    for i, d in enumerate(kept):
        group = d["group"]
        rx1, ry1, rx2, ry2 = pad_box(d["box"], w, h)
        roi = region_rgb[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue
        roi_path = roi_dir / f"det_{i:04d}_{group}.png"
        cv2.imwrite(str(roi_path), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        out_mask_path = mask_dir / f"det_{i:04d}_{group}.png"
        # Reuse a previously computed mask (skip the heavy segmentation model) if present.
        cached = cv2.imread(str(out_mask_path), cv2.IMREAD_GRAYSCALE) if out_mask_path.exists() else None
        if cached is not None:
            if cached.shape[:2] != roi.shape[:2]:
                cached = cv2.resize(cached, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_roi = (cached > 127).astype(np.uint8) * 255
            segmenter = "sam_finetune_coarse_refine" if group == "crack" else "sam_vit_h"
            cv2.imwrite(str(out_mask_path), mask_roi)
            results.append({"idx": i, "group": group, "score": d["score"], "label": d["label"],
                            "box": d["box"], "roi": [rx1, ry1, rx2, ry2],
                            "segmenter": segmenter, "mask_area_px": int((mask_roi > 127).sum())})
            continue
        mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        if group == "crack":
            if ft_runner is None:
                ft_runner = SamFinetuneRunner()
                ft_params = SamFinetuneParams(
                    sam_checkpoint=str(lora / "sam_vit_b_01ec64.pth"),
                    sam_model_type="vit_b", delta_type="lora",
                    delta_checkpoint=str(lora / "coarse_best_model.pth"), rank=4,
                    predict_mode="coarse_refine",
                    refine_delta_checkpoint=str(lora / "refine_best_model.pth"),
                    refine_delta_type="lora", refine_rank=4, threshold="auto",
                    device="auto", output_dir=str(tmp_dir), task_group="crack_only")
            res = ft_runner.predict(str(roi_path), ft_params, log_fn=None)
            mp = res.get("mask_path")
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp else None
            if m is not None:
                if m.shape[:2] != roi.shape[:2]:
                    m = cv2.resize(m, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_roi = (m > 127).astype(np.uint8) * 255
            segmenter = "sam_finetune_coarse_refine"
        else:
            if sam_runner is None:
                sam_runner = SamRunner()
            sp = SamParams(sam_checkpoint=str(sam_vit_h_ckpt), sam_model_type="vit_h",
                           device="auto", output_dir=str(tmp_dir), task_group="more_damage")
            local_box = [d["box"][0] - rx1, d["box"][1] - ry1, d["box"][2] - rx1, d["box"][3] - ry1]
            res = sam_runner.segment_boxes(str(roi_path), sp,
                                           [{"box": local_box, "label": group, "score": d["score"]}], log_fn=None)
            dets = list(res.get("detections") or [])
            if dets and dets[0].get("mask_b64"):
                arr = np.frombuffer(base64.b64decode(dets[0]["mask_b64"]), dtype=np.uint8)
                m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape[:2] != roi.shape[:2]:
                        m = cv2.resize(m, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_roi = (m > 127).astype(np.uint8) * 255
            segmenter = "sam_vit_h"
        cv2.imwrite(str(mask_dir / f"det_{i:04d}_{group}.png"), mask_roi)
        results.append({"idx": i, "group": group, "score": d["score"], "label": d["label"],
                        "box": d["box"], "roi": [rx1, ry1, rx2, ry2],
                        "segmenter": segmenter, "mask_area_px": int((mask_roi > 127).sum())})
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results

def draw_label(canvas, p1, text, color):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    ty = max(th + bl + 2, p1[1])
    cv2.rectangle(canvas, (p1[0], ty - th - bl - 3), (p1[0] + tw + 5, ty + 2), color, -1)
    cv2.putText(canvas, text, (p1[0] + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)


def legend(canvas, labels):
    y = 26
    for g in sorted(labels):
        c = color_for(g)
        cv2.rectangle(canvas, (12, y - 16), (34, y + 2), c, -1)
        cv2.putText(canvas, g, (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, g, (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 30


def build_overlays(full_bgr, region_offset, region_bgr, seg_results, out_dir, crop_box=None, extra_box=None):
    """Render 4 overlays on the full image. region_offset=(ox,oy) maps region->full.

    If crop_box=(x1,y1,x2,y2) is given, the saved overlays are cropped to that box
    so the output shows only the cut-out region (e.g. the house / the column).
    extra_box=(x1,y1,x2,y2,text) draws an extra labelled box (e.g. the column).
    """
    ox, oy = region_offset
    rh, rw = region_bgr.shape[:2]
    labels = sorted({r["group"] for r in seg_results})

    box_label = full_bgr.copy()
    box_only = full_bgr.copy()
    seg_only = full_bgr.copy()
    box_seg = full_bgr.copy()

    # paint masks first onto seg canvases
    for r in seg_results:
        if r.get("segmenter") == "box_only":
            continue
        rx1, ry1, rx2, ry2 = r["roi"]
        mp = out_dir / "masks" / f"det_{r['idx']:04d}_{r['group']}.png"
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (ry2 - ry1, rx2 - rx1):
            m = cv2.resize(m, (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_NEAREST)
        mb = m > 127
        c = np.array(color_for(r["group"]), dtype=np.uint8)
        fy1, fy2 = oy + ry1, oy + ry2
        fx1, fx2 = ox + rx1, ox + rx2
        for canvas in (seg_only, box_seg):
            roi = canvas[fy1:fy2, fx1:fx2]
            ci = np.zeros_like(roi); ci[:] = c
            canvas[fy1:fy2, fx1:fx2] = np.where(mb[..., None], (roi * 0.5 + ci * 0.5).astype(np.uint8), roi)

    # boxes
    for r in seg_results:
        c = color_for(r["group"])
        bx1, by1, bx2, by2 = r["box"]
        p1 = (ox + int(round(bx1)), oy + int(round(by1)))
        p2 = (ox + int(round(bx2)), oy + int(round(by2)))
        for canvas in (box_label, box_only, box_seg):
            cv2.rectangle(canvas, p1, p2, c, 3)
        text = f"{r['group']} {r['score']:.2f}"
        draw_label(box_label, p1, text, c)
        draw_label(box_seg, p1, text, c)

    if extra_box is not None:
        ex1, ey1, ex2, ey2, etext = extra_box
        ec = color_for("column")
        for canvas in (box_label, box_only, seg_only, box_seg):
            cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), ec, 5)
        draw_label(box_label, (ex1, ey1), etext, ec)
        draw_label(box_seg, (ex1, ey1), etext, ec)

    canvases = {
        "1_box_label.png": box_label,
        "2_box_only.png": box_only,
        "3_segment_only.png": seg_only,
        "4_box_label_segment.png": box_seg,
    }
    if crop_box is not None:
        cx1, cy1, cx2, cy2 = [int(round(v)) for v in crop_box]
        canvases = {name: img[cy1:cy2, cx1:cx2] for name, img in canvases.items()}
    # legend drawn after crop so it stays visible in the corner
    for img in canvases.values():
        legend(img, labels)
    for name, img in canvases.items():
        cv2.imwrite(str(out_dir / name), img)

def export_data(out_dir, image_path, region_box, detect_stats, seg_results):
    summary_csv = out_dir / "detections_summary.csv"
    fields = ["idx", "group", "label", "score", "segmenter", "mask_area_px",
              "box_x1", "box_y1", "box_x2", "box_y2",
              "full_box_x1", "full_box_y1", "full_box_x2", "full_box_y2"]
    ox, oy = region_box[0], region_box[1]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in seg_results:
            bx1, by1, bx2, by2 = r["box"]
            w.writerow({
                "idx": r["idx"], "group": r["group"], "label": r["label"],
                "score": round(r["score"], 6), "segmenter": r["segmenter"],
                "mask_area_px": r["mask_area_px"],
                "box_x1": round(bx1, 2), "box_y1": round(by1, 2),
                "box_x2": round(bx2, 2), "box_y2": round(by2, 2),
                "full_box_x1": round(ox + bx1, 2), "full_box_y1": round(oy + by1, 2),
                "full_box_x2": round(ox + bx2, 2), "full_box_y2": round(oy + by2, 2),
            })
    # stats
    by_group = {}
    for r in seg_results:
        g = r["group"]
        agg = by_group.setdefault(g, {"count": 0, "mask_area_px": 0,
                                      "score_sum": 0.0, "segmenter": r["segmenter"]})
        agg["count"] += 1
        agg["mask_area_px"] += int(r["mask_area_px"])
        agg["score_sum"] += float(r["score"])
    for g, agg in by_group.items():
        agg["score_mean"] = round(agg["score_sum"] / max(agg["count"], 1), 4)
        del agg["score_sum"]
    stats = {
        "image_path": str(image_path),
        "region_box": [int(v) for v in region_box[:4]],
        "detect": detect_stats,
        "total_detections": len(seg_results),
        "by_group": by_group,
        "colors_bgr": {g: list(color_for(g)) for g in by_group},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "stats_by_group.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "count", "mask_area_px", "score_mean", "segmenter"])
        for g, agg in sorted(by_group.items()):
            w.writerow([g, agg["count"], agg["mask_area_px"], agg["score_mean"], agg["segmenter"]])
    return summary_csv


def detections_from_cutout_csv(csv_path):
    """Load curated cutout detections in the same shape used by build_overlays()."""
    seg_results = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            group = row.get("group_name") or row.get("group") or row.get("label") or "damage"
            score = float(row.get("score") or 0.0)
            box = [
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]),
            ]
            seg_results.append({
                "idx": int(row.get("id") or row.get("idx") or i),
                "group": group,
                "score": score,
                "label": row.get("label") or group,
                "box": box,
                "roi": [int(round(v)) for v in box],
                "segmenter": "box_only",
                "mask_area_px": 0,
            })
    return seg_results


def run_cutout_from_csv(*, image_path, csv_path, out_dir):
    """Export g8-style box overlays directly on an already-cropped image.

    This path is intentionally separate from run_one(): run_one detects on a full
    image, filters a region, then crops the rendered full-frame overlay. For Cau
    Tran Phu the input image is already the cutout, so using run_one would put the
    overlay in the wrong coordinate system/background.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = image_to_bgr_respecting_alpha(image_path, background=(0, 0, 0))
    oh, ow = full.shape[:2]
    seg_results = detections_from_cutout_csv(csv_path)
    cv2.imwrite(str(out_dir / "region_crop.png"), full)
    build_overlays(full, (0, 0), full, seg_results, out_dir, crop_box=None)
    summary_csv = export_data(
        out_dir,
        image_path,
        (0, 0, ow, oh),
        {"source_csv": str(csv_path), "kept_in_cutout": len(seg_results)},
        seg_results,
    )

    preview_w = min(1800, ow)
    if preview_w > 0:
        preview = cv2.imread(str(out_dir / "1_box_label.png"), cv2.IMREAD_COLOR)
        if preview is not None:
            preview_h = max(1, int(round(preview.shape[0] * preview_w / preview.shape[1])))
            preview = cv2.resize(preview, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / "1_box_label_preview.jpg"), preview,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(
        f"cutout overlay on {image_path} ({ow}x{oh}) -> {out_dir} "
        f"detections={len(seg_results)} summary={summary_csv}",
        flush=True,
    )
    return {"out_dir": str(out_dir), "detections": len(seg_results)}


def box_only_results(kept):
    return [{
        "idx": i,
        "group": d["group"],
        "score": float(d["score"]),
        "label": d.get("label") or d["group"],
        "box": d["box"],
        "roi": [int(round(v)) for v in d["box"]],
        "segmenter": "box_only",
        "mask_area_px": 0,
    } for i, d in enumerate(kept)]


def run_one(*, image_path, region_box, out_dir, sam_vit_h_ckpt, lora_model_dir,
            edge_pad_lr=10, edge_pad_tb=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = image_to_bgr_respecting_alpha(image_path, background=(0, 0, 0))
    oh, ow = full.shape[:2]
    rx1, ry1, rx2, ry2 = [int(round(v)) for v in region_box]
    rx1 = max(0, min(ow, rx1)); rx2 = max(0, min(ow, rx2))
    ry1 = max(0, min(oh, ry1)); ry2 = max(0, min(oh, ry2))
    cv2.imwrite(str(out_dir / "region_crop.png"), full[ry1:ry2, rx1:rx2])
    print(f"region [{rx1},{ry1},{rx2},{ry2}] on full {ow}x{oh} pad_lr={edge_pad_lr} pad_tb={edge_pad_tb}", flush=True)

    # 1. Detect damage on the FULL image (square tiling, no region distortion).
    kept_all, stats = detect_cracks_and_damage(full, edge_pad=2)

    # 2. Keep only boxes inside the region box. Left/right shrink by edge_pad_lr,
    #    top/bottom shrink by edge_pad_tb (set 0 to keep the full height, e.g. a column).
    ix1, iy1 = rx1 + edge_pad_lr, ry1 + edge_pad_tb
    ix2, iy2 = rx2 - edge_pad_lr, ry2 - edge_pad_tb
    kept = []
    for d in kept_all:
        bx1, by1, bx2, by2 = d["box"]
        if bx1 >= ix1 and by1 >= iy1 and bx2 <= ix2 and by2 <= iy2:
            kept.append(d)
    stats["kept_in_region"] = len(kept)
    print(f"detect stats: {stats}", flush=True)

    # 3. Segment + overlay on the full image. Each box is segmented on its own ROI.
    seg_results = segment_region(full, kept, out_dir,
                                 sam_vit_h_ckpt=sam_vit_h_ckpt, lora_model_dir=lora_model_dir)
    build_overlays(full, (0, 0), full, seg_results, out_dir, crop_box=(rx1, ry1, rx2, ry2))
    summary_csv = export_data(out_dir, image_path, (0, 0, ow, oh), stats, seg_results)
    print(f"done -> {out_dir} (detections={len(seg_results)}, summary={summary_csv})", flush=True)
    return {"out_dir": str(out_dir), "detections": len(seg_results), "stats": stats}

def run_one_trained(*, image_path, region_box, out_dir, sam_vit_h_ckpt, lora_model_dir,
                    detector, conf=0.1, edge_pad_lr=10, edge_pad_tb=10,
                    crop_to_region=True, render_bg=None, extra_column=False, region_offset_box=None,
                    no_filter=False, box_only=False, merge=False, detector_bg="black",
                    detector_max_side=0, detector_canvas_size=0, detect_classes=None):
    """Like run_one() but detect damage with a trained detector (yolo/stabledino).

    no_filter=True keeps every detection above conf and does not drop boxes outside
    the region box, for maximum coverage.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = image_to_bgr_respecting_alpha(image_path, background=(0, 0, 0))
    oh, ow = full.shape[:2]
    rx1, ry1, rx2, ry2 = [int(round(v)) for v in region_box]
    rx1 = max(0, min(ow, rx1)); rx2 = max(0, min(ow, rx2))
    ry1 = max(0, min(oh, ry1)); ry2 = max(0, min(oh, ry2))
    cv2.imwrite(str(out_dir / "region_crop.png"), full[ry1:ry2, rx1:rx2])
    print(f"[{detector}] region [{rx1},{ry1},{rx2},{ry2}] on full {ow}x{oh} no_filter={no_filter}", flush=True)

    bg_color = (255, 255, 255) if detector_bg == "white" else (0, 0, 0)
    detector_bgr = image_to_bgr_respecting_alpha(image_path, background=bg_color)
    det_h, det_w = detector_bgr.shape[:2]
    detector_scale = 1.0
    detector_offset = (0.0, 0.0)
    if int(detector_max_side or 0) > 0 and max(det_w, det_h) > int(detector_max_side):
        detector_scale = max(det_w, det_h) / float(detector_max_side)
        new_w = max(1, int(round(det_w / detector_scale)))
        new_h = max(1, int(round(det_h / detector_scale)))
        detector_bgr = cv2.resize(detector_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if int(detector_canvas_size or 0) > 0:
        canvas_size = int(detector_canvas_size)
        dh, dw = detector_bgr.shape[:2]
        if dw > canvas_size or dh > canvas_size:
            raise ValueError(f"Detector input {dw}x{dh} is larger than canvas {canvas_size}.")
        canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
        pad_x = (canvas_size - dw) // 2
        pad_y = (canvas_size - dh) // 2
        canvas[pad_y:pad_y + dh, pad_x:pad_x + dw] = detector_bgr
        detector_bgr = canvas
        detector_offset = (float(pad_x), float(pad_y))
    detector_input = out_dir / f"detector_input_{detector_bg}.png"
    cv2.imwrite(str(detector_input), detector_bgr)
    kept_all, stats = detect_damage_trained(full, detector_input, detector=detector, conf=conf,
                                             no_filter=no_filter, merge=merge,
                                             scale_to_original=detector_scale,
                                             offset_to_original=detector_offset,
                                             classes=detect_classes)
    stats["detector_input_background"] = detector_bg
    stats["detector_input_size"] = [int(detector_bgr.shape[1]), int(detector_bgr.shape[0])]
    stats["detector_scale_to_original"] = detector_scale
    stats["detector_offset_to_content"] = [detector_offset[0], detector_offset[1]]
    stats["detect_classes"] = list(detect_classes or TRAINED_CLASSES)

    if no_filter:
        # Keep everything; do not restrict to the region box.
        kept = kept_all
    else:
        ix1, iy1 = rx1 + edge_pad_lr, ry1 + edge_pad_tb
        ix2, iy2 = rx2 - edge_pad_lr, ry2 - edge_pad_tb
        kept = [d for d in kept_all
                if d["box"][0] >= ix1 and d["box"][1] >= iy1 and d["box"][2] <= ix2 and d["box"][3] <= iy2]
    stats["kept_in_region"] = len(kept)
    print(f"[{detector}] detect stats: {stats}", flush=True)

    render_full = full
    crop_box = (rx1, ry1, rx2, ry2) if crop_to_region else None
    extra_box = None
    if extra_column:
        extra_box = (rx1, ry1, rx2, ry2, "column")

    if box_only:
        seg_results = box_only_results(kept)
        build_overlays(render_full, (0, 0), render_full, seg_results, out_dir,
                       crop_box=crop_box, extra_box=extra_box)
        render_text_x3(out_dir, render_full, seg_results, crop_box, extra_box=extra_box)
        summary_csv = export_data(out_dir, image_path, (0, 0, ow, oh), stats, seg_results)
        preview = cv2.imread(str(out_dir / "1_box_label.png"), cv2.IMREAD_COLOR)
        if preview is not None:
            preview_w = min(1800, preview.shape[1])
            preview_h = max(1, int(round(preview.shape[0] * preview_w / preview.shape[1])))
            preview = cv2.resize(preview, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / "1_box_label_preview.jpg"), preview,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[{detector}] done -> {out_dir} (detections={len(seg_results)}, summary={summary_csv})", flush=True)
        return {"out_dir": str(out_dir), "detections": len(seg_results), "stats": stats}

    seg_results = segment_region(full, kept, out_dir,
                                 sam_vit_h_ckpt=sam_vit_h_ckpt, lora_model_dir=lora_model_dir)
    build_overlays(render_full, (0, 0), render_full, seg_results, out_dir, crop_box=crop_box, extra_box=extra_box)
    render_text_x3(out_dir, render_full, seg_results, crop_box, extra_box=extra_box)
    summary_csv = export_data(out_dir, image_path, (0, 0, ow, oh), stats, seg_results)
    print(f"[{detector}] done -> {out_dir} (detections={len(seg_results)})", flush=True)
    return {"out_dir": str(out_dir), "detections": len(seg_results), "stats": stats}


def region_from_column_db(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT x1,y1,x2,y2 FROM detections LIMIT 1").fetchone()
    finally:
        conn.close()
    return [float(v) for v in row]


def region_house_via_sam(image_path, sam_vit_h_ckpt, *, max_side=1024):
    """Detect the whole-house region on a downscaled image, return box in full-res coords."""
    from segmentation.sam.runtime.engine import SamParams, SamRunner

    with Image.open(image_path) as im:
        rgb = np.array(im.convert("RGB"))
    oh, ow = rgb.shape[:2]
    scale = min(max_side / ow, max_side / oh, 1.0)
    work = cv2.resize(rgb, (int(round(ow * scale)), int(round(oh * scale))), interpolation=cv2.INTER_AREA)
    wh, ww = work.shape[:2]
    runner = SamRunner()
    params = SamParams(sam_checkpoint=str(sam_vit_h_ckpt), sam_model_type="vit_h", device="auto")
    predictor, _ = runner.ensure_model_loaded(params, log_fn=lambda s: print(s, flush=True))
    predictor.set_image(work)
    box = np.array([int(ww * 0.03), int(wh * 0.02), int(ww * 0.97), int(wh * 0.995)], dtype=np.float32)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    best = 0; best_s = -1e9
    center = np.array([ww / 2.0, wh / 2.0])
    for i, m in enumerate(masks):
        mm = m.astype(bool)
        if not mm.any():
            continue
        ys, xs = np.where(mm)
        bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max(), ys.max()
        area = float(mm.sum()) / (ww * wh)
        hr = (by2 - by1) / wh; wr = (bx2 - bx1) / ww
        cd = float(np.linalg.norm((np.array([xs.mean(), ys.mean()]) - center) / np.array([ww, wh])))
        s = area * 1.4 + hr * 1.2 + wr * 0.6 - cd * 0.4 + float(scores[i]) * 0.2
        if s > best_s:
            best_s = s; best = i
    mm = masks[best].astype(bool)
    ys, xs = np.where(mm)
    bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [bx1 / scale, by1 / scale, bx2 / scale, by2 / scale]


def main():
    lab = Path("/Users/nguyenquangvinh/Desktop/Lab")
    results_root = lab / "model_with_inference" / "semi_labeling_training" / "results"
    sam_vit_h = lab / "DamageDetector" / "models" / "sam" / "sam_vit_h_4b8939.pth"
    lora_dir = lab / "model_with_inference" / "crack_segmentation" / "sam_lora_hq_coarse_refine" / "model"

    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["g8", "ntt", "cau_tran_phu", "both", "all",
                                        "g8_trained", "ntt_trained", "cau_tran_phu_trained",
                                        "trained"], default="both")
    p.add_argument("--detector", choices=["yolo", "stabledino", "both"], default="both",
                   help="Trained detector(s) to use for *_trained targets.")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--no-filter", dest="no_filter", action="store_true",
                   help="Keep every detection above conf: no NMS, no area cap, no edge/region drop.")
    p.add_argument("--merge", dest="merge", action="store_true",
                   help="Merge same-class boxes that overlap/are near into one big box (cover whole streaks).")
    p.add_argument("--detector-bg", choices=["black", "white"], default="black",
                   help="Background used when converting alpha/cutout inputs to detector PNG.")
    p.add_argument("--box-only", dest="box_only", action="store_true",
                   help="Render detector boxes only; skip SAM segmentation for quick inspection.")
    p.add_argument("--detector-max-side", type=int, default=0,
                   help="Resize detector input so its largest side equals this value, then scale boxes back.")
    p.add_argument("--detector-canvas-size", type=int, default=0,
                   help="Center resized detector input on a square canvas of this size before detection.")
    p.add_argument("--classes", default="",
                   help="Comma-separated trained classes to detect, e.g. spall or crack,mold.")
    args = p.parse_args()

    detectors = ["yolo", "stabledino"] if args.detector == "both" else [args.detector]
    detect_classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()] or None
    if detect_classes:
        bad = [c for c in detect_classes if c not in TRAINED_CLASSES]
        if bad:
            raise ValueError(f"Unsupported trained classes: {bad}. Valid: {TRAINED_CLASSES}")

    if args.target in ("g8", "both", "all"):
        run_one(
            image_path=lab / "data" / "HinhAnhThucTe" / "1.JPG",
            region_box=region_from_column_db(results_root / "g8" / "step_gdino_column_top1" / "detections.sqlite3"),
            out_dir=results_root / "g8" / "damage_segmentation",
            sam_vit_h_ckpt=sam_vit_h, lora_model_dir=lora_dir,
            edge_pad_lr=10, edge_pad_tb=0,
        )
    if args.target in ("ntt", "both", "all"):
        ntt_img = lab / "data" / "HinhAnhThucTe" / "NTT - 16m Lan 3.tif"
        run_one(
            image_path=ntt_img,
            region_box=region_house_via_sam(ntt_img, sam_vit_h),
            out_dir=results_root / "nha_truyen_thong" / "damage_segmentation",
            sam_vit_h_ckpt=sam_vit_h, lora_model_dir=lora_dir,
            edge_pad_lr=10, edge_pad_tb=10,
        )
    if args.target in ("g8_trained", "trained"):
        g8_region = region_from_column_db(results_root / "g8" / "step_gdino_column_top1" / "detections.sqlite3")
        for det in detectors:
            suffix = "_white_bg" if args.detector_bg == "white" else ""
            if args.detector_max_side > 0:
                suffix += f"_max{args.detector_max_side}"
            if args.detector_canvas_size > 0:
                suffix += f"_canvas{args.detector_canvas_size}"
            if detect_classes:
                suffix += "_" + "_".join(detect_classes)
            run_one_trained(
                image_path=lab / "data" / "HinhAnhThucTe" / "1.JPG",
                region_box=g8_region,
                out_dir=results_root / "g8" / f"damage_segmentation_{det}{suffix}",
                sam_vit_h_ckpt=sam_vit_h, lora_model_dir=lora_dir,
                detector=det, conf=args.conf,
                edge_pad_lr=10, edge_pad_tb=0,
                crop_to_region=False, extra_column=True,
                no_filter=args.no_filter,
                merge=args.merge,
                detector_bg=args.detector_bg,
                box_only=args.box_only,
                detector_max_side=args.detector_max_side,
                detector_canvas_size=args.detector_canvas_size,
                detect_classes=detect_classes,
            )
    if args.target in ("ntt_trained", "trained"):
        # NTT detects directly on the already-cut clean house image (no re-crop).
        ntt_clean = (results_root / "nha_truyen_thong" / "damage_segmentation_clean_overlays"
                     / "clean_house_bg.png")
        with Image.open(ntt_clean) as _im:
            cw, ch = _im.size
        for det in detectors:
            suffix = "_white_bg" if args.detector_bg == "white" else ""
            if args.detector_max_side > 0:
                suffix += f"_max{args.detector_max_side}"
            if args.detector_canvas_size > 0:
                suffix += f"_canvas{args.detector_canvas_size}"
            if detect_classes:
                suffix += "_" + "_".join(detect_classes)
            run_one_trained(
                image_path=ntt_clean,
                region_box=[0, 0, cw, ch],
                out_dir=results_root / "nha_truyen_thong" / f"damage_segmentation_{det}{suffix}",
                sam_vit_h_ckpt=sam_vit_h, lora_model_dir=lora_dir,
                detector=det, conf=args.conf,
                edge_pad_lr=0, edge_pad_tb=0,
                crop_to_region=False,
                no_filter=args.no_filter,
                merge=args.merge,
                detector_bg=args.detector_bg,
                box_only=args.box_only,
                detector_max_side=args.detector_max_side,
                detector_canvas_size=args.detector_canvas_size,
                detect_classes=detect_classes,
            )
    if args.target in ("cau_tran_phu_trained", "trained"):
        ctp_dir = results_root / "cau_tran_phu"
        ctp_img = ctp_dir / "DJI_0093_point_sam_cutout.png"
        with Image.open(ctp_img) as im:
            ctp_region = (0, 0, im.width, im.height)
        for det in detectors:
            suffix = "_white_bg" if args.detector_bg == "white" else ""
            if args.detector_max_side > 0:
                suffix += f"_max{args.detector_max_side}"
            if args.detector_canvas_size > 0:
                suffix += f"_canvas{args.detector_canvas_size}"
            if detect_classes:
                suffix += "_" + "_".join(detect_classes)
            run_one_trained(
                image_path=ctp_img,
                region_box=ctp_region,
                out_dir=ctp_dir / f"damage_segmentation_{det}{suffix}",
                sam_vit_h_ckpt=sam_vit_h, lora_model_dir=lora_dir,
                detector=det, conf=args.conf,
                edge_pad_lr=0, edge_pad_tb=0,
                crop_to_region=True,
                no_filter=args.no_filter,
                merge=args.merge,
                box_only=True,
                detector_bg=args.detector_bg,
                detector_max_side=args.detector_max_side,
                detector_canvas_size=args.detector_canvas_size,
                detect_classes=detect_classes,
            )
    if args.target in ("cau_tran_phu", "all"):
        ctp_dir = results_root / "cau_tran_phu"
        run_cutout_from_csv(
            image_path=ctp_dir / "DJI_0093_point_sam_cutout.png",
            csv_path=ctp_dir / "detections_final_ided.csv",
            out_dir=ctp_dir,
        )


if __name__ == "__main__":
    raise SystemExit(main())
