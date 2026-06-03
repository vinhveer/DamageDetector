"""Eyeball probe: dump detector-spall crops (stratified by GDINO score) with the
OpenCLIP verdict + per-label cosine in the filename, so we can SEE whether the
'spall' boxes are really spall (GDINO right) or really mold/crack (OpenCLIP right).

Reuses cached ViT-H. Writes crops to tmp_spall_crops/. Read-only on DB.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

SRC_DB = Path(r"C:\Users\Dell Precision 7810\Desktop\quangvinh_workspace\model_with_inference\semi_labeling\pipeline.sqlite3")
IMAGE_ROOT = Path(r"C:\Users\Dell Precision 7810\Desktop\quangvinh_workspace\data\HinhAnh")
OUT = Path(r"C:\Users\Dell Precision 7810\Desktop\quangvinh_workspace\DamageDetector\semi-labeling\tmp_spall_crops")
MODEL, PRETRAINED = "ViT-H-14", "laion2b_s32b_b79k"

CRACK = [
    "a close-up photo of a thin narrow crack line on concrete surface",
    "a long dark fracture line with sharp edges on a wall",
    "a fine irregular hairline crack on plaster or concrete",
    "a linear split with clear boundaries in building material",
    "a jagged crack line running across a surface",
]
MOLD = [
    "a close-up photo of a mold stain patch on a wall surface",
    "a dark or green mold area with blurry edges",
    "a dirty discoloration patch without sharp lines",
    "mildew or moss growing on concrete surface",
    "an irregular stain area with soft boundaries",
]
SPALL = [
    "a close-up photo of broken concrete surface with missing material",
    "a chipped concrete area with rough texture",
    "spalling damage exposing inner material",
    "a hole or flaked region on concrete surface",
    "a damaged surface with pieces falling off",
]
LABELS = ["crack", "mold", "spall"]
GROUPS = {"crack": CRACK, "mold": MOLD, "spall": SPALL}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()
    conn = sqlite3.connect(str(SRC_DB)); conn.row_factory = sqlite3.Row
    # Stratify detector-spall by score: top, mid, low — 20 each.
    rows = []
    for band, order in [("hi", "DESC"), ("lo", "ASC")]:
        rows += [(band, r) for r in conn.execute(
            f"""SELECT d.detection_id,d.label det_label,d.score,d.x1,d.y1,d.x2,d.y2,
                       i.rel_path,i.width,i.height
                FROM detections d JOIN images i ON i.image_id=d.image_id
                WHERE d.stage='final' AND d.label='spall'
                ORDER BY d.score {order} LIMIT 25""").fetchall()]
    # mid band
    n_spall = conn.execute("SELECT COUNT(*) n FROM detections WHERE stage='final' AND label='spall'").fetchone()["n"]
    rows += [("mid", r) for r in conn.execute(
        """SELECT d.detection_id,d.label det_label,d.score,d.x1,d.y1,d.x2,d.y2,
                  i.rel_path,i.width,i.height
           FROM detections d JOIN images i ON i.image_id=d.image_id
           WHERE d.stage='final' AND d.label='spall'
           ORDER BY d.score DESC LIMIT 25 OFFSET ?""", (n_spall // 2,)).fetchall()]

    crops, meta = [], []
    for band, r in rows:
        p = IMAGE_ROOT / str(r["rel_path"])
        if not p.is_file():
            continue
        w, h = int(r["width"]), int(r["height"])
        x1 = max(0, min(w - 1, int(r["x1"]))); y1 = max(0, min(h - 1, int(r["y1"])))
        x2 = max(0, min(w, int(r["x2"])));     y2 = max(0, min(h, int(r["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue
        with Image.open(p) as im:
            crops.append(im.convert("RGB").crop((x1, y1, x2, y2)))
        meta.append((band, r, (x2 - x1), (y2 - y1)))
    print(f"{len(crops)} detector-spall crops (hi/mid/lo score bands)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(MODEL)
    print(f"model loaded (cache) in {time.time()-t0:.1f}s")

    def enc_text(texts):
        with torch.inference_mode():
            f = model.encode_text(tokenizer(texts).to(device))
            return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()
    tf = {l: enc_text(GROUPS[l]) for l in LABELS}

    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
        feats = []
        for i in range(0, len(crops), 8):
            t = torch.stack([preprocess(c) for c in crops[i:i+8]]).to(device)
            f = model.encode_image(t); f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu().numpy())
    img = np.concatenate(feats, 0)

    # per-label mean cosine
    mean_cos = {l: (img @ tf[l].T).mean(1) for l in LABELS}
    band_keep = {"hi": [0, 0], "mid": [0, 0], "lo": [0, 0]}  # [kept_spall, total]
    for idx, (band, r, bw, bh) in enumerate(meta):
        sims = {l: float(mean_cos[l][idx]) for l in LABELS}
        pred = max(sims, key=sims.get)
        band_keep[band][1] += 1
        if pred == "spall":
            band_keep[band][0] += 1
        fn = (f"{band}_score{float(r['score']):.2f}_pred-{pred}"
              f"_sp{sims['spall']:.3f}_mo{sims['mold']:.3f}_cr{sims['crack']:.3f}"
              f"_{bw}x{bh}_det{r['detection_id']}.png")
        crops[idx].save(OUT / fn)

    print("\n=== how often OpenCLIP keeps detector-spall as spall, by GDINO score band ===")
    for b in ["hi", "mid", "lo"]:
        k, n = band_keep[b]
        print(f"  {b}: {k}/{n} kept as spall")
    print("\n=== mean cosine per label (avg over all sampled spall crops) ===")
    for l in LABELS:
        print(f"  {l}: {mean_cos[l].mean():.4f}")
    print(f"\ncrops saved to: {OUT}")


if __name__ == "__main__":
    main()
