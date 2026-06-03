"""One-off probe: encode a sample of crops ONCE with ViT-H, then try several
prompt / pooling configs to see which recovers spall without wrecking mold/crack.

Image features do not depend on prompts, so we encode once and re-score text
cheaply. Read-only on the source DB; writes nothing.

Run:  python tmp_spall_probe.py --sample 600
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

SRC_DB = Path(r"C:\Users\Dell Precision 7810\Desktop\quangvinh_workspace\model_with_inference\semi_labeling\pipeline.sqlite3")
IMAGE_ROOT = Path(r"C:\Users\Dell Precision 7810\Desktop\quangvinh_workspace\data\HinhAnh")
MODEL = "ViT-H-14"
PRETRAINED = "laion2b_s32b_b79k"

# ── Prompt sets ──────────────────────────────────────────────────────────────
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
SPALL_OLD = [
    "a close-up photo of broken concrete surface with missing material",
    "a chipped concrete area with rough texture",
    "spalling damage exposing inner material",
    "a hole or flaked region on concrete surface",
    "a damaged surface with pieces falling off",
]
SPALL_NEW = [
    "a close-up photo of concrete where the surface layer has broken off exposing rough stone aggregate underneath",
    "a crater-like cavity in a concrete wall with crumbling broken edges",
    "a patch of concrete with chunks of material missing leaving a rough pitted hole",
    "exposed rusty steel reinforcement bars in broken spalled concrete",
    "a deep chipped-out hole in concrete with jagged broken edges and loose debris",
    "a rough pockmarked concrete surface with shallow pits where the top layer flaked away",
]

LABELS = ["crack", "mold", "spall"]


def sample_detections(conn: sqlite3.Connection, n: int) -> list[sqlite3.Row]:
    """Stratified: prioritise detector-spall (where the bug bites) + mold/crack."""
    rows: list[sqlite3.Row] = []
    quota = {"spall": n // 2, "mold": n // 4, "crack": n - n // 2 - n // 4}
    for lab, k in quota.items():
        rows += conn.execute(
            """
            SELECT d.detection_id, d.label AS det_label, d.score, d.x1, d.y1, d.x2, d.y2,
                   i.rel_path, i.width, i.height
            FROM detections d JOIN images i ON i.image_id = d.image_id
            WHERE d.stage='final' AND d.label=?
            ORDER BY d.detection_id
            LIMIT ?
            """,
            (lab, k),
        ).fetchall()
    return rows


def load_crop(row: sqlite3.Row) -> Image.Image | None:
    path = IMAGE_ROOT / str(row["rel_path"])
    if not path.is_file():
        return None
    w, h = int(row["width"]), int(row["height"])
    x1 = max(0, min(w - 1, int(row["x1"])));  y1 = max(0, min(h - 1, int(row["y1"])))
    x2 = max(0, min(w, int(row["x2"])));      y2 = max(0, min(h, int(row["y2"])))
    if x2 <= x1 or y2 <= y1:
        return None
    with Image.open(path) as im:
        return im.convert("RGB").crop((x1, y1, x2, y2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    conn = sqlite3.connect(str(SRC_DB)); conn.row_factory = sqlite3.Row
    rows = sample_detections(conn, args.sample)
    crops, kept = [], []
    for r in rows:
        c = load_crop(r)
        if c is not None:
            crops.append(c); kept.append(r)
    print(f"sampled {len(rows)} detections, {len(crops)} crops decoded")
    det_counts = Counter(r["det_label"] for r in kept)
    print("detector labels in sample:", dict(det_counts))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL} ({PRETRAINED}) on {device} ... (first run downloads ~3.9GB)")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(MODEL)
    print(f"model loaded in {time.time()-t0:.1f}s")

    # ── encode images ONCE ───────────────────────────────────────────────────
    t0 = time.time()
    feats = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
        for i in range(0, len(crops), args.batch):
            batch = crops[i:i + args.batch]
            t = torch.stack([preprocess(c) for c in batch]).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu().numpy())
    img_feats = np.concatenate(feats, 0)  # (N, D)
    enc_dt = time.time() - t0
    cps = len(crops) / enc_dt
    print(f"\nENCODED {len(crops)} crops in {enc_dt:.1f}s = {cps:.1f} crops/s")
    print(f"  -> extrapolate to 183,639 crops: ~{183639/cps/60:.1f} min GPU (decode/IO extra)\n")

    def encode_text(texts: list[str]) -> np.ndarray:
        with torch.inference_mode():
            tok = tokenizer(texts).to(device)
            f = model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    det = np.array([r["det_label"] for r in kept])

    def report(name: str, pred: np.ndarray) -> None:
        print(f"=== {name} ===")
        print("  predicted dist:", dict(Counter(pred.tolist())))
        # recovery: of detector-spall, how many predicted spall
        for lab in LABELS:
            mask = det == lab
            if mask.sum():
                keep = (pred[mask] == lab).mean()
                print(f"  detector={lab:5s} n={mask.sum():4d}  kept as {lab}: {keep*100:5.1f}%")
        print()

    # Config A: baseline — current prompts, sum-of-softmax-over-all-prompts pooling
    def pooled_sum_softmax(spall_prompts):
        groups = {"crack": CRACK, "mold": MOLD, "spall": spall_prompts}
        labs, texts = [], []
        for lab in LABELS:
            for p in groups[lab]:
                labs.append(lab); texts.append(p)
        tf = encode_text(texts)                       # (P, D)
        logits = 100.0 * img_feats @ tf.T             # (N, P)
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)           # softmax over all prompts
        labs = np.array(labs)
        scores = np.stack([probs[:, labs == l].sum(1) for l in LABELS], 1)
        return np.array(LABELS)[scores.argmax(1)], scores

    # Config C: per-label MEAN cosine sim, softmax over 3 labels w/ temperature
    def pooled_mean(spall_prompts, T=0.01, prior=None):
        groups = {"crack": CRACK, "mold": MOLD, "spall": spall_prompts}
        per = []
        for lab in LABELS:
            tf = encode_text(groups[lab])             # (k, D)
            sims = img_feats @ tf.T                    # (N, k) cosine
            per.append(sims.mean(1))
        S = np.stack(per, 1)                           # (N, 3)
        if prior is not None:
            S = S + np.array([prior.get(l, 0.0) for l in LABELS])
        e = np.exp((S - S.max(1, keepdims=True)) / T)
        e = e / e.sum(1, keepdims=True)
        return np.array(LABELS)[e.argmax(1)], e

    # Config D: per-label MAX cosine sim
    def pooled_max(spall_prompts):
        groups = {"crack": CRACK, "mold": MOLD, "spall": spall_prompts}
        per = [ (img_feats @ encode_text(groups[lab]).T).max(1) for lab in LABELS ]
        S = np.stack(per, 1)
        return np.array(LABELS)[S.argmax(1)], S

    report("A. BASELINE (old spall, sum-softmax)", pooled_sum_softmax(SPALL_OLD)[0])
    report("B. NEW spall prompts, sum-softmax", pooled_sum_softmax(SPALL_NEW)[0])
    report("C. NEW spall, MEAN-cosine softmax T=0.01", pooled_mean(SPALL_NEW)[0])
    report("D. NEW spall, MAX-cosine", pooled_max(SPALL_NEW)[0])
    report("E. NEW spall, MEAN + spall prior +0.02", pooled_mean(SPALL_NEW, prior={"spall": 0.02})[0])


if __name__ == "__main__":
    main()
