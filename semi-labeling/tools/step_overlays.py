#!/usr/bin/env python3
"""Render per-step overlays for a resemi run.

For each pipeline step, draws every detection box (from crop_views) on its source
image, colored/labeled by that step's own output table. One folder of overlays
per step so you can see how labels/decisions evolve across the 9 steps.

Usage:
  python -m tools.step_overlays --db <resemi.sqlite3> --run-id <id> \
      --image-root <dir> --out <overlays_dir> [--step step01 ...] [--limit N]
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.runtime.paths import default_image_root, default_resemi_db  # noqa: E402


LABEL_COLORS = {
    "crack": (248, 81, 73),
    "mold": (63, 185, 80),
    "spall": (210, 153, 34),
    "stain": (188, 140, 60),
    "reject": (130, 130, 130),
}
DECISION_COLORS = {
    "auto_accept": (63, 185, 80),
    "auto_accept_low_priority": (120, 200, 120),
    "relabel_candidate": (210, 153, 34),
    "suspect": (243, 156, 18),
    "reject": (231, 76, 60),
    "promote_clean": (63, 185, 80),
    "defer_review": (243, 156, 18),
}
DEFAULT_COLOR = (88, 166, 255)


@dataclass(frozen=True)
class OverlayBox:
    result_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    label: str       # text drawn on the box
    color_key: str   # used to pick a color


# Each step: SQL returning (result_id, x1, y1, x2, y2, image_rel_path, label, color_key).
# All join crop_views (cv) for geometry; view_name tight is the canonical box.
def _step_query(step: str) -> str | None:
    base = """
        SELECT cv.result_id, cv.x1, cv.y1, cv.x2, cv.y2, cv.image_rel_path,
               {label} AS label, {color} AS color_key
        FROM crop_views cv
        {join}
        WHERE cv.run_id = :run_id AND cv.view_name = 'tight'
    """
    q = {
        # step01: initial semantic label + decision
        "step01": base.format(
            label="sd.initial_label", color="sd.initial_label",
            join="JOIN semantic_decisions sd ON sd.run_id=cv.run_id AND sd.result_id=cv.result_id",
        ),
        # step04: core cluster membership (label = cluster id, color by is_core)
        "step04": base.format(
            label="('c' || cm.core_cluster_id)", color="cc.label",
            join=("JOIN core_cluster_members cm ON cm.run_id=cv.run_id AND cm.result_id=cv.result_id "
                  "JOIN core_clusters cc ON cc.run_id=cv.run_id AND cc.core_cluster_id=cm.core_cluster_id"),
        ),
        # step05: prototype top label + similarity
        "step05": base.format(
            label="ps.top_prototype_label", color="ps.top_prototype_label",
            join="JOIN prototype_scores ps ON ps.result_id=cv.result_id",
        ),
        # step06: reliability final label + score (color by decision in semantic_decisions)
        "step06": base.format(
            label="(sd.final_label || ' ' || printf('%.2f', rs.reliability_score))", color="sd.decision_type",
            join=("JOIN reliability_scores rs ON rs.run_id=cv.run_id AND rs.result_id=cv.result_id "
                  "JOIN semantic_decisions sd ON sd.run_id=cv.run_id AND sd.result_id=cv.result_id"),
        ),
        # step07: decision policy -> cleaned (green) vs review (amber)
        "step07_cleaned": base.format(
            label="cl.final_label", color="cl.decision_type",
            join="JOIN cleaned_labels cl ON cl.run_id=cv.run_id AND cl.result_id=cv.result_id",
        ),
        "step07_review": base.format(
            label="(rq.suggested_label || ' [' || rq.queue_type || ']')", color="rq.queue_type",
            join="JOIN review_queue rq ON rq.run_id=cv.run_id AND rq.result_id=cv.result_id",
        ),
        # step08: classifier top-1 prediction
        "step08": """
            SELECT cv.result_id, cv.x1, cv.y1, cv.x2, cv.y2, cv.image_rel_path,
                   p.label AS label, p.label AS color_key
            FROM crop_views cv
            JOIN (
              SELECT result_id, label,
                     ROW_NUMBER() OVER (PARTITION BY result_id ORDER BY probability DESC) rn
              FROM classifier_predictions WHERE classifier_run_id = (
                  SELECT classifier_run_id FROM classifier_runs WHERE run_id = :run_id
                  ORDER BY created_at_utc DESC LIMIT 1)
            ) p ON p.result_id = cv.result_id AND p.rn = 1
            WHERE cv.run_id = :run_id AND cv.view_name = 'tight'
        """,
        # step09: self-training action (promote vs defer)
        "step09": base.format(
            label="(st.predicted_label || ' [' || st.action || ']')", color="st.action",
            join=("JOIN self_training_promotions st ON st.result_id=cv.result_id AND "
                  "st.self_training_run_id=(SELECT self_training_run_id FROM self_training_runs "
                  "WHERE run_id=:run_id ORDER BY created_at_utc DESC LIMIT 1)"),
        ),
    }
    return q.get(step)


STEP_ORDER = ["step01", "step04", "step05", "step06", "step07_cleaned", "step07_review", "step08", "step09"]


def connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def pick_color(color_key: str) -> tuple[int, int, int]:
    key = str(color_key or "").strip().lower()
    return LABEL_COLORS.get(key) or DECISION_COLORS.get(key) or DEFAULT_COLOR


def render(image_path: Path, boxes: list[OverlayBox], out_path: Path) -> None:
    with Image.open(image_path) as im:
        canvas = im.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for b in boxes:
        color = pick_color(b.color_key)
        draw.rectangle((b.x1, b.y1, b.x2, b.y2), outline=color, width=3)
        text = f"{b.result_id} {b.label}"
        tb = draw.textbbox((b.x1, b.y1), text, font=font)
        draw.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2),
                       fill=(max(0, color[0] - 50), max(0, color[1] - 50), max(0, color[2] - 50)))
        draw.text((b.x1, b.y1), text, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render per-step overlays for a resemi run.")
    p.add_argument("--db", default=str(default_resemi_db()))
    p.add_argument("--run-id", required=True)
    p.add_argument("--image-root", default=str(default_image_root()))
    p.add_argument("--out", required=True, help="Output root; one subfolder per step.")
    p.add_argument("--step", action="append", default=[], help="Limit to these steps; default all.")
    p.add_argument("--limit", type=int, default=0, help="Max images per step. 0 = all.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    steps = [s for s in (args.step or STEP_ORDER) if _step_query(s)]

    conn = connect_ro(db_path)
    try:
        for step in steps:
            rows = conn.execute(_step_query(step), {"run_id": args.run_id}).fetchall()
            by_image: dict[str, list[OverlayBox]] = {}
            for r in rows:
                by_image.setdefault(str(r["image_rel_path"]), []).append(
                    OverlayBox(int(r["result_id"]), float(r["x1"]), float(r["y1"]),
                               float(r["x2"]), float(r["y2"]), str(r["label"]), str(r["color_key"]))
                )
            images = list(by_image.items())
            if int(args.limit) > 0:
                images = images[: int(args.limit)]
            step_dir = out_root / step
            rendered = 0
            for rel, boxes in images:
                src = image_root / rel
                if not src.is_file():
                    continue
                render(src, boxes, step_dir / f"{Path(rel).stem}__{step}.png")
                rendered += 1
            print(f"{step:16} images={len(by_image)} boxes={len(rows)} rendered={rendered} -> {step_dir}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
