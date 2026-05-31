#!/usr/bin/env python3
"""Tool — render bbox cleanup overlay images for manual review."""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.runtime.paths import LAB_ROOT, default_image_root, default_resemi_db  # noqa: E402


@dataclass(frozen=True)
class OverlayBox:
    result_id: int
    image_rel_path: str
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    decision_type: str
    box_quality_score: float
    semantic_decision: str
    reliability_score: float
    reason_codes_json: str


COLORS = {
    "keep_representative": (40, 180, 99),
    "keep_long_crack_parent": (52, 152, 219),
    "drop_nested_duplicate": (149, 165, 166),
    "suspect_composite_box": (231, 76, 60),
    "suspect_broad_box": (243, 156, 18),
    "manual_box_review": (155, 89, 182),
}


def default_output_dir(run_id: str) -> Path:
    return LAB_ROOT / "infer_results" / "semi-labeling" / "resemi" / "overlays" / f"{run_id}_review30"


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_image_path(row: sqlite3.Row, image_root: Path | None) -> Path:
    rel_path = str(row["rel_path"] or "").strip()
    stored_path = str(row["path"] or "").strip()
    source_input_dir = Path(str(row["input_dir"] or "")).expanduser()
    candidates: list[Path] = []
    if image_root is not None:
        root = Path(image_root).expanduser().resolve()
        if rel_path:
            candidates.append(root / rel_path)
        if stored_path:
            candidates.append(root / Path(stored_path).name)
    if stored_path:
        stored = Path(stored_path).expanduser()
        candidates.append(stored if stored.is_absolute() else source_input_dir / stored_path)
    if rel_path:
        candidates.append(source_input_dir / rel_path)
    if stored_path:
        candidates.append(source_input_dir / Path(stored_path).name)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    if image_root is not None and rel_path:
        return (Path(image_root).expanduser().resolve() / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()


def load_run(resemi_conn: sqlite3.Connection, run_id: str) -> sqlite3.Row:
    row = resemi_conn.execute("SELECT * FROM resemi_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"Run not found: {run_id}")
    return row


def select_images(resemi_conn: sqlite3.Connection, *, run_id: str, limit: int) -> list[str]:
    box_graph_run_id = f"{run_id}_box_graph_v1"
    rows = resemi_conn.execute(
        """
        SELECT image_rel_path,
               SUM(CASE WHEN decision_type IN ('suspect_composite_box', 'suspect_broad_box', 'manual_box_review') THEN 1 ELSE 0 END) AS review_count,
               SUM(CASE WHEN decision_type = 'drop_nested_duplicate' THEN 1 ELSE 0 END) AS drop_count,
               SUM(CASE WHEN decision_type = 'keep_long_crack_parent' THEN 1 ELSE 0 END) AS long_crack_count,
               COUNT(*) AS total_count
        FROM box_cleanup_decisions
        WHERE box_graph_run_id = ?
        GROUP BY image_rel_path
        HAVING review_count > 0 OR drop_count > 0 OR long_crack_count > 0
        ORDER BY review_count DESC, drop_count DESC, long_crack_count DESC, total_count DESC, image_rel_path
        LIMIT ?
        """,
        (box_graph_run_id, int(limit)),
    ).fetchall()
    return [str(row["image_rel_path"]) for row in rows]


def load_boxes(
    resemi_conn: sqlite3.Connection,
    *,
    run_id: str,
    image_rel_path: str,
    include_decisions: set[str] | None,
) -> list[OverlayBox]:
    box_graph_run_id = f"{run_id}_box_graph_v1"
    rows = resemi_conn.execute(
        """
        SELECT c.result_id, c.image_rel_path, c.x1, c.y1, c.x2, c.y2,
               b.label, b.decision_type, b.box_quality_score, b.reason_codes_json,
               s.decision_type AS semantic_decision, s.reliability_score
        FROM crop_views c
        JOIN box_cleanup_decisions b ON b.result_id = c.result_id
        JOIN semantic_decisions s ON s.run_id = c.run_id AND s.result_id = c.result_id
        WHERE c.run_id = ?
          AND c.image_rel_path = ?
          AND c.view_name IN ('openclip_crop', 'tight')
          AND b.box_graph_run_id = ?
        ORDER BY
          CASE b.decision_type
            WHEN 'drop_nested_duplicate' THEN 1
            WHEN 'keep_representative' THEN 2
            WHEN 'keep_long_crack_parent' THEN 3
            WHEN 'suspect_composite_box' THEN 4
            WHEN 'suspect_broad_box' THEN 5
            ELSE 6
          END,
          b.box_quality_score DESC,
          c.result_id
        """,
        (run_id, image_rel_path, box_graph_run_id),
    ).fetchall()
    boxes = [
        OverlayBox(
            result_id=int(row["result_id"]),
            image_rel_path=str(row["image_rel_path"]),
            x1=float(row["x1"]),
            y1=float(row["y1"]),
            x2=float(row["x2"]),
            y2=float(row["y2"]),
            label=str(row["label"]),
            decision_type=str(row["decision_type"]),
            box_quality_score=float(row["box_quality_score"]),
            semantic_decision=str(row["semantic_decision"]),
            reliability_score=float(row["reliability_score"]),
            reason_codes_json=str(row["reason_codes_json"]),
        )
        for row in rows
    ]
    if include_decisions is None:
        return boxes
    return [box for box in boxes if box.decision_type in include_decisions]


def load_source_image_row(source_conn: sqlite3.Connection, *, semantic_run_id: str, image_rel_path: str) -> sqlite3.Row | None:
    return source_conn.execute(
        """
        SELECT DISTINCT i.rel_path, i.path, r.input_dir
        FROM openclip_semantic_results res
        JOIN images i ON i.image_id = res.image_id
        JOIN runs r ON r.run_id = res.source_run_id
        WHERE res.semantic_run_id = ? AND res.image_rel_path = ?
        LIMIT 1
        """,
        (semantic_run_id, image_rel_path),
    ).fetchone()


def render_overlay(image_path: Path, boxes: list[OverlayBox], output_path: Path) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for box in boxes:
        color = COLORS.get(box.decision_type, (255, 255, 255))
        width = 4 if box.decision_type in {"suspect_composite_box", "suspect_broad_box", "keep_long_crack_parent"} else 2
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=width)
        label = f"{box.result_id} {box.label} {box.decision_type} q={box.box_quality_score:.2f}"
        text_bbox = draw.textbbox((box.x1, box.y1), label, font=font)
        pad = 3
        bg = (max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40))
        draw.rectangle((text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad), fill=bg)
        draw.text((box.x1, box.y1), label, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render bbox cleanup overlay images for manual review.")
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--run-id", default="resemi_full_bbox_v1")
    parser.add_argument("--image-root", default=str(default_image_root()))
    parser.add_argument("--output-dir", default="", help="Default: infer_results/semi-labeling/resemi/overlays/<run_id>_review30")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--include-decisions", default="", help="Comma-separated decision types to draw. Empty = all.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    run_id = str(args.run_id)
    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir or "").strip() else default_output_dir(run_id)
    image_root = Path(args.image_root).expanduser().resolve() if str(args.image_root or "").strip() else None
    include_decisions = {item.strip() for item in str(args.include_decisions or "").split(",") if item.strip()} or None

    resemi_conn = connect_readonly(db_path)
    try:
        run = load_run(resemi_conn, run_id)
        source_db = Path(str(run["source_db_path"])).expanduser().resolve()
        semantic_run_id = str(run["source_semantic_run_id"])
        image_rel_paths = select_images(resemi_conn, run_id=run_id, limit=int(args.limit))
    finally:
        resemi_conn.close()

    source_conn = connect_readonly(source_db)
    resemi_conn = connect_readonly(db_path)
    index_rows = []
    try:
        for idx, image_rel_path in enumerate(image_rel_paths, start=1):
            source_row = load_source_image_row(source_conn, semantic_run_id=semantic_run_id, image_rel_path=image_rel_path)
            if source_row is None:
                continue
            image_path = resolve_image_path(source_row, image_root)
            if not image_path.is_file():
                continue
            boxes = load_boxes(resemi_conn, run_id=run_id, image_rel_path=image_rel_path, include_decisions=include_decisions)
            if not boxes:
                continue
            safe_name = image_rel_path.replace("/", "__").replace("\\", "__")
            output_path = output_dir / f"{idx:02d}_{Path(safe_name).stem}.png"
            render_overlay(image_path, boxes, output_path)
            counts = {decision: sum(1 for box in boxes if box.decision_type == decision) for decision in sorted(COLORS)}
            index_rows.append(
                {
                    "overlay_path": str(output_path),
                    "image_rel_path": image_rel_path,
                    "source_image_path": str(image_path),
                    "box_count": len(boxes),
                    **counts,
                }
            )
    finally:
        source_conn.close()
        resemi_conn.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.csv"
    fieldnames = list(index_rows[0].keys()) if index_rows else ["overlay_path", "image_rel_path", "source_image_path", "box_count"]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"output_dir={output_dir}")
    print(f"index_csv={index_path}")
    print(f"rendered_count={len(index_rows)}")
    if len(index_rows) < int(args.limit):
        print(f"warning=only_rendered_{len(index_rows)}_of_{int(args.limit)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
