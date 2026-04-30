#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

LABEL_COLORS = {
    "crack": (255, 99, 71),
    "mold": (50, 205, 50),
    "spall": (30, 144, 255),
}


@dataclass(frozen=True)
class FilterRun:
    filter_run_id: str
    created_at_utc: str
    source_db_path: str
    source_semantic_run_id: str
    model_name: str
    device: str
    total_boxes: int
    kept_boxes: int
    dropped_boxes: int
    suspect_boxes: int


@dataclass(frozen=True)
class ImageSummary:
    image_rel_path: str
    total_boxes: int
    kept_boxes: int
    dropped_boxes: int
    suspect_boxes: int


@dataclass(frozen=True)
class BoxRow:
    result_id: int
    image_rel_path: str
    image_path: str
    source_input_dir: str
    predicted_label: str
    predicted_probability_pct: float
    detector_score: float
    x1: float
    y1: float
    x2: float
    y2: float
    keep: int
    drop_reason: str
    suspect_reason: str
    similarity_to_kept: float
    max_similarity: float
    max_iou: float
    max_containment: float
    area_ratio: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source_db() -> Path:
    return repo_root().parent / "infer_results" / "semi-labeling" / "2_sematic" / "damage_scan.sqlite3"


def default_output_dir() -> Path:
    source_parent = default_source_db().parent
    preferred = source_parent / "step3_spatial_filter"
    if (preferred / "filtered.sqlite3").is_file():
        return preferred
    roi_dryrun = source_parent / "step3_spatial_filter_roi5_roi6"
    if (roi_dryrun / "filtered.sqlite3").is_file():
        return roi_dryrun
    other_dryrun = source_parent / "step3_spatial_filter_dryrun_other"
    if (other_dryrun / "filtered.sqlite3").is_file():
        return other_dryrun
    strict_dryrun = source_parent / "step3_spatial_filter_dryrun_strict"
    if (strict_dryrun / "filtered.sqlite3").is_file():
        return strict_dryrun
    dryrun = source_parent / "step3_spatial_filter_dryrun"
    if (dryrun / "filtered.sqlite3").is_file():
        return dryrun
    return preferred


def default_image_root() -> Path:
    return repo_root().parent / "HinhAnh"


def connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.expanduser().resolve()}?mode=ro", uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def list_filter_runs(filtered_db: Path) -> list[FilterRun]:
    conn = connect_ro(filtered_db)
    try:
        rows = conn.execute(
            """
            SELECT filter_run_id, created_at_utc, source_db_path, source_semantic_run_id,
                   model_name, device, total_boxes, kept_boxes, dropped_boxes, suspect_boxes
            FROM filter_runs
            ORDER BY created_at_utc DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [FilterRun(**dict(row)) for row in rows]


def attach_source(conn: sqlite3.Connection, source_db: Path) -> None:
    conn.execute("ATTACH DATABASE ? AS src", (str(source_db.expanduser().resolve()),))


def list_images(source_db: Path, filtered_db: Path, filter_run_id: str, mode: str) -> list[ImageSummary]:
    conn = connect_ro(filtered_db)
    attach_source(conn, source_db)
    where = "fr.filter_run_id = ?"
    params: list[Any] = [filter_run_id]
    if mode == "Kept only":
        where += " AND fr.keep = 1"
    elif mode == "Dropped only":
        where += " AND fr.keep = 0"
    elif mode == "Suspect only":
        where += " AND fr.suspect_reason != ''"
    try:
        rows = conn.execute(
            f"""
            SELECT fr.image_rel_path,
                   COUNT(*) AS total_boxes,
                   SUM(CASE WHEN fr.keep = 1 THEN 1 ELSE 0 END) AS kept_boxes,
                   SUM(CASE WHEN fr.keep = 0 THEN 1 ELSE 0 END) AS dropped_boxes,
                   SUM(CASE WHEN fr.suspect_reason != '' THEN 1 ELSE 0 END) AS suspect_boxes
            FROM filter_results fr
            JOIN src.openclip_semantic_results res ON res.result_id = fr.result_id
            WHERE {where}
            GROUP BY fr.image_rel_path
            ORDER BY total_boxes DESC, fr.image_rel_path
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    return [ImageSummary(**dict(row)) for row in rows]


def list_boxes(source_db: Path, filtered_db: Path, filter_run_id: str, image_rel_path: str, mode: str) -> list[BoxRow]:
    conn = connect_ro(filtered_db)
    attach_source(conn, source_db)
    where = "fr.filter_run_id = ? AND fr.image_rel_path = ?"
    params: list[Any] = [filter_run_id, image_rel_path]
    if mode == "Kept only":
        where += " AND fr.keep = 1"
    elif mode == "Dropped only":
        where += " AND fr.keep = 0"
    elif mode == "Suspect only":
        where += " AND fr.suspect_reason != ''"
    try:
        rows = conn.execute(
            f"""
            SELECT res.result_id, res.image_rel_path, res.image_path, src_run.input_dir AS source_input_dir,
                   res.predicted_label, res.predicted_probability_pct, res.detector_score,
                   res.x1, res.y1, res.x2, res.y2,
                   fr.keep, fr.drop_reason, fr.suspect_reason, fr.similarity_to_kept,
                   fr.max_similarity, fr.max_iou, fr.max_containment, fr.area_ratio
            FROM filter_results fr
            JOIN src.openclip_semantic_results res ON res.result_id = fr.result_id
            JOIN src.runs src_run ON src_run.run_id = res.source_run_id
            WHERE {where}
            ORDER BY fr.keep DESC, res.predicted_label, fr.max_similarity DESC, res.result_id
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    return [BoxRow(**dict(row)) for row in rows]


def list_suspect_pairs(suspect_db: Path, filter_run_id: str, image_rel_path: str) -> list[dict[str, Any]]:
    if not suspect_db.is_file():
        return []
    conn = connect_ro(suspect_db)
    try:
        rows = conn.execute(
            """
            SELECT reason, result_id_a, result_id_b, predicted_label_a, predicted_label_b,
                   ROUND(iou, 4) AS iou, ROUND(containment, 4) AS containment,
                   ROUND(similarity, 4) AS similarity, keep_a, keep_b
            FROM suspect_pairs
            WHERE filter_run_id = ? AND image_rel_path = ?
            ORDER BY reason, similarity DESC
            """,
            (filter_run_id, image_rel_path),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def resolve_image_path(box: BoxRow, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(box.image_rel_path or "").strip()
    stored = str(box.image_path or "").strip()
    source_input_dir = Path(str(box.source_input_dir or "")).expanduser()
    if image_root is not None:
        root = image_root.expanduser().resolve()
        candidates.append(root / rel_path)
        if stored:
            candidates.append(root / Path(stored).name)
    if stored:
        stored_path = Path(stored).expanduser()
        candidates.append(stored_path if stored_path.is_absolute() else source_input_dir / stored)
    candidates.append(source_input_dir / rel_path)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    if image_root is not None:
        return (image_root.expanduser().resolve() / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()


def color_for_box(box: BoxRow) -> tuple[int, int, int]:
    if not int(box.keep):
        return (160, 160, 160)
    return LABEL_COLORS.get(str(box.predicted_label).lower(), (255, 165, 0))


def draw_overlay(image_path: Path, boxes: list[BoxRow], *, line_width: int, fill: bool) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for box in boxes:
        color = color_for_box(box)
        alpha = 255 if int(box.keep) else 180
        coords = (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
        if fill:
            draw.rectangle(coords, fill=(*color, 35 if int(box.keep) else 20))
        draw.rectangle(coords, outline=(*color, alpha), width=max(1, int(line_width)))
        status = "keep" if int(box.keep) else "drop"
        label = f"{status} {box.predicted_label} {box.predicted_probability_pct:.1f}% sim={box.max_similarity:.2f}"
        x = max(0.0, float(box.x1))
        y = max(0.0, float(box.y1))
        if font is not None:
            left, top, right, bottom = draw.textbbox((x, y), label, font=font)
            text_w = right - left
            text_h = bottom - top
            bg_y = max(0.0, y - text_h - 5)
            draw.rectangle((x, bg_y, x + text_w + 6, bg_y + text_h + 4), fill=(*color, 220))
            draw.text((x + 3, bg_y + 2), label, fill=(0, 0, 0, 255), font=font)
        else:
            draw.text((x + 2, y + 2), label, fill=(*color, alpha))
    return image


def rows_for_table(boxes: list[BoxRow]) -> list[dict[str, Any]]:
    return [
        {
            "result_id": box.result_id,
            "keep": int(box.keep),
            "label": box.predicted_label,
            "clip_pct": round(box.predicted_probability_pct, 2),
            "det_score": round(box.detector_score, 4),
            "sim_to_kept": round(box.similarity_to_kept, 4),
            "max_sim": round(box.max_similarity, 4),
            "max_iou": round(box.max_iou, 4),
            "max_containment": round(box.max_containment, 4),
            "area_ratio": round(box.area_ratio, 4),
            "drop_reason": box.drop_reason,
            "suspect_reason": box.suspect_reason,
        }
        for box in boxes
    ]


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Spatial Filter Viewer", layout="wide")
    st.title("Spatial Box Filter Viewer")

    source_db = Path(st.sidebar.text_input("Source step2 DB", value=str(default_source_db()))).expanduser().resolve()
    output_dir = Path(st.sidebar.text_input("Step3 output dir", value=str(default_output_dir()))).expanduser().resolve()
    filtered_db = output_dir / "filtered.sqlite3"
    suspect_db = output_dir / "suspect.sqlite3"
    image_root_input = st.sidebar.text_input("Image root", value=str(default_image_root()))
    image_root = Path(image_root_input).expanduser().resolve() if image_root_input.strip() else None

    if not filtered_db.is_file():
        fallback_dirs = [
            source_db.parent / "step3_spatial_filter",
            source_db.parent / "step3_spatial_filter_roi5_roi6",
            source_db.parent / "step3_spatial_filter_dryrun_other",
            source_db.parent / "step3_spatial_filter_dryrun_strict",
            source_db.parent / "step3_spatial_filter_dryrun",
        ]
        fallback_dir = next((item for item in fallback_dirs if (item / "filtered.sqlite3").is_file()), None)
        if fallback_dir is not None:
            output_dir = fallback_dir
            filtered_db = fallback_dir / "filtered.sqlite3"
            suspect_db = output_dir / "suspect.sqlite3"
            st.sidebar.warning(f"Using dry-run output: {output_dir}")
        else:
            st.error(f"filtered.sqlite3 not found: {filtered_db}")
            st.info("Run step 3 first, or set Step3 output dir to a folder that contains filtered.sqlite3.")
            st.stop()
    if not source_db.is_file():
        st.error(f"Source DB not found: {source_db}")
        st.stop()

    runs = list_filter_runs(filtered_db)
    if not runs:
        st.warning("No filter runs found.")
        st.stop()
    run_labels = [f"{run.created_at_utc} | {run.filter_run_id[:8]} | kept {run.kept_boxes}/{run.total_boxes} | {run.model_name}" for run in runs]
    run_idx = st.sidebar.selectbox("Filter run", range(len(runs)), format_func=lambda idx: run_labels[idx])
    run = runs[int(run_idx)]
    mode = st.sidebar.radio("View mode", ["Kept only", "All filtered", "Dropped only", "Suspect only"])
    line_width = st.sidebar.slider("Box line width", 1, 8, 3)
    image_display = st.sidebar.radio("Image display", ["Container width", "Custom width", "Original size"], index=0)
    preview_width = st.sidebar.slider("Custom preview width", 400, 4000, 1600, 50)
    tint_fill = st.sidebar.checkbox("Tint box fill", value=True)

    images = list_images(source_db, filtered_db, run.filter_run_id, mode)
    if not images:
        st.warning("No images matched this mode.")
        st.stop()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total", run.total_boxes)
    col_b.metric("Kept", run.kept_boxes)
    col_c.metric("Dropped", run.dropped_boxes)
    col_d.metric("Suspect", run.suspect_boxes)

    image_idx = st.sidebar.selectbox(
        "Image",
        range(len(images)),
        format_func=lambda idx: f"{images[idx].image_rel_path} (total={images[idx].total_boxes}, kept={images[idx].kept_boxes}, drop={images[idx].dropped_boxes}, suspect={images[idx].suspect_boxes})",
    )
    image_summary = images[int(image_idx)]
    boxes = list_boxes(source_db, filtered_db, run.filter_run_id, image_summary.image_rel_path, mode)
    if not boxes:
        st.warning("No boxes for selected image.")
        st.stop()
    image_path = resolve_image_path(boxes[0], image_root)
    st.caption(f"filter_run_id={run.filter_run_id} | source_semantic_run_id={run.source_semantic_run_id}")
    st.subheader(image_summary.image_rel_path)
    if not image_path.is_file():
        st.error(f"Image not found: {image_path}")
        st.stop()

    overlay = draw_overlay(image_path, boxes, line_width=int(line_width), fill=bool(tint_fill))
    if image_display == "Container width":
        st.image(overlay, caption=str(image_path), use_container_width=True)
    elif image_display == "Original size":
        st.image(overlay, caption=str(image_path))
    else:
        st.image(overlay, caption=str(image_path), width=int(preview_width))
    st.subheader(f"Boxes ({len(boxes)})")
    st.dataframe(rows_for_table(boxes), use_container_width=True, hide_index=True)

    pairs = list_suspect_pairs(suspect_db, run.filter_run_id, image_summary.image_rel_path)
    if pairs:
        st.subheader(f"Suspect pairs ({len(pairs)})")
        st.dataframe(pairs, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
