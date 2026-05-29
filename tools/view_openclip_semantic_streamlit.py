#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "crack": (255, 99, 71),
    "mold": (50, 205, 50),
    "spall": (30, 144, 255),
}
FALLBACK_COLORS = [
    (255, 165, 0),
    (186, 85, 211),
    (0, 206, 209),
    (255, 215, 0),
]


@dataclass(frozen=True)
class SemanticRun:
    semantic_run_id: str
    created_at_utc: str
    source_run_id: str
    model_name: str
    pretrained: str
    device: str
    result_count: int


@dataclass(frozen=True)
class ImageSummary:
    image_rel_path: str
    image_path: str
    source_input_dir: str
    total_boxes: int
    max_probability_pct: float
    crack_count: int
    mold_count: int
    spall_count: int


@dataclass(frozen=True)
class BoxRecord:
    result_id: int
    source_detection_id: int
    image_rel_path: str
    image_path: str
    source_input_dir: str
    predicted_label: str
    predicted_probability_pct: float
    detector_label: str
    detector_score: float
    x1: float
    y1: float
    x2: float
    y2: float
    top_prompt: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_db_path() -> Path:
    return repo_root().parent / "infer_results" / "semi-labeling" / "step2_sematic" / "damage_scan.sqlite3"


def default_image_root() -> Path:
    return repo_root().parent / "data" / "HinhAnh"


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def db_mtime(db_path: str | Path) -> float:
    path = Path(db_path).expanduser().resolve()
    return path.stat().st_mtime if path.exists() else 0.0


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def validate_db(db_path: str | Path) -> None:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DB not found: {path}")
    conn = connect_readonly(path)
    try:
        required = ["runs", "images", "detections", "openclip_semantic_runs", "openclip_semantic_results"]
        missing = [name for name in required if not _table_exists(conn, name)]
        if missing:
            raise ValueError(f"Missing tables in DB: {', '.join(missing)}")
    finally:
        conn.close()


def list_semantic_runs(db_path: str | Path, mtime: float) -> list[SemanticRun]:
    del mtime
    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            """
            SELECT run.semantic_run_id, run.created_at_utc, run.source_run_id,
                   run.model_name, run.pretrained, run.device,
                   COUNT(res.result_id) AS result_count
            FROM openclip_semantic_runs run
            LEFT JOIN openclip_semantic_results res
              ON res.semantic_run_id = run.semantic_run_id
            GROUP BY run.semantic_run_id
            ORDER BY run.created_at_utc DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [SemanticRun(**dict(row)) for row in rows]


def list_labels(db_path: str | Path, semantic_run_id: str, mtime: float) -> list[str]:
    del mtime
    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT predicted_label
            FROM openclip_semantic_results
            WHERE semantic_run_id = ? AND status = 'ok' AND predicted_label != ''
            ORDER BY predicted_label
            """,
            (semantic_run_id,),
        ).fetchall()
    finally:
        conn.close()
    return [str(row["predicted_label"]) for row in rows]


def _label_filter_sql(labels: list[str]) -> tuple[str, list[Any]]:
    if not labels:
        return "", []
    placeholders = ", ".join("?" for _ in labels)
    return f" AND res.predicted_label IN ({placeholders})", list(labels)


def list_image_summaries(
    db_path: str | Path,
    semantic_run_id: str,
    labels: tuple[str, ...],
    min_probability_pct: float,
    mtime: float,
) -> list[ImageSummary]:
    del mtime
    label_sql, label_params = _label_filter_sql(list(labels))
    params: list[Any] = [semantic_run_id, float(min_probability_pct), *label_params]
    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT res.image_rel_path, res.image_path, src.input_dir AS source_input_dir,
                   COUNT(*) AS total_boxes,
                   MAX(res.predicted_probability_pct) AS max_probability_pct,
                   SUM(CASE WHEN res.predicted_label = 'crack' THEN 1 ELSE 0 END) AS crack_count,
                   SUM(CASE WHEN res.predicted_label = 'mold' THEN 1 ELSE 0 END) AS mold_count,
                   SUM(CASE WHEN res.predicted_label = 'spall' THEN 1 ELSE 0 END) AS spall_count
            FROM openclip_semantic_results res
            JOIN runs src ON src.run_id = res.source_run_id
            WHERE res.semantic_run_id = ?
              AND res.status = 'ok'
              AND res.predicted_probability_pct >= ?
              {label_sql}
            GROUP BY res.image_rel_path, res.image_path, src.input_dir
            ORDER BY total_boxes DESC, res.image_rel_path
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    return [ImageSummary(**dict(row)) for row in rows]


def list_boxes_for_image(
    db_path: str | Path,
    semantic_run_id: str,
    image_rel_path: str,
    labels: tuple[str, ...],
    min_probability_pct: float,
    mtime: float,
) -> list[BoxRecord]:
    del mtime
    label_sql, label_params = _label_filter_sql(list(labels))
    params: list[Any] = [semantic_run_id, image_rel_path, float(min_probability_pct), *label_params]
    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT res.result_id, res.source_detection_id, res.image_rel_path, res.image_path,
                   src.input_dir AS source_input_dir, res.predicted_label,
                   res.predicted_probability_pct, res.detector_label, res.detector_score,
                   res.x1, res.y1, res.x2, res.y2, res.top_prompt
            FROM openclip_semantic_results res
            JOIN runs src ON src.run_id = res.source_run_id
            WHERE res.semantic_run_id = ?
              AND res.image_rel_path = ?
              AND res.status = 'ok'
              AND res.predicted_probability_pct >= ?
              {label_sql}
            ORDER BY res.predicted_probability_pct DESC, res.result_id
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    return [BoxRecord(**dict(row)) for row in rows]


def resolve_image_path(summary: ImageSummary | BoxRecord, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(summary.image_rel_path or "").strip()
    stored_path = str(summary.image_path or "").strip()
    source_input_dir = Path(str(summary.source_input_dir or "")).expanduser()

    if image_root is not None:
        root = image_root.expanduser().resolve()
        candidates.append(root / rel_path)
        if stored_path:
            candidates.append(root / Path(stored_path).name)

    if stored_path:
        stored = Path(stored_path).expanduser()
        if stored.is_absolute():
            candidates.append(stored)
        else:
            candidates.append(source_input_dir / stored_path)

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

    if image_root is not None:
        return (image_root.expanduser().resolve() / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()


def color_for_label(label: str) -> tuple[int, int, int]:
    key = str(label or "").strip().lower()
    if key in LABEL_COLORS:
        return LABEL_COLORS[key]
    idx = abs(hash(key)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def draw_overlay(
    image_path: Path,
    boxes: list[BoxRecord],
    *,
    line_width: int,
    show_detector_score: bool,
    draw_fill: bool,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box in boxes:
        color = color_for_label(box.predicted_label)
        outline = (*color, 255)
        fill = (*color, 35) if draw_fill else None
        coords = (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
        if fill is not None:
            draw.rectangle(coords, fill=fill)
        draw.rectangle(coords, outline=outline, width=max(1, int(line_width)))

        label = f"{box.predicted_label} {box.predicted_probability_pct:.1f}%"
        if show_detector_score:
            label = f"{label} | det {box.detector_score:.2f}"
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
            draw.text((x + 2, y + 2), label, fill=outline)
    return image


def image_option(summary: ImageSummary) -> str:
    counts = []
    if summary.crack_count:
        counts.append(f"crack={summary.crack_count}")
    if summary.mold_count:
        counts.append(f"mold={summary.mold_count}")
    if summary.spall_count:
        counts.append(f"spall={summary.spall_count}")
    suffix = ", ".join(counts) if counts else f"boxes={summary.total_boxes}"
    return f"{summary.image_rel_path} ({summary.total_boxes}; {suffix})"


def boxes_as_rows(boxes: list[BoxRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for box in boxes:
        rows.append(
            {
                "result_id": box.result_id,
                "detection_id": box.source_detection_id,
                "label": box.predicted_label,
                "clip_pct": round(float(box.predicted_probability_pct), 2),
                "detector_label": box.detector_label,
                "detector_score": round(float(box.detector_score), 4),
                "x1": round(float(box.x1), 1),
                "y1": round(float(box.y1), 1),
                "x2": round(float(box.x2), 1),
                "y2": round(float(box.y2), 1),
                "top_prompt": box.top_prompt,
            }
        )
    return rows


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="OpenCLIP Semantic Overlay", layout="wide")
    st.title("OpenCLIP Semantic Overlay Viewer")

    db_input = st.sidebar.text_input("SQLite DB", value=str(default_db_path()))
    image_root_input = st.sidebar.text_input("Image root", value=str(default_image_root()))
    db_path = Path(db_input).expanduser().resolve()
    image_root = Path(image_root_input).expanduser().resolve() if image_root_input.strip() else None

    try:
        validate_db(db_path)
        mtime = db_mtime(db_path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    runs = list_semantic_runs(db_path, mtime)
    if not runs:
        st.warning("No OpenCLIP semantic runs found.")
        st.stop()

    run_labels = [
        f"{run.created_at_utc} | {run.semantic_run_id[:8]} | {run.model_name} | {run.device} | {run.result_count} results"
        for run in runs
    ]
    selected_run_idx = st.sidebar.selectbox("Semantic run", range(len(runs)), format_func=lambda idx: run_labels[idx])
    selected_run = runs[int(selected_run_idx)]

    labels = list_labels(db_path, selected_run.semantic_run_id, mtime)
    selected_labels = st.sidebar.multiselect("Predicted labels", options=labels, default=labels)
    min_probability_pct = st.sidebar.slider("Min OpenCLIP confidence (%)", 0.0, 100.0, 0.0, 1.0)
    line_width = st.sidebar.slider("Box line width", 1, 8, 3)
    preview_width = st.sidebar.slider("Preview width", 400, 1800, 1100, 50)
    show_detector_score = st.sidebar.checkbox("Show detector score", value=False)
    draw_fill = st.sidebar.checkbox("Tint box fill", value=True)

    summaries = list_image_summaries(
        db_path,
        selected_run.semantic_run_id,
        tuple(selected_labels),
        float(min_probability_pct),
        mtime,
    )
    if not summaries:
        st.warning("No images matched the current filters.")
        st.stop()

    total_boxes = sum(item.total_boxes for item in summaries)
    crack_total = sum(item.crack_count for item in summaries)
    mold_total = sum(item.mold_count for item in summaries)
    spall_total = sum(item.spall_count for item in summaries)
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Images", len(summaries))
    col_b.metric("Boxes", total_boxes)
    col_c.metric("Crack", crack_total)
    col_d.metric("Mold", mold_total)
    col_e.metric("Spall", spall_total)

    image_idx = st.sidebar.selectbox("Image", range(len(summaries)), format_func=lambda idx: image_option(summaries[idx]))
    summary = summaries[int(image_idx)]
    image_path = resolve_image_path(summary, image_root)

    boxes = list_boxes_for_image(
        db_path,
        selected_run.semantic_run_id,
        summary.image_rel_path,
        tuple(selected_labels),
        float(min_probability_pct),
        mtime,
    )

    st.caption(f"semantic_run_id={selected_run.semantic_run_id}")
    st.subheader(summary.image_rel_path)
    if not image_path.is_file():
        st.error(f"Image not found: {image_path}")
        st.stop()

    overlay = draw_overlay(
        image_path,
        boxes,
        line_width=int(line_width),
        show_detector_score=bool(show_detector_score),
        draw_fill=bool(draw_fill),
    )
    st.image(overlay, caption=str(image_path), width=int(preview_width))

    st.subheader(f"Boxes ({len(boxes)})")
    st.dataframe(boxes_as_rows(boxes), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
