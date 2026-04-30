#!/usr/bin/env python3
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class GroupRun:
    grouping_run_id: str
    created_at_utc: str
    source_db_path: str
    filtered_db_path: str
    source_filter_run_id: str
    model_name: str
    device: str
    total_boxes: int
    total_clusters: int
    outlier_boxes: int
    label_suspect_boxes: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source_db() -> Path:
    return repo_root().parent / "infer_results" / "semi-labeling" / "2_sematic" / "damage_scan.sqlite3"


def default_output_db() -> Path:
    return default_source_db().parent / "step4_feature_grouping" / "feature_groups.sqlite3"


def default_image_root() -> Path:
    return repo_root().parent / "HinhAnh"


def connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.expanduser().resolve()}?mode=ro", uri=True, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def attach_source(conn: sqlite3.Connection, source_db: Path) -> None:
    conn.execute("ATTACH DATABASE ? AS src", (str(source_db.expanduser().resolve()),))


def list_runs(output_db: Path) -> list[GroupRun]:
    conn = connect_ro(output_db)
    try:
        rows = conn.execute(
            """
            SELECT grouping_run_id, created_at_utc, source_db_path, filtered_db_path,
                   source_filter_run_id, model_name, device, total_boxes, total_clusters,
                   outlier_boxes, label_suspect_boxes
            FROM feature_group_runs
            ORDER BY created_at_utc DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [GroupRun(**dict(row)) for row in rows]


def list_clusters(output_db: Path, grouping_run_id: str, label_scope: str, mode: str) -> list[dict[str, Any]]:
    clauses = ["grouping_run_id = ?"]
    params: list[Any] = [grouping_run_id]
    if label_scope != "All":
        clauses.append("predicted_label_scope = ?")
        params.append(label_scope)
    if mode == "Non-outlier clusters":
        clauses.append("outlier_count = 0")
    elif mode == "Outliers only":
        clauses.append("outlier_count > 0")
    elif mode == "Label suspect only":
        clauses.append("purity < 1.0")
    conn = connect_ro(output_db)
    try:
        rows = conn.execute(
            f"""
            SELECT cluster_key, predicted_label_scope, cluster_id, cluster_size,
                   major_label, ROUND(purity, 4) AS purity, crack_count, mold_count,
                   spall_count, outlier_count, representative_nearest_result_id,
                   representative_farthest_result_id, representative_low_confidence_result_id,
                   representative_mismatch_result_id
            FROM feature_group_clusters
            WHERE {' AND '.join(clauses)}
            ORDER BY outlier_count ASC, cluster_size DESC, purity ASC, cluster_key
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def list_assignments(output_db: Path, source_db: Path, grouping_run_id: str, cluster_key: str) -> list[dict[str, Any]]:
    conn = connect_ro(output_db)
    attach_source(conn, source_db)
    try:
        rows = conn.execute(
            """
            SELECT a.result_id, a.source_detection_id, a.image_rel_path, res.image_path,
                   src_run.input_dir AS source_input_dir, a.predicted_label,
                   a.predicted_probability_pct, a.detector_score, a.cluster_id,
                   a.cluster_key, a.is_outlier, a.distance_to_center, a.suggested_label,
                   a.label_suspect, a.cluster_purity, a.cluster_size,
                   res.x1, res.y1, res.x2, res.y2
            FROM feature_group_assignments a
            JOIN src.openclip_semantic_results res ON res.result_id = a.result_id
            JOIN src.runs src_run ON src_run.run_id = res.source_run_id
            WHERE a.grouping_run_id = ? AND a.cluster_key = ?
            ORDER BY a.is_outlier ASC, a.distance_to_center ASC, a.predicted_probability_pct ASC, a.result_id
            """,
            (grouping_run_id, cluster_key),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def resolve_image_path(row: dict[str, Any], image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(row.get("image_rel_path") or "").strip()
    stored = str(row.get("image_path") or "").strip()
    source_input_dir = Path(str(row.get("source_input_dir") or "")).expanduser()
    if image_root is not None:
        root = image_root.expanduser().resolve()
        candidates.append(root / rel_path)
        if stored:
            candidates.append(root / Path(stored).name)
    if stored:
        stored_path = Path(stored).expanduser()
        candidates.append(stored_path if stored_path.is_absolute() else source_input_dir / stored)
    if rel_path:
        candidates.append(source_input_dir / rel_path)
    if stored:
        candidates.append(source_input_dir / Path(stored).name)
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


def crop_row(row: dict[str, Any], image_root: Path | None, *, padding_ratio: float) -> Image.Image:
    image_path = resolve_image_path(row, image_root)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    pad_x = max(0.0, float(row["x2"]) - float(row["x1"])) * float(padding_ratio)
    pad_y = max(0.0, float(row["y2"]) - float(row["y1"])) * float(padding_ratio)
    x1 = max(0, int(math.floor(float(row["x1"]) - pad_x)))
    y1 = max(0, int(math.floor(float(row["y1"]) - pad_y)))
    x2 = min(width, int(math.ceil(float(row["x2"]) + pad_x)))
    y2 = min(height, int(math.ceil(float(row["y2"]) + pad_y)))
    return image.crop((x1, y1, x2, y2))


def rows_for_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "result_id": row["result_id"],
            "image": row["image_rel_path"],
            "label": row["predicted_label"],
            "clip_pct": round(float(row["predicted_probability_pct"]), 2),
            "det_score": round(float(row["detector_score"]), 4),
            "distance": round(float(row["distance_to_center"]), 4),
            "outlier": int(row["is_outlier"]),
            "purity": round(float(row["cluster_purity"]), 4),
            "suggested_label": row["suggested_label"],
        }
        for row in rows
    ]


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Feature Group Viewer", layout="wide")
    st.title("DINOv2 Feature Group Viewer")
    st.caption("Review-only clustering output. This viewer does not modify labels or source datasets.")

    output_db = Path(st.sidebar.text_input("Step4 output DB", value=str(default_output_db()))).expanduser().resolve()
    image_root_input = st.sidebar.text_input("Image root", value=str(default_image_root()))
    image_root = Path(image_root_input).expanduser().resolve() if image_root_input.strip() else None
    padding_ratio = st.sidebar.slider("Crop padding", 0.0, 0.30, 0.05, 0.01)
    thumb_width = st.sidebar.slider("Representative crop width", 120, 600, 260, 20)

    if not output_db.is_file():
        st.error(f"feature_groups.sqlite3 not found: {output_db}")
        st.info("Run step4_feature_grouping/run_feature_grouping.py first.")
        st.stop()

    runs = list_runs(output_db)
    if not runs:
        st.warning("No feature grouping runs found.")
        st.stop()
    run_labels = [f"{run.created_at_utc} | {run.grouping_run_id[:8]} | clusters {run.total_clusters} | outliers {run.outlier_boxes} | {run.model_name}" for run in runs]
    run_idx = st.sidebar.selectbox("Grouping run", range(len(runs)), format_func=lambda idx: run_labels[idx])
    run = runs[int(run_idx)]
    source_db = Path(run.source_db_path).expanduser().resolve()
    if not source_db.is_file():
        st.error(f"Source DB not found: {source_db}")
        st.stop()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Boxes", run.total_boxes)
    col_b.metric("Clusters", run.total_clusters)
    col_c.metric("Outliers", run.outlier_boxes)
    col_d.metric("Label suspect", run.label_suspect_boxes)

    label_scope = st.sidebar.selectbox("Label scope", ["All", "crack", "mold", "spall", "all"])
    mode = st.sidebar.radio("View mode", ["All clusters", "Non-outlier clusters", "Outliers only", "Label suspect only"])
    clusters = list_clusters(output_db, run.grouping_run_id, label_scope, mode)
    if not clusters:
        st.warning("No clusters matched this filter.")
        st.stop()

    cluster_idx = st.sidebar.selectbox(
        "Cluster",
        range(len(clusters)),
        format_func=lambda idx: f"{clusters[idx]['cluster_key']} size={clusters[idx]['cluster_size']} purity={clusters[idx]['purity']}",
    )
    cluster = clusters[int(cluster_idx)]
    st.subheader(f"Cluster {cluster['cluster_key']}")
    st.dataframe([cluster], use_container_width=True, hide_index=True)

    rows = list_assignments(output_db, source_db, run.grouping_run_id, str(cluster["cluster_key"]))
    reps = [
        ("nearest", cluster.get("representative_nearest_result_id")),
        ("farthest", cluster.get("representative_farthest_result_id")),
        ("low confidence", cluster.get("representative_low_confidence_result_id")),
        ("label mismatch", cluster.get("representative_mismatch_result_id")),
    ]
    by_id = {int(row["result_id"]): row for row in rows}
    visible_reps = [(name, by_id.get(int(result_id))) for name, result_id in reps if result_id is not None and by_id.get(int(result_id)) is not None]
    if visible_reps:
        st.subheader("Representatives")
        columns = st.columns(len(visible_reps))
        for col, (name, row) in zip(columns, visible_reps):
            try:
                crop = crop_row(row, image_root, padding_ratio=float(padding_ratio))
                col.image(crop, caption=f"{name}: {row['result_id']} {row['predicted_label']} {float(row['predicted_probability_pct']):.1f}%", width=int(thumb_width))
            except Exception as exc:
                col.error(f"{name}: {exc}")

    st.subheader(f"Boxes ({len(rows)})")
    st.dataframe(rows_for_table(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
