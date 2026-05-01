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


def cluster_option_label(cluster: dict[str, Any]) -> str:
    return (
        f"{cluster['cluster_key']} | size={cluster['cluster_size']} | "
        f"major={cluster['major_label']} | purity={cluster['purity']}"
    )


def card_html(cluster: dict[str, Any]) -> str:
    return f"""
    <div style="border: 1px solid #dddddd; border-radius: 16px; padding: 14px; min-height: 142px; background: #ffffff;">
      <div style="font-weight: 800; font-size: 16px; margin-bottom: 8px;">{cluster['cluster_key']}</div>
      <div style="font-size: 14px; color: #222;">Images: <b>{cluster['cluster_size']}</b></div>
      <div style="font-size: 14px; color: #222;">Major: <b>{cluster['major_label']}</b></div>
      <div style="font-size: 14px; color: #222;">Purity: <b>{cluster['purity']}</b></div>
      <div style="font-size: 12px; color: #666; margin-top: 8px;">crack {cluster['crack_count']} | mold {cluster['mold_count']} | spall {cluster['spall_count']}</div>
    </div>
    """


def page_count(total_items: int, page_size: int) -> int:
    return max(1, math.ceil(max(0, int(total_items)) / max(1, int(page_size))))


def page_slice(items: list[Any], page: int, page_size: int) -> list[Any]:
    safe_page_size = max(1, int(page_size))
    start = max(0, int(page) - 1) * safe_page_size
    return items[start : start + safe_page_size]


def page_picker(st: Any, *, label: str, total_items: int, page_size: int, key: str) -> int:
    total_pages = page_count(total_items, page_size)
    if total_pages <= 1:
        return 1
    return int(st.number_input(label, min_value=1, max_value=total_pages, value=1, step=1, key=key))


def render_crop_grid(st: Any, rows: list[dict[str, Any]], image_root: Path | None, *, padding_ratio: float, thumb_width: int) -> None:
    visible_rows = rows
    if not visible_rows:
        st.info("No boxes in this cluster.")
        return
    cols_per_row = 4
    for start in range(0, len(visible_rows), cols_per_row):
        columns = st.columns(cols_per_row)
        for col, row in zip(columns, visible_rows[start : start + cols_per_row]):
            caption = (
                f"id={row['result_id']} | {row['predicted_label']} | "
                f"conf={float(row['predicted_probability_pct']):.1f}% | dist={float(row['distance_to_center']):.3f}"
            )
            try:
                crop = crop_row(row, image_root, padding_ratio=padding_ratio)
                col.image(crop, caption=caption, width=int(thumb_width))
            except Exception as exc:
                col.error(f"{row['result_id']}: {exc}")


def render_group_list_page(st: Any, *, output_db: Path, run: GroupRun, mode: str, cards_per_page: int) -> None:
    st.header("Danh Sách Các Nhóm")
    tabs = st.tabs(["crack", "mold", "spall"])
    for tab, label_scope in zip(tabs, ["crack", "mold", "spall"]):
        with tab:
            clusters = list_clusters(output_db, run.grouping_run_id, label_scope, mode)
            if not clusters:
                st.warning(f"No {label_scope} clusters matched this filter.")
                continue
            total_boxes = sum(int(item["cluster_size"]) for item in clusters)
            c1, c2, c3 = st.columns(3)
            c1.metric("Groups", len(clusters))
            c2.metric("Images", total_boxes)
            c3.metric("Largest", max(int(item["cluster_size"]) for item in clusters))

            page = page_picker(
                st,
                label="Group page",
                total_items=len(clusters),
                page_size=int(cards_per_page),
                key=f"group_list_page_{label_scope}",
            )
            page_clusters = page_slice(clusters, page, int(cards_per_page))
            page_start = (int(page) - 1) * int(cards_per_page)
            st.caption(f"Page {page}/{page_count(len(clusters), int(cards_per_page))} | showing {page_start + 1}-{page_start + len(page_clusters)} of {len(clusters)}")

            cols_per_row = 3
            for start in range(0, len(page_clusters), cols_per_row):
                columns = st.columns(cols_per_row)
                for col, cluster in zip(columns, page_clusters[start : start + cols_per_row]):
                    col.markdown(card_html(cluster), unsafe_allow_html=True)
                    if col.button("Open group", key=f"open_group_{label_scope}_{cluster['cluster_key']}", use_container_width=True):
                        st.session_state["app_page"] = "Group detail"
                        st.session_state["detail_label_scope"] = label_scope
                        st.session_state["detail_cluster_key"] = cluster["cluster_key"]
                        st.rerun()


def render_group_detail_page(
    st: Any,
    *,
    output_db: Path,
    source_db: Path,
    run: GroupRun,
    mode: str,
    image_root: Path | None,
    padding_ratio: float,
    thumb_width: int,
    images_per_page: int,
) -> None:
    label_scope = st.session_state.get("detail_label_scope", "crack")
    cluster_key = st.session_state.get("detail_cluster_key", "")
    clusters = list_clusters(output_db, run.grouping_run_id, label_scope, mode)
    cluster = next((item for item in clusters if item["cluster_key"] == cluster_key), None)
    if cluster is None:
        st.warning("Selected group is not available with the current filters.")
        if st.button("Back to group list"):
            st.session_state["app_page"] = "Group list"
            st.rerun()
        return

    if st.button("Back to group list"):
        st.session_state["app_page"] = "Group list"
        st.rerun()
    st.header(f"Group {cluster['cluster_key']}")
    st.dataframe([cluster], use_container_width=True, hide_index=True)

    rows = list_assignments(output_db, source_db, run.grouping_run_id, str(cluster["cluster_key"]))
    image_page = page_picker(
        st,
        label="Image page",
        total_items=len(rows),
        page_size=int(images_per_page),
        key=f"detail_image_page_{cluster['cluster_key']}",
    )
    visible_rows = page_slice(rows, image_page, int(images_per_page))
    st.subheader(f"Images ({len(rows)})")
    st.caption(f"Page {image_page}/{page_count(len(rows), int(images_per_page))} | showing {len(visible_rows)} images")
    render_crop_grid(
        st,
        visible_rows,
        image_root,
        padding_ratio=float(padding_ratio),
        thumb_width=int(thumb_width),
    )
    st.subheader("Box List")
    st.dataframe(rows_for_table(rows), use_container_width=True, hide_index=True)


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
    if "app_page" not in st.session_state:
        st.session_state["app_page"] = "Group list"
    st.sidebar.caption(f"Page: {st.session_state['app_page']}")
    if st.sidebar.button("Group list", use_container_width=True):
        st.session_state["app_page"] = "Group list"
        st.rerun()
    if st.sidebar.button("Group detail", use_container_width=True):
        st.session_state["app_page"] = "Group detail"
        st.rerun()
    page = st.session_state["app_page"]
    cards_per_page = st.sidebar.slider("Group cards per page", 6, 60, 18, 3)
    padding_ratio = 0.05
    thumb_width = 220
    images_per_page = 32
    if page == "Group detail":
        st.sidebar.subheader("Image Controls")
        padding_ratio = st.sidebar.slider("Crop padding", 0.0, 0.30, 0.05, 0.01)
        thumb_width = st.sidebar.slider("Image size", 120, 700, 240, 20)
        images_per_page = st.sidebar.slider("Images per page", 8, 160, 32, 4)

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

    mode = st.sidebar.radio("View mode", ["All clusters", "Non-outlier clusters", "Outliers only", "Label suspect only"])
    if page == "Group list":
        render_group_list_page(st, output_db=output_db, run=run, mode=mode, cards_per_page=int(cards_per_page))
    else:
        render_group_detail_page(
            st,
            output_db=output_db,
            source_db=source_db,
            run=run,
            mode=mode,
            image_root=image_root,
            padding_ratio=float(padding_ratio),
            thumb_width=int(thumb_width),
            images_per_page=int(images_per_page),
        )


if __name__ == "__main__":
    main()
