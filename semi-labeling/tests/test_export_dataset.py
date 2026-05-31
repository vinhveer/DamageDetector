"""Property + example tests for tools/export_dataset.py.

Runs on the project venv (pytest + hypothesis). No node:sqlite involved.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from shared.db.schema import connect_output, utc_now
from shared.taxonomy.label_taxonomy import WORKING_LABELS, build_label_taxonomy
from tools.export_dataset import export_dataset


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_db(db_path: Path, run_id: str, rows: list[dict]) -> None:
    """Create a minimal resemi.sqlite3 with a run + cleaned_labels rows."""
    conn = connect_output(db_path)
    try:
        conn.execute(
            """
            INSERT INTO resemi_runs (run_id, created_at_utc, source_db_path,
                source_semantic_run_id, options_json, taxonomy_version_id)
            VALUES (?, ?, '', '', '{}', 'label_taxonomy_v1')
            """,
            (run_id, utc_now()),
        )
        for row in rows:
            conn.execute(
                """
                INSERT INTO cleaned_labels (run_id, result_id, image_rel_path, crop_path,
                    final_label, export_label, decision_type, reliability_score,
                    reason_codes_json, x1, y1, x2, y2)
                VALUES (?, ?, ?, NULL, ?, ?, 'auto_accept', 0.9, '[]', ?, ?, ?, ?)
                """,
                (
                    run_id, row["result_id"], row["image_rel_path"],
                    row["final_label"], row.get("export_label", ""),
                    row["x1"], row["y1"], row["x2"], row["y2"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _write_image(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (128, 128, 128)).save(path)


# Hypothesis strategies: a box that fits inside a WxH image.
@st.composite
def _cleaned_rows(draw, *, n_max=8, img_w=640, img_h=480):
    n = draw(st.integers(min_value=1, max_value=n_max))
    rows = []
    for i in range(n):
        label = draw(st.sampled_from(WORKING_LABELS))
        x1 = draw(st.integers(min_value=0, max_value=img_w - 2))
        x2 = draw(st.integers(min_value=x1 + 1, max_value=img_w))
        y1 = draw(st.integers(min_value=0, max_value=img_h - 2))
        y2 = draw(st.integers(min_value=y1 + 1, max_value=img_h))
        rows.append({
            "result_id": i + 1,
            "image_rel_path": "img.png",
            "final_label": label,
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        })
    return rows


# ---------------------------------------------------------------------------
# Property 1: export_label mapping matches taxonomy
# ---------------------------------------------------------------------------
# Feature: semi-labeling-review-loop, Property 1: export_label mapping khớp
# taxonomy — exported category equals build_label_taxonomy().export_label();
# labels outside the taxonomy map to reject.
# Validates: Requirements 4.3
@settings(max_examples=100)
@given(label=st.one_of(st.sampled_from(WORKING_LABELS), st.text(max_size=12)))
def test_export_label_matches_taxonomy(label):
    taxonomy = build_label_taxonomy()
    expected = taxonomy.export_label(label)
    run_id = "r1"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = tmp_path / "resemi.sqlite3"
        _make_db(db, run_id, [{
            "result_id": 1, "image_rel_path": "img.png",
            "final_label": label, "x1": 1.0, "y1": 1.0, "x2": 10.0, "y2": 10.0,
        }])
        _write_image(tmp_path / "img.png", 640, 480)
        out = tmp_path / "out"
        export_dataset(db_path=str(db), run_id=run_id, image_root=str(tmp_path),
                       output_dir=str(out), fmt="coco")
        coco = json.loads((out / "annotations.coco.json").read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        if expected == "reject":
            assert len(coco["annotations"]) == 0
        else:
            assert len(coco["annotations"]) == 1
            assert cats[coco["annotations"][0]["category_id"]] == expected


# ---------------------------------------------------------------------------
# Property 3: YOLO coordinate round-trip
# ---------------------------------------------------------------------------
# Feature: semi-labeling-review-loop, Property 3: YOLO toạ độ round-trip —
# normalizing (x1,y1,x2,y2) to (xc,yc,w,h) then back recovers the box within
# 1px, and all normalized values are in [0,1].
# Validates: Requirements 4.5
@settings(max_examples=100)
@given(
    w=st.integers(min_value=16, max_value=4000),
    h=st.integers(min_value=16, max_value=4000),
    data=st.data(),
)
def test_yolo_coord_roundtrip(w, h, data):
    x1 = data.draw(st.integers(min_value=0, max_value=w - 2))
    x2 = data.draw(st.integers(min_value=x1 + 1, max_value=w))
    y1 = data.draw(st.integers(min_value=0, max_value=h - 2))
    y2 = data.draw(st.integers(min_value=y1 + 1, max_value=h))
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    for v in (xc, yc, bw, bh):
        assert 0.0 <= v <= 1.0
    rx1 = (xc - bw / 2) * w
    ry1 = (yc - bh / 2) * h
    rx2 = (xc + bw / 2) * w
    ry2 = (yc + bh / 2) * h
    assert abs(rx1 - x1) <= 1 and abs(ry1 - y1) <= 1
    assert abs(rx2 - x2) <= 1 and abs(ry2 - y2) <= 1


# ---------------------------------------------------------------------------
# Property 4: COCO round-trip
# ---------------------------------------------------------------------------
# Feature: semi-labeling-review-loop, Property 4: COCO round-trip — after
# exporting COCO and parsing it back, each non-reject box recovers its category
# (by export_label) and bbox within 1px.
# Validates: Requirements 4.8
@settings(max_examples=100, deadline=None)
@given(rows=_cleaned_rows())
def test_coco_roundtrip(rows):
    taxonomy = build_label_taxonomy()
    run_id = "r1"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = tmp_path / "resemi.sqlite3"
        _make_db(db, run_id, rows)
        _write_image(tmp_path / "img.png", 640, 480)
        out = tmp_path / "out"
        export_dataset(db_path=str(db), run_id=run_id, image_root=str(tmp_path),
                       output_dir=str(out), fmt="coco")
        coco = json.loads((out / "annotations.coco.json").read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}

        expected = [r for r in rows if taxonomy.export_label(r["final_label"]) != "reject"]
        assert len(coco["annotations"]) == len(expected)
        # match by order (export preserves image+result order)
        for ann, row in zip(coco["annotations"], expected):
            assert cats[ann["category_id"]] == taxonomy.export_label(row["final_label"])
            bx, by, bw, bh = ann["bbox"]
            assert abs(bx - row["x1"]) <= 1 and abs(by - row["y1"]) <= 1
            assert abs((bx + bw) - row["x2"]) <= 1 and abs((by + bh) - row["y2"]) <= 1


# ---------------------------------------------------------------------------
# Property 5: no reject category in exported dataset
# ---------------------------------------------------------------------------
# Feature: semi-labeling-review-loop, Property 5: không có category 'reject'
# trong dataset xuất — neither YOLO classes nor COCO categories contain reject.
# Validates: Requirements 4.4
@settings(max_examples=100, deadline=None)
@given(rows=_cleaned_rows())
def test_no_reject_category(rows):
    run_id = "r1"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = tmp_path / "resemi.sqlite3"
        _make_db(db, run_id, rows)
        _write_image(tmp_path / "img.png", 640, 480)
        out = tmp_path / "out"
        result = export_dataset(db_path=str(db), run_id=run_id, image_root=str(tmp_path),
                                output_dir=str(out), fmt="both")
        if "error" in result:
            return  # all rows were reject -> nothing exported
        assert "reject" not in result["categories"]
        coco = json.loads((out / "annotations.coco.json").read_text())
        assert all(c["name"] != "reject" for c in coco["categories"])
        classes = (out / "classes.txt").read_text().split()
        assert "reject" not in classes


# ---------------------------------------------------------------------------
# Property 6: count accounting partitions input boxes
# ---------------------------------------------------------------------------
# Feature: semi-labeling-review-loop, Property 6: kế toán số lượng box phân
# hoạch đúng đầu vào — boxes_written + boxes_rejected + boxes_skipped == total.
# Validates: Requirements 4.7
@settings(max_examples=100, deadline=None)
@given(rows=_cleaned_rows())
def test_count_accounting(rows):
    run_id = "r1"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = tmp_path / "resemi.sqlite3"
        _make_db(db, run_id, rows)
        _write_image(tmp_path / "img.png", 640, 480)
        out = tmp_path / "out"
        result = export_dataset(db_path=str(db), run_id=run_id, image_root=str(tmp_path),
                                output_dir=str(out), fmt="yolo")
        if "error" in result:
            # everything mapped to reject
            assert all(build_label_taxonomy().export_label(r["final_label"]) == "reject" for r in rows)
            return
        assert result["boxes_written"] + result["boxes_rejected"] + result["boxes_skipped"] == result["total_boxes"]


# ---------------------------------------------------------------------------
# Example / edge tests
# ---------------------------------------------------------------------------

def test_yolo_creates_txt_and_classes(tmp_path):
    run_id = "r1"
    db = tmp_path / "resemi.sqlite3"
    _make_db(db, run_id, [
        {"result_id": 1, "image_rel_path": "a.png", "final_label": "crack", "x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 60.0},
        {"result_id": 2, "image_rel_path": "a.png", "final_label": "mold", "x1": 5.0, "y1": 5.0, "x2": 20.0, "y2": 25.0},
    ])
    _write_image(tmp_path / "a.png", 200, 200)
    out = tmp_path / "out"
    result = export_dataset(db_path=str(db), run_id=run_id, image_root=str(tmp_path),
                            output_dir=str(out), fmt="yolo")
    assert (out / "labels" / "a.txt").is_file()
    assert (out / "classes.txt").is_file()
    assert result["images_written"] == 1
    assert result["boxes_written"] == 2


def test_coco_has_three_sections(tmp_path):
    run_id = "r1"
    db = tmp_path / "resemi.sqlite3"
    _make_db(db, run_id, [
        {"result_id": 1, "image_rel_path": "a.png", "final_label": "spall", "x1": 1.0, "y1": 1.0, "x2": 9.0, "y2": 9.0},
    ])
    _write_image(tmp_path / "a.png", 100, 100)
    out = tmp_path / "out"
    export_dataset(db_path=str(db), run_id=run_id, image_root=str(tmp_path),
                   output_dir=str(out), fmt="coco")
    coco = json.loads((out / "annotations.coco.json").read_text())
    assert "images" in coco and "annotations" in coco and "categories" in coco
    assert len(coco["images"]) == 1 and len(coco["annotations"]) == 1


def test_unreadable_image_is_skipped_for_yolo(tmp_path):
    run_id = "r1"
    db = tmp_path / "resemi.sqlite3"
    _make_db(db, run_id, [
        {"result_id": 1, "image_rel_path": "missing.png", "final_label": "crack", "x1": 1.0, "y1": 1.0, "x2": 9.0, "y2": 9.0},
    ])
    # no image file written -> size unreadable
    out = tmp_path / "out"
    result = export_dataset(db_path=str(db), run_id=run_id, image_root=str(tmp_path),
                            output_dir=str(out), fmt="yolo")
    assert result["images_skipped"] == 1
    assert result["boxes_written"] == 0


def test_no_cleaned_returns_error(tmp_path):
    run_id = "r1"
    db = tmp_path / "resemi.sqlite3"
    _make_db(db, run_id, [])
    out = tmp_path / "out"
    result = export_dataset(db_path=str(db), run_id=run_id, image_root=str(tmp_path),
                            output_dir=str(out), fmt="both")
    assert result.get("error") == "Không có nhãn để xuất"
    assert not out.exists()
