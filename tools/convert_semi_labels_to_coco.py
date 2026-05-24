from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_split.assign import assign_sources, cluster_sources, group_embeddings
from data_split.embedding import embed_images
from data_split.dataset import infer_source_id
from data_split.types import SampleRecord
from object_detection.dinov2.dinov2_prototypes import default_dinov2_embedding_checkpoint


_WORKSPACE_ROOT = _REPO_ROOT.parent
_DEFAULT_LABELS_CSV = _WORKSPACE_ROOT / "infer_results/semi-labeling/step7_label_review/final_labels_10f76adfeb234581.csv"
_DEFAULT_SEMANTIC_DB = _WORKSPACE_ROOT / "infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3"
_DEFAULT_IMAGE_ROOT = _WORKSPACE_ROOT / "HinhAnh"
_DEFAULT_OUTPUT_ROOT = _WORKSPACE_ROOT / "BestDatasets/semi_labeling_coco"


@dataclass(frozen=True)
class LabelRow:
    result_id: int
    image_rel_path: str
    cluster_id: int
    predicted_label_step2: str
    final_class: str
    review_source: str
    confidence: float


@dataclass(frozen=True)
class DetectionRow:
    result_id: int
    image_rel_path: str
    x1: float
    y1: float
    x2: float
    y2: float
    width: int
    height: int


@dataclass(frozen=True)
class AnnotationRecord:
    result_id: int
    image_rel_path: str
    image_path: Path
    source_id: str
    category_name: str
    category_id: int
    bbox: tuple[float, float, float, float]
    area: float
    width: int
    height: int
    confidence: float
    cluster_id: int
    predicted_label_step2: str
    review_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert reviewed semi-labeling CSV + semantic SQLite detections to a train/val COCO dataset.",
    )
    parser.add_argument("--labels-csv", type=Path, default=_DEFAULT_LABELS_CSV, help="Final reviewed labels CSV.")
    parser.add_argument("--semantic-db", type=Path, default=_DEFAULT_SEMANTIC_DB, help="Step 2 semantic SQLite DB containing boxes.")
    parser.add_argument("--image-root", type=Path, default=_DEFAULT_IMAGE_ROOT, help="Root folder containing source images.")
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT, help="Output COCO dataset root.")
    parser.add_argument("--splits", nargs=2, type=float, default=(0.8, 0.2), metavar=("TRAIN", "VAL"), help="Train/val split ratios.")
    parser.add_argument("--split-names", nargs=2, default=("train", "val"), metavar=("TRAIN", "VAL"), help="Output split names.")
    parser.add_argument("--categories", nargs="+", default=("crack", "spall", "mold"), help="Accepted categories, in COCO id order starting at 1.")
    parser.add_argument("--num-clusters", type=int, default=36, help="Number of source visual clusters for data_split.")
    parser.add_argument("--checkpoint", default=default_dinov2_embedding_checkpoint(), help="DINOv2 checkpoint/model id used by data_split embeddings.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"), help="Torch device preference for embeddings.")
    parser.add_argument("--dedupe-iou", type=float, default=0.95, help="Drop same-image same-class boxes with IoU >= this value. Set <=0 to disable.")
    parser.add_argument("--include-reject-only-images", action="store_true", help="Keep images that only have reject labels as empty negative COCO images.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output-root if it already exists.")
    return parser.parse_args()


def normalize_two_ratios(values: Iterable[float]) -> list[float]:
    ratios = [float(value) for value in values]
    if len(ratios) != 2:
        raise ValueError("Exactly two ratios are required for train/val split.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return [value / total for value in ratios]


def infer_roi_source_id(image_rel_path: str) -> str:
    stem = Path(image_rel_path).stem
    stem = re.sub(r"(__roi|_roi)\d+$", "", stem)
    return infer_source_id(stem)


def load_label_rows(labels_csv: Path) -> list[LabelRow]:
    rows: list[LabelRow] = []
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"result_id", "image_rel_path", "cluster_id", "predicted_label_step2", "final_class", "source", "confidence"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Labels CSV is missing required columns: {sorted(missing)}")

        for raw in reader:
            final_class = str(raw["final_class"]).strip().lower()
            rows.append(
                LabelRow(
                    result_id=int(raw["result_id"]),
                    image_rel_path=str(raw["image_rel_path"]).strip(),
                    cluster_id=int(raw["cluster_id"]),
                    predicted_label_step2=str(raw["predicted_label_step2"]).strip().lower(),
                    final_class=final_class,
                    review_source=str(raw["source"]).strip(),
                    confidence=float(raw["confidence"]),
                )
            )
    return rows


def fetch_detection_rows(db_path: Path, result_ids: list[int]) -> dict[int, DetectionRow]:
    if not result_ids:
        return {}
    out: dict[int, DetectionRow] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for start in range(0, len(result_ids), 900):
            chunk = result_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
                SELECT
                    r.result_id,
                    r.image_rel_path,
                    r.x1,
                    r.y1,
                    r.x2,
                    r.y2,
                    i.width,
                    i.height
                FROM openclip_semantic_results AS r
                JOIN images AS i ON r.image_id = i.image_id
                WHERE r.result_id IN ({placeholders})
            """
            for row in conn.execute(query, chunk):
                result_id, image_rel_path, x1, y1, x2, y2, width, height = row
                out[int(result_id)] = DetectionRow(
                    result_id=int(result_id),
                    image_rel_path=str(image_rel_path),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    width=int(width),
                    height=int(height),
                )
    finally:
        conn.close()
    return out


def resolve_image_path(image_root: Path, image_rel_path: str) -> Path:
    image_path = image_root / image_rel_path
    if image_path.is_file():
        return image_path
    raise FileNotFoundError(f"Missing image for {image_rel_path}: {image_path}")


def clip_box(det: DetectionRow) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(det.x1), float(det.width)))
    y1 = max(0.0, min(float(det.y1), float(det.height)))
    x2 = max(0.0, min(float(det.x2), float(det.width)))
    y2 = max(0.0, min(float(det.y2), float(det.height)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    return x1, y1, box_w, box_h


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / max(union, 1e-12))


def build_annotations(
    labels: list[LabelRow],
    detections: dict[int, DetectionRow],
    image_root: Path,
    category_to_id: dict[str, int],
) -> tuple[list[AnnotationRecord], list[str]]:
    annotations: list[AnnotationRecord] = []
    warnings: list[str] = []
    for label in labels:
        if label.final_class == "reject":
            continue
        if label.final_class not in category_to_id:
            warnings.append(f"Skipped unknown class result_id={label.result_id}: {label.final_class}")
            continue
        det = detections.get(label.result_id)
        if det is None:
            warnings.append(f"Missing SQLite detection for result_id={label.result_id}")
            continue
        if det.image_rel_path != label.image_rel_path:
            warnings.append(
                f"CSV/SQLite image_rel_path mismatch result_id={label.result_id}: "
                f"csv={label.image_rel_path} sqlite={det.image_rel_path}"
            )
        bbox = clip_box(det)
        if bbox is None:
            warnings.append(f"Skipped invalid bbox result_id={label.result_id}")
            continue
        image_path = resolve_image_path(image_root, label.image_rel_path)
        box_w, box_h = bbox[2], bbox[3]
        annotations.append(
            AnnotationRecord(
                result_id=label.result_id,
                image_rel_path=label.image_rel_path,
                image_path=image_path,
                source_id=infer_roi_source_id(label.image_rel_path),
                category_name=label.final_class,
                category_id=category_to_id[label.final_class],
                bbox=bbox,
                area=float(box_w * box_h),
                width=det.width,
                height=det.height,
                confidence=label.confidence,
                cluster_id=label.cluster_id,
                predicted_label_step2=label.predicted_label_step2,
                review_source=label.review_source,
            )
        )
    return annotations, warnings


def dedupe_annotations(annotations: list[AnnotationRecord], threshold: float) -> tuple[list[AnnotationRecord], int]:
    if threshold <= 0.0:
        return annotations, 0
    grouped: dict[tuple[str, int], list[AnnotationRecord]] = {}
    for ann in annotations:
        grouped.setdefault((ann.image_rel_path, ann.category_id), []).append(ann)

    kept: list[AnnotationRecord] = []
    removed = 0
    for key in sorted(grouped):
        current: list[AnnotationRecord] = []
        for ann in sorted(grouped[key], key=lambda item: (item.confidence, item.area), reverse=True):
            if any(box_iou(ann.bbox, existing.bbox) >= threshold for existing in current):
                removed += 1
                continue
            current.append(ann)
        kept.extend(sorted(current, key=lambda item: item.result_id))
    return sorted(kept, key=lambda item: (item.image_rel_path, item.result_id)), removed


def build_image_records(annotations: list[AnnotationRecord]) -> dict[str, dict[str, object]]:
    image_records: dict[str, dict[str, object]] = {}
    for ann in annotations:
        existing = image_records.get(ann.image_rel_path)
        if existing is None:
            image_records[ann.image_rel_path] = {
                "image_rel_path": ann.image_rel_path,
                "image_path": ann.image_path,
                "source_id": ann.source_id,
                "width": ann.width,
                "height": ann.height,
                "annotation_area": ann.area,
                "annotation_count": 1,
            }
            continue
        existing["annotation_area"] = float(existing["annotation_area"]) + ann.area
        existing["annotation_count"] = int(existing["annotation_count"]) + 1
    return image_records


def add_reject_only_images(
    image_records: dict[str, dict[str, object]],
    labels: list[LabelRow],
    detections: dict[int, DetectionRow],
    image_root: Path,
) -> int:
    added = 0
    for label in labels:
        if label.image_rel_path in image_records:
            continue
        det = detections.get(label.result_id)
        if det is None:
            continue
        image_path = resolve_image_path(image_root, label.image_rel_path)
        image_records[label.image_rel_path] = {
            "image_rel_path": label.image_rel_path,
            "image_path": image_path,
            "source_id": infer_roi_source_id(label.image_rel_path),
            "width": det.width,
            "height": det.height,
            "annotation_area": 0.0,
            "annotation_count": 0,
        }
        added += 1
    return added


def make_sample_records(image_records: dict[str, dict[str, object]]) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for rel_path in sorted(image_records):
        item = image_records[rel_path]
        width = max(1, int(item["width"]))
        height = max(1, int(item["height"]))
        positive_ratio = min(1.0, float(item["annotation_area"]) / float(width * height))
        image_path = Path(item["image_path"])
        records.append(
            SampleRecord(
                image_path=image_path,
                mask_path=None,
                stem=image_path.stem,
                source_id=str(item["source_id"]),
                positive_ratio=positive_ratio,
            )
        )
    return records


def compute_splits(
    records: list[SampleRecord],
    output_root: Path,
    split_names: list[str],
    split_ratios: list[float],
    num_clusters: int,
    checkpoint: str,
    batch_size: int,
    device: str,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    if len({record.source_id for record in records}) < len(split_names):
        raise ValueError("Not enough source groups to create requested splits.")

    cache_path = output_root / "data_split" / "embedding_cache.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = embed_images(
        image_paths=[record.image_path for record in records],
        checkpoint=checkpoint,
        batch_size=batch_size,
        device_preference=device,
        cache_path=cache_path,
    )
    group_df = group_embeddings(records, embeddings)
    clustered_df = cluster_sources(group_df, num_clusters)
    source_to_split, summary_df = assign_sources(clustered_df, split_names, split_ratios)
    return source_to_split, summary_df, clustered_df


def copy_images(image_records: dict[str, dict[str, object]], source_to_split: dict[str, str], output_root: Path) -> None:
    for rel_path in sorted(image_records):
        item = image_records[rel_path]
        split = source_to_split[str(item["source_id"])]
        dst = output_root / "images" / split / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(item["image_path"]), dst)


def build_coco(
    split_name: str,
    image_records: dict[str, dict[str, object]],
    annotations: list[AnnotationRecord],
    source_to_split: dict[str, str],
    categories: list[str],
) -> dict[str, object]:
    split_images = [
        item
        for item in image_records.values()
        if source_to_split[str(item["source_id"])] == split_name
    ]
    split_images = sorted(split_images, key=lambda item: str(item["image_rel_path"]))
    image_id_by_rel: dict[str, int] = {}
    coco_images: list[dict[str, object]] = []
    for image_id, item in enumerate(split_images, start=1):
        rel_path = str(item["image_rel_path"])
        image_id_by_rel[rel_path] = image_id
        coco_images.append(
            {
                "id": image_id,
                "file_name": rel_path,
                "width": int(item["width"]),
                "height": int(item["height"]),
                "source_id": str(item["source_id"]),
            }
        )

    coco_annotations: list[dict[str, object]] = []
    for ann_id, ann in enumerate(
        [ann for ann in annotations if ann.image_rel_path in image_id_by_rel],
        start=1,
    ):
        coco_annotations.append(
            {
                "id": ann_id,
                "image_id": image_id_by_rel[ann.image_rel_path],
                "category_id": ann.category_id,
                "bbox": [round(value, 4) for value in ann.bbox],
                "area": round(ann.area, 4),
                "iscrowd": 0,
                "segmentation": [],
                "source_result_id": ann.result_id,
                "source_cluster_id": ann.cluster_id,
                "source_confidence": ann.confidence,
                "source_review": ann.review_source,
                "predicted_label_step2": ann.predicted_label_step2,
            }
        )

    return {
        "info": {
            "description": "Reviewed semi-labeling structural damage dataset",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "date_created": datetime.now(timezone.utc).isoformat(),
            "split": split_name,
        },
        "licenses": [],
        "categories": [
            {"id": idx, "name": name, "supercategory": "damage"}
            for idx, name in enumerate(categories, start=1)
        ],
        "images": coco_images,
        "annotations": coco_annotations,
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_split_workbook(
    output_root: Path,
    image_records: dict[str, dict[str, object]],
    source_to_split: dict[str, str],
    summary_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
) -> Path:
    cluster_by_source = dict(zip(clustered_df["source_id"], clustered_df["cluster_id"]))
    rows: list[dict[str, object]] = []
    for rel_path in sorted(image_records):
        item = image_records[rel_path]
        source_id = str(item["source_id"])
        rows.append(
            {
                "image_rel_path": rel_path,
                "split": source_to_split[source_id],
                "source_id": source_id,
                "cluster_id": int(cluster_by_source[source_id]),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "annotation_count": int(item["annotation_count"]),
                "annotation_area": float(item["annotation_area"]),
                "image_path": str(item["image_path"]),
            }
        )
    assignments = pd.DataFrame(rows)
    workbook_path = output_root / "split_assignments.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        assignments.to_excel(writer, sheet_name="assignments", index=False)
        for split_name in sorted(assignments["split"].unique()):
            assignments[assignments["split"] == split_name].to_excel(writer, sheet_name=split_name, index=False)
    return workbook_path


def class_counts_by_split(annotations: list[AnnotationRecord], source_to_split: dict[str, str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for ann in annotations:
        split = source_to_split[ann.source_id]
        counts.setdefault(split, {})
        counts[split][ann.category_name] = counts[split].get(ann.category_name, 0) + 1
    return counts


def find_source_leakage(image_records: dict[str, dict[str, object]], source_to_split: dict[str, str]) -> dict[str, list[str]]:
    source_splits: dict[str, set[str]] = {}
    for item in image_records.values():
        source_id = str(item["source_id"])
        source_splits.setdefault(source_id, set()).add(source_to_split[source_id])
    return {source_id: sorted(splits) for source_id, splits in source_splits.items() if len(splits) > 1}


def ensure_clean_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    labels_csv = args.labels_csv.expanduser().resolve()
    semantic_db = args.semantic_db.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_names = [str(value) for value in args.split_names]
    split_ratios = normalize_two_ratios(args.splits)
    categories = [str(value).strip().lower() for value in args.categories if str(value).strip()]
    if not categories:
        raise ValueError("At least one category is required.")
    category_to_id = {name: idx for idx, name in enumerate(categories, start=1)}

    ensure_clean_output(output_root, bool(args.overwrite))

    labels = load_label_rows(labels_csv)
    requested_detection_ids = sorted({label.result_id for label in labels})
    detections = fetch_detection_rows(semantic_db, requested_detection_ids)
    annotations, warnings = build_annotations(labels, detections, image_root, category_to_id)
    annotations, deduped_count = dedupe_annotations(annotations, float(args.dedupe_iou))
    image_records = build_image_records(annotations)
    reject_only_count = 0
    if args.include_reject_only_images:
        reject_only_count = add_reject_only_images(image_records, labels, detections, image_root)
    if not annotations:
        raise ValueError("No valid annotations were produced.")
    if not image_records:
        raise ValueError("No valid images were produced.")

    sample_records = make_sample_records(image_records)
    source_to_split, summary_df, clustered_df = compute_splits(
        records=sample_records,
        output_root=output_root,
        split_names=split_names,
        split_ratios=split_ratios,
        num_clusters=int(args.num_clusters),
        checkpoint=str(args.checkpoint),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )

    copy_images(image_records, source_to_split, output_root)
    annotations_dir = output_root / "annotations"
    coco_paths: dict[str, str] = {}
    for split_name in split_names:
        coco = build_coco(split_name, image_records, annotations, source_to_split, categories)
        coco_path = annotations_dir / f"instances_{split_name}.json"
        write_json(coco_path, coco)
        coco_paths[split_name] = str(coco_path)

    workbook_path = write_split_workbook(output_root, image_records, source_to_split, summary_df, clustered_df)
    source_leakage = find_source_leakage(image_records, source_to_split)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_csv": str(labels_csv),
        "semantic_db": str(semantic_db),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "categories": category_to_id,
        "split_names": split_names,
        "split_ratios": split_ratios,
        "num_clusters": int(args.num_clusters),
        "checkpoint": str(args.checkpoint),
        "label_rows": len(labels),
        "sqlite_rows_found": len(detections),
        "annotations": len(annotations),
        "images": len(image_records),
        "source_groups": len(source_to_split),
        "deduped_annotations": deduped_count,
        "reject_only_images_added": reject_only_count,
        "class_counts_by_split": class_counts_by_split(annotations, source_to_split),
        "split_summary": summary_df.to_dict(orient="records"),
        "coco_annotations": coco_paths,
        "split_workbook": str(workbook_path),
        "source_leakage": source_leakage,
        "warnings": warnings[:200],
        "warning_count": len(warnings),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    report_path = output_root / "conversion_report.json"
    write_json(report_path, report)

    print(f"Wrote COCO dataset to: {output_root}")
    print(f"Images: {len(image_records)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Source groups: {len(source_to_split)}")
    print(summary_df.to_string(index=False))
    print(f"Workbook: {workbook_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
