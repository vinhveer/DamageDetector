#!/usr/bin/env python3
"""Tool — export cleaned_labels to a training dataset (YOLO and/or COCO).

Applies the taxonomy export_label mapping, drops boxes that map to `reject`,
and writes YOLO `.txt` (+ `classes.txt`) and/or a single COCO `.json`.

Invoked by the Electron app via the pybridge allow-list:

    python -m tools.export_dataset --db <resemi.sqlite3> --run-id <run> \
        --image-root <HinhAnh> --output-dir <out> --format {yolo,coco,both}

Prints exactly one JSON result line to stdout for the Electron parser.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.db.schema import connect_output  # noqa: E402
from shared.taxonomy.label_taxonomy import build_label_taxonomy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export resemi cleaned_labels to YOLO/COCO dataset.")
    parser.add_argument("--db", required=True, help="resemi.sqlite3 path.")
    parser.add_argument("--run-id", required=True, help="run_id to export.")
    parser.add_argument("--image-root", required=True, help="Image root for resolving image_rel_path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for dataset files.")
    parser.add_argument("--format", default="both", choices=["yolo", "coco", "both"])
    parser.add_argument("--taxonomy-version-id", default="", help="Override taxonomy version id.")
    parser.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test fractions. Use 1,0,0 for no split.")
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=False,
                        help="Copy source images into YOLO images/<split>. Off keeps legacy labels-only export.")
    return parser


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None


def _read_cleaned(conn, run_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT result_id, image_rel_path, final_label, x1, y1, x2, y2
        FROM cleaned_labels WHERE run_id = ?
        ORDER BY image_rel_path ASC, result_id ASC
        """,
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _parse_split(raw: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("--split must contain exactly 3 comma-separated fractions: train,val,test")
    if any(value < 0 for value in parts):
        raise ValueError("--split fractions must be non-negative")
    total = sum(parts)
    if total <= 0:
        raise ValueError("--split total must be > 0")
    return tuple(value / total for value in parts)  # type: ignore[return-value]


def _assign_splits(image_rels: list[str], *, split: tuple[float, float, float], random_state: int) -> dict[str, str]:
    names = ["train", "val", "test"]
    rels = list(sorted(set(image_rels)))
    rng = random.Random(int(random_state))
    rng.shuffle(rels)
    n = len(rels)
    train_n = int(round(n * split[0]))
    val_n = int(round(n * split[1]))
    if train_n + val_n > n:
        val_n = max(0, n - train_n)
    assignments: dict[str, str] = {}
    for idx, rel in enumerate(rels):
        if idx < train_n:
            part = names[0]
        elif idx < train_n + val_n:
            part = names[1]
        else:
            part = names[2]
        assignments[rel] = part
    return assignments


def _safe_image_name(rel: str, used: set[str]) -> str:
    path = Path(rel)
    name = path.name
    if name not in used:
        used.add(name)
        return name
    stem = "__".join(path.with_suffix("").parts)
    candidate = f"{stem}{path.suffix}"
    counter = 2
    while candidate in used:
        candidate = f"{stem}_{counter}{path.suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _resolve_taxonomy(conn, run_id: str, override: str):
    version_id = str(override or "").strip()
    if not version_id:
        row = conn.execute(
            "SELECT taxonomy_version_id FROM resemi_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        version_id = str(row["taxonomy_version_id"]) if row and row["taxonomy_version_id"] else "label_taxonomy_v1"
    return build_label_taxonomy(version_id=version_id or "label_taxonomy_v1")


def export_dataset(
    *,
    db_path: str,
    run_id: str,
    image_root: str,
    output_dir: str,
    fmt: str,
    taxonomy_version_id: str = "",
    split: str = "0.8,0.1,0.1",
    random_state: int = 17,
    copy_images: bool = False,
) -> dict:
    out_dir = Path(output_dir).expanduser().resolve()
    img_root = Path(image_root).expanduser().resolve()
    conn = connect_output(Path(db_path))
    try:
        taxonomy = _resolve_taxonomy(conn, run_id, taxonomy_version_id)
        cleaned = _read_cleaned(conn, run_id)
    finally:
        conn.close()

    total_boxes = len(cleaned)
    if total_boxes == 0:
        return {"error": "Không có nhãn để xuất"}

    # Map each box to its export label; drop reject.
    kept: list[dict] = []
    boxes_rejected = 0
    for row in cleaned:
        export = taxonomy.export_label(str(row["final_label"]))
        if export == "reject":
            boxes_rejected += 1
            continue
        kept.append({**row, "export_label": export})

    split_tuple = _parse_split(split)

    # Stable category id assignment (sorted export labels).
    categories = sorted({r["export_label"] for r in kept})
    cat_to_id = {name: idx for idx, name in enumerate(categories)}

    out_dir.mkdir(parents=True, exist_ok=True)

    # Group kept boxes by image.
    by_image: dict[str, list[dict]] = {}
    for row in kept:
        by_image.setdefault(str(row["image_rel_path"]), []).append(row)
    split_by_image = _assign_splits(list(by_image), split=split_tuple, random_state=int(random_state))

    used_image_names: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    exported_name_by_rel: dict[str, str] = {}
    for rel in sorted(by_image):
        part = split_by_image.get(rel, "train")
        exported_name_by_rel[rel] = _safe_image_name(rel, used_image_names.setdefault(part, set()))

    # Pre-read image sizes once per image.
    sizes: dict[str, tuple[int, int] | None] = {}
    for rel in by_image:
        sizes[rel] = _image_size(img_root / rel)

    images_written = 0
    boxes_written = 0
    images_skipped = 0
    boxes_skipped = 0

    do_yolo = fmt in ("yolo", "both")
    do_coco = fmt in ("coco", "both")

    if do_yolo:
        labels_dir = out_dir / "labels"
        images_dir = out_dir / "images"
        if copy_images:
            images_dir.mkdir(parents=True, exist_ok=True)
        for split_name in sorted(set(split_by_image.values()) or {"train"}):
            (labels_dir / split_name).mkdir(parents=True, exist_ok=True)
            if copy_images:
                (images_dir / split_name).mkdir(parents=True, exist_ok=True)
        (out_dir / "classes.txt").write_text("\n".join(categories) + "\n", encoding="utf-8")
        data_yaml = {
            "path": str(out_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {idx: name for name, idx in cat_to_id.items()},
        }
        (out_dir / "data.yaml").write_text(
            "path: " + str(out_dir).replace("\\", "/") + "\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            + "".join(f"  {idx}: {name}\n" for idx, name in sorted(data_yaml["names"].items())),
            encoding="utf-8",
        )

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1
    image_id = 1

    for rel, boxes in sorted(by_image.items()):
        wh = sizes.get(rel)
        split_name = split_by_image.get(rel, "train")
        image_name = exported_name_by_rel.get(rel, Path(rel).name)
        stem = Path(image_name).stem

        if do_yolo:
            if wh is None:
                images_skipped += 1
                boxes_skipped += len(boxes)
                # COCO can still emit these (bbox is pixel-space); but if YOLO-only, skip.
                if not do_coco:
                    continue
            else:
                w, h = wh
                lines = []
                for b in boxes:
                    x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                    xc = ((x1 + x2) / 2.0) / w
                    yc = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"{cat_to_id[b['export_label']]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                (labels_dir / split_name / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                if copy_images:
                    src = img_root / rel
                    dst = images_dir / split_name / image_name
                    if src.is_file() and not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                images_written += 1
                boxes_written += len(boxes)

        if do_coco:
            w = wh[0] if wh else 0
            h = wh[1] if wh else 0
            file_name = f"{split_name}/{image_name}" if copy_images else rel
            coco_images.append({"id": image_id, "file_name": file_name, "width": w, "height": h, "split": split_name})
            for b in boxes:
                x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                bw = x2 - x1
                bh = y2 - y1
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_to_id[b["export_label"]],
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1
            image_id += 1
            if not do_yolo:
                images_written += 1
                boxes_written += len(boxes)

    if do_coco:
        coco = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": cat_to_id[name], "name": name} for name in categories],
        }
        (out_dir / "annotations.coco.json").write_text(
            json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # For YOLO+both, boxes_written/images_written counted on the YOLO branch.
    return {
        "images_written": images_written,
        "boxes_written": boxes_written,
        "boxes_rejected": boxes_rejected,
        "images_skipped": images_skipped,
        "boxes_skipped": boxes_skipped,
        "total_boxes": total_boxes,
        "categories": categories,
        "format": fmt,
        "output_dir": str(out_dir),
        "split": {key: sum(1 for value in split_by_image.values() if value == key) for key in ("train", "val", "test")},
        "copy_images": bool(copy_images),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_dataset(
            db_path=str(args.db),
            run_id=str(args.run_id),
            image_root=str(args.image_root),
            output_dir=str(args.output_dir),
            fmt=str(args.format),
            taxonomy_version_id=str(args.taxonomy_version_id or ""),
            split=str(args.split),
            random_state=int(args.random_state),
            copy_images=bool(args.copy_images),
        )
    except Exception as exc:  # noqa: BLE001 - surface failure as JSON for the UI
        print(json.dumps({"error": str(exc)}), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
