#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from PIL import Image


def _slugify(text: str) -> str:
    out: list[str] = []
    prev_sep = False
    for ch in str(text or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        elif not prev_sep:
            out.append("-")
            prev_sep = True
    slug = "".join(out).strip("-")
    return slug or "label"


def _resolve_semantic_run_id(conn: sqlite3.Connection, requested: str) -> str:
    raw = str(requested or "latest").strip()
    if raw and raw.lower() != "latest":
        return raw
    row = conn.execute(
        "SELECT semantic_run_id FROM openclip_semantic_runs ORDER BY created_at_utc DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("No OpenCLIP semantic run found in SQLite.")
    return str(row[0])


def _read_results(conn: sqlite3.Connection, semantic_run_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT res.result_id, res.source_detection_id, res.source_run_id, run.input_dir AS source_input_dir,
                   res.image_rel_path, res.image_path,
                   x1, y1, x2, y2, predicted_label, predicted_probability_pct, crop_path, status
            FROM openclip_semantic_results res
            JOIN runs run ON run.run_id = res.source_run_id
            WHERE semantic_run_id = ? AND status = 'ok'
            ORDER BY predicted_label, result_id
            """,
            (semantic_run_id,),
        ).fetchall()
    )


def _resolve_image_path(row: sqlite3.Row, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(row["image_rel_path"] or "").strip()
    stored_path = str(row["image_path"] or "").strip()
    source_input_dir = Path(str(row["source_input_dir"] or "")).expanduser()

    if image_root is not None:
        candidates.append(image_root / rel_path)
        candidates.append(image_root / Path(stored_path).name)

    if stored_path:
        stored = Path(stored_path).expanduser()
        if stored.is_absolute():
            candidates.append(stored)
        else:
            candidates.append(source_input_dir / stored_path)

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
        return (image_root / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()


def _crop_box(row: sqlite3.Row, output_path: Path, *, image_root: Path | None) -> None:
    image_path = _resolve_image_path(row, image_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        x1 = max(0, min(width - 1, int(float(row["x1"]))))
        y1 = max(0, min(height - 1, int(float(row["y1"]))))
        x2 = max(0, min(width, int(round(float(row["x2"])))))
        y2 = max(0, min(height, int(round(float(row["y2"])))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop box for result_id={row['result_id']}: {(x1, y1, x2, y2)}")
        crop = rgb.crop((x1, y1, x2, y2))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)


def export_top1_labels(*, db_path: Path, semantic_run_id: str, output_dir: Path, image_root: Path | None) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        resolved_run_id = _resolve_semantic_run_id(conn, semantic_run_id)
        rows = _read_results(conn, resolved_run_id)
    finally:
        conn.close()

    if not rows:
        raise RuntimeError(f"No successful semantic results found for semantic_run_id={semantic_run_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    counts: dict[str, int] = {}
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "semantic_run_id",
                "result_id",
                "source_detection_id",
                "predicted_label",
                "predicted_probability_pct",
                "image_rel_path",
                "exported_crop_path",
            ]
        )
        for row in rows:
            label = str(row["predicted_label"] or "unknown")
            counts[label] = counts.get(label, 0) + 1
            rel_path = Path(str(row["image_rel_path"] or "image"))
            ext = Path(str(row["crop_path"] or "")).suffix or ".png"
            filename = f"{rel_path.stem}__res{int(row['result_id'])}__det{int(row['source_detection_id'])}{ext}"
            label_dir = output_dir / _slugify(label)
            exported_crop_path = label_dir / filename
            crop_path = Path(str(row["crop_path"] or "")).expanduser()
            if crop_path.is_file():
                exported_crop_path.parent.mkdir(parents=True, exist_ok=True)
                exported_crop_path.write_bytes(crop_path.read_bytes())
            else:
                _crop_box(row, exported_crop_path, image_root=image_root)
            writer.writerow(
                [
                    resolved_run_id,
                    int(row["result_id"]),
                    int(row["source_detection_id"]),
                    label,
                    float(row["predicted_probability_pct"]),
                    str(row["image_rel_path"]),
                    str(exported_crop_path),
                ]
            )
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export OpenCLIP box crops grouped by top-1 predicted label.")
    parser.add_argument("--db", required=True, help="Path to damage_scan.sqlite3.")
    parser.add_argument("--image-root", default="", help="Override image root used to resolve `image_rel_path` from the DB.")
    parser.add_argument("--semantic-run-id", default="latest", help="OpenCLIP semantic run id to export. Default: latest")
    parser.add_argument("--output-dir", required=True, help="Output directory for grouped crops.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    counts = export_top1_labels(
        db_path=db_path,
        semantic_run_id=str(args.semantic_run_id),
        output_dir=output_dir,
        image_root=Path(args.image_root).expanduser().resolve() if str(args.image_root or "").strip() else None,
    )
    print(f"export_dir={output_dir}")
    for label, total in sorted(counts.items()):
        print(f"{label}: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
