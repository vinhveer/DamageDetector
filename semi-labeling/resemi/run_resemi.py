#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_lab_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "DamageDetector").exists() and (candidate / "infer_results").exists():
            return candidate
    return current.parents[3]


def _prepare_imports() -> None:
    package_parent = Path(__file__).resolve().parents[1]
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))


_LAB_ROOT = _resolve_lab_root()
_prepare_imports()

from resemi.crop_generation import parse_view_specs
from resemi.pipeline import ResemiConfig, ResemiPipeline


def default_source_db() -> Path:
    return _LAB_ROOT / "infer_results" / "semi-labeling" / "step2_sematic" / "damage_scan.sqlite3"


def default_output_db() -> Path:
    return _LAB_ROOT / "infer_results" / "semi-labeling" / "resemi" / "resemi.sqlite3"


def default_image_root() -> Path:
    return _LAB_ROOT / "data" / "HinhAnh"


def parse_labels(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw or "").split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Resemi v2 semantic-cleaning SQLite artifact from Step 2 OpenCLIP results.")
    parser.add_argument("--source-db", default=str(default_source_db()), help="Step 2 damage_scan.sqlite3 with OpenCLIP semantic results.")
    parser.add_argument("--output-db", default=str(default_output_db()), help="Output Resemi SQLite artifact.")
    parser.add_argument("--semantic-run-id", default="latest", help="Semantic run id to import, or latest.")
    parser.add_argument("--dedup-db", default="", help="Optional Step 4 dedup SQLite. If set, only keep=1 result_id rows are imported.")
    parser.add_argument("--dedup-run-id", default="latest", help="Dedup run id to filter by, or latest.")
    parser.add_argument("--labels", default="", help="Comma-separated source labels to import. Empty = all labels.")
    parser.add_argument("--limit", type=int, default=0, help="Debug mode: import first N detections. 0 = all.")
    parser.add_argument("--run-id", default="", help="Optional stable output run id. Existing rows for this run are replaced.")
    parser.add_argument("--prototype-version-id", default="", help="Optional audited prototype version id. No prototype voting is performed in v0.")
    parser.add_argument("--export-dir", default="", help="Optional export directory. Default: output DB parent / exports / run_id.")
    parser.add_argument("--image-root", default=str(default_image_root()), help="Image root used to resolve source images for multi-crop generation.")
    parser.add_argument("--crop-dir", default="", help="Optional crop cache directory. Default: output DB parent / crops / run_id.")
    parser.add_argument("--crop-views", default="tight,pad10,pad25,context", help="Comma-separated views: tight,pad10,pad25,context.")
    parser.add_argument("--generate-crops", action=argparse.BooleanOptionalAction, default=True, help="Generate multi-crop PNG cache and crop_views rows.")
    parser.add_argument("--taxonomy-version-id", default="label_taxonomy_v1")
    parser.add_argument("--stain-export-label", default="stain", choices=["stain", "mold", "reject"], help="Downstream export mapping for stain.")
    parser.add_argument("--accept-threshold", type=float, default=0.75)
    parser.add_argument("--suspect-threshold", type=float, default=0.50)
    parser.add_argument("--low-margin-threshold", type=float, default=0.03)
    parser.add_argument("--strong-margin-threshold", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ResemiConfig(
        source_db=Path(args.source_db).expanduser().resolve(),
        output_db=Path(args.output_db).expanduser().resolve(),
        semantic_run_id=str(args.semantic_run_id),
        dedup_db=Path(args.dedup_db).expanduser().resolve() if str(args.dedup_db or "").strip() else None,
        dedup_run_id=str(args.dedup_run_id),
        labels=parse_labels(args.labels),
        limit=int(args.limit),
        run_id=str(args.run_id),
        prototype_version_id=str(args.prototype_version_id),
        export_dir=Path(args.export_dir).expanduser().resolve() if str(args.export_dir or "").strip() else None,
        image_root=Path(args.image_root).expanduser().resolve() if str(args.image_root or "").strip() else None,
        crop_dir=Path(args.crop_dir).expanduser().resolve() if str(args.crop_dir or "").strip() else None,
        crop_view_specs=parse_view_specs(str(args.crop_views)),
        generate_crops=bool(args.generate_crops),
        taxonomy_version_id=str(args.taxonomy_version_id),
        stain_export_label=str(args.stain_export_label),
        accept_threshold=float(args.accept_threshold),
        suspect_threshold=float(args.suspect_threshold),
        low_margin_threshold=float(args.low_margin_threshold),
        strong_margin_threshold=float(args.strong_margin_threshold),
    )
    summary = ResemiPipeline(config).run()
    print(f"run_id={summary.run_id}")
    print(f"source_semantic_run_id={summary.source_semantic_run_id}")
    print(f"sqlite={summary.output_db}")
    print(f"total_detections={summary.total_detections}")
    print(f"cleaned_count={summary.cleaned_count}")
    print(f"suspect_count={summary.suspect_count}")
    print(f"reject_count={summary.reject_count}")
    print(f"export_dir={summary.export_dir}")
    print(f"review_csv={summary.review_csv}")
    print(f"review_json={summary.review_json}")
    print(f"cleaned_csv={summary.cleaned_csv}")
    print(f"cleaned_json={summary.cleaned_json}")
    print(f"box_cleanup_csv={summary.box_cleanup_csv}")
    print(f"box_cleanup_json={summary.box_cleanup_json}")
    print(f"box_review_csv={summary.box_review_csv}")
    print(f"box_review_json={summary.box_review_json}")
    print(f"semantic_ensemble_csv={summary.semantic_ensemble_csv}")
    print(f"semantic_ensemble_json={summary.semantic_ensemble_json}")
    print(f"label_taxonomy_json={summary.label_taxonomy_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
