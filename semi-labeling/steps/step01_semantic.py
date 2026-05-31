#!/usr/bin/env python3
"""Step 01 — Semantic Init.

Reads source detections from the Step 2 OpenCLIP DB, builds the semantic
ensemble, applies the initial decision policy, runs BBox quality analysis, and
writes the baseline into resemi.sqlite3 (semantic_decisions, cleaned_labels,
review_queue, box_* tables).

Crop generation is off by default — run step02 separately so slow crop I/O
doesn't block the semantic pass. Pass --generate-crops to do both in one go.

Inputs:  step2_sematic/damage_scan.sqlite3  (+ optional step4 dedup)
Outputs: resemi.sqlite3
"""
from __future__ import annotations

import argparse

from lib import bootstrap

bootstrap.ensure_on_path()

from lib.crop_generation import parse_view_specs  # noqa: E402
from lib.paths import default_dedup_db, default_image_root, default_resemi_db, default_source_db  # noqa: E402
from lib.pipeline import ResemiConfig, ResemiPipeline  # noqa: E402


def parse_labels(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw or "").split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 01: semantic init — read OpenCLIP results, build ensemble, "
            "apply decision policy, BBox quality. Writes resemi.sqlite3."
        )
    )
    parser.add_argument("--source-db", default=str(default_source_db()),
                        help="Step 2 damage_scan.sqlite3 with OpenCLIP semantic results.")
    parser.add_argument("--output-db", default=str(default_resemi_db()),
                        help="Output Resemi SQLite artifact.")
    parser.add_argument("--semantic-run-id", default="latest",
                        help="Semantic run id to import, or 'latest'.")
    parser.add_argument("--dedup-db", default="",
                        help=f"Optional Step 4 dedup SQLite (keep=1 filter). Example: {default_dedup_db()}")
    parser.add_argument("--dedup-run-id", default="latest")
    parser.add_argument("--labels", default="",
                        help="Comma-separated source labels to import. Empty = all.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Debug: import first N detections. 0 = all.")
    parser.add_argument("--run-id", default="",
                        help="Stable output run id. Existing rows are replaced.")
    parser.add_argument("--prototype-version-id", default="")
    parser.add_argument("--export-dir", default="",
                        help="Export dir. Default: output DB parent/exports/run_id.")
    parser.add_argument("--image-root", default=str(default_image_root()),
                        help="Image root for multi-crop generation (used only when --generate-crops).")
    parser.add_argument("--crop-dir", default="",
                        help="Crop cache dir. Default: output DB parent/crops/run_id.")
    parser.add_argument("--crop-views", default="tight,pad10,pad25,context")
    parser.add_argument("--generate-crops", action=argparse.BooleanOptionalAction, default=False,
                        help="Generate multi-crop PNGs in this step. Default off — use step02 instead.")
    parser.add_argument("--taxonomy-version-id", default="label_taxonomy_v1")
    parser.add_argument("--stain-export-label", default="stain",
                        choices=["stain", "mold", "reject"])
    parser.add_argument("--accept-threshold", type=float, default=0.75)
    parser.add_argument("--suspect-threshold", type=float, default=0.50)
    parser.add_argument("--low-margin-threshold", type=float, default=0.03)
    parser.add_argument("--strong-margin-threshold", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    from pathlib import Path
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
    if bool(args.generate_crops):
        print(f"review_csv={summary.review_csv}")
        print(f"cleaned_csv={summary.cleaned_csv}")
    else:
        print("crops=skipped (run step02 next)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
