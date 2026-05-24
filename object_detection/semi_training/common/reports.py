from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .coco import write_json


def write_prediction_report(*, output_dir: str | Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = write_json(out / "semantic_validation_report.json", summary)
    csv_path = out / "predictions.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["image_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    review_path = out / "review_queue.csv"
    review_rows = [row for row in rows if str(row.get("decision", "")) != "accept"]
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)
    return {
        "report": str(report_path),
        "predictions_csv": str(csv_path),
        "review_queue_csv": str(review_path),
        "preview_dir": str(out / "preview"),
    }
