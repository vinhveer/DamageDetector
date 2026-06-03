#!/usr/bin/env python3
"""Export per-image metrics from SQLite to CSV for qualitative visualization."""
import sqlite3
import pandas as pd
from pathlib import Path
import sys

def export_per_image_metrics(model_dir: Path, output_csv: Path):
    """Export per-image metrics from SQLite."""
    sqlite_path = model_dir / "eval" / "unet_eval_metrics.sqlite3"
    if not sqlite_path.exists():
        sqlite_path = model_dir / "eval" / "sam_eval_metrics.sqlite3"

    if not sqlite_path.exists():
        print(f"No SQLite file found in {model_dir / 'eval'}")
        return False

    conn = sqlite3.connect(str(sqlite_path))

    # Get best threshold
    cursor = conn.cursor()
    cursor.execute("SELECT dataset_label, best_threshold FROM datasets WHERE dataset_label LIKE '%crack500%'")
    result = cursor.fetchone()
    if not result:
        print(f"No crack500 data found in {sqlite_path}")
        conn.close()
        return False

    dataset_label, best_threshold = result

    # Export per-image metrics at best threshold
    query = f"""
    SELECT image_name as image_id, dice, iou, precision, recall,
           skeleton_dice, centerline_precision, centerline_recall, component_fragmentation as fragmentation
    FROM image_threshold_metrics
    WHERE dataset_label = ? AND threshold = ?
    """

    df = pd.read_sql_query(query, conn, params=(dataset_label, best_threshold))
    conn.close()

    # Rename columns to match expected format
    df = df.rename(columns={
        'precision': 'precision',
        'recall': 'recall',
        'skeleton_dice': 'skeleton_dice',
        'centerline_precision': 'centerline_p',
        'centerline_recall': 'centerline_r'
    })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Exported {len(df)} rows to {output_csv}")
    return True

if __name__ == "__main__":
    workspace = Path("/mnt/c/Users/Dell Precision 7810/Desktop/quangvinh_workspace")
    model_root = workspace / "model_with_inference/crack_segmentation"

    models = {
        "unet_v1_baseline_b16_img512": "unet_v1",
        "sam_ablation_b1_lora_only_coarse": "sam_b1_lora_only"
    }

    for full_name, short_name in models.items():
        model_dir = model_root / full_name
        output_csv = model_root / short_name / "crack500_test" / "per_image_metrics.csv"
        print(f"Exporting {full_name}...")
        export_per_image_metrics(model_dir, output_csv)
