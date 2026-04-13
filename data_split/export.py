from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def write_workbook(output_root: Path, assignments: pd.DataFrame, summary: pd.DataFrame) -> Path:
    workbook_path = output_root / "split_assignments.xlsx"
    output_root.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        for split_name in ["train", "val", "test"]:
            split_df = assignments[assignments["split"] == split_name].copy()
            split_df.to_excel(writer, sheet_name=split_name, index=False)
    return workbook_path


def export_split_folders(output_root: Path, assignments: pd.DataFrame) -> None:
    for split_name in ["train", "val", "test"]:
        split_root = output_root / split_name
        if split_root.exists():
            shutil.rmtree(split_root)

    for row in tqdm(assignments.itertuples(index=False), total=int(assignments.shape[0]), desc="Exporting files", unit="image"):
        split_root = output_root / row.split
        target_image = split_root / "images" / Path(row.relative_image_path)
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.image_path, target_image)
        if row.mask_path:
            target_mask = split_root / "masks" / Path(row.relative_mask_path)
            target_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row.mask_path, target_mask)
