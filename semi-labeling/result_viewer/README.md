# Semi-labeling Result Viewer

PySide6 local viewer for `step4_feature_grouping` SQLite outputs.

## Run

```bash
python /Users/nguyenquangvinh/Desktop/Lab/DamageDetector/semi-labeling/result_viewer/main.py
```

## Inputs

- `Feature DB`: `feature_groups.sqlite3` from step4.
- `Source DB`: `damage_scan.sqlite3` from step2, used for crop coordinates and original image metadata.
- `Image root`: usually `/Users/nguyenquangvinh/Desktop/Lab/HinhAnh`.

The app defaults to the latest full DINOv2 giant output if it exists:

```text
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/2_sematic/step4_feature_grouping_giant_agglo045_full_restart/feature_groups.sqlite3
```

## UI Flow

- Group list page has 3 tabs: `crack`, `mold`, `spall`.
- Each tab shows paginated group cards.
- Click `Open group` to view the images in that group.
- Detail page has paginated crop thumbnails and a result table.
- Sidebar controls card page size, images per page, image size, and crop padding.

This tool is review-only. It does not modify any SQLite database or dataset files.
