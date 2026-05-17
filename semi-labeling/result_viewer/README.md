# Semi-labeling Step5 Result Viewer

PySide6 local review UI for step5, using `step4_feature_grouping` SQLite outputs as input.

## Run

```bash
python /Users/nguyenquangvinh/Desktop/Lab/DamageDetector/semi-labeling/result_viewer/main.py
```

## Inputs

- `Feature DB`: `/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step4_feature_grouping/feature_groups.sqlite3`.
- `Source DB`: `/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3`.
- `Image root`: usually `/Users/nguyenquangvinh/Desktop/Lab/HinhAnh`.

The app defaults to the canonical feature grouping output:

```text
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step4_feature_grouping/feature_groups.sqlite3
```

## UI Flow

- Group list page has 3 tabs: `crack`, `mold`, `spall`.
- Each tab shows group cards.
- Click `Open group` to view the images in that group.
- Detail page has lazy-loaded crop thumbnails, single-image mode, an image loading progress bar, and a result table.
- Sidebar controls image size and crop padding.

This tool is review-only. It does not modify any SQLite database or dataset files.
