# Step 3: Spatial duplicate box filter

This step removes duplicate semantic boxes with spatial overlap rules.

It does not mutate the source step 2 SQLite DB. It writes two SQLite outputs:

- `filtered.sqlite3`: all processed boxes with `keep=1/0` decisions.
- `suspect.sqlite3`: uncertain cases for Streamlit review.

## Run

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
python semi-labeling/step3_spatial_filter/filter_duplicates.py \
  --db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3 \
  --image-root /Users/nguyenquangvinh/Desktop/Lab/HinhAnh \
  --semantic-run-id latest
```

Debug on selected images:

```bash
python semi-labeling/step3_spatial_filter/filter_duplicates.py \
  --db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3 \
  --image-root /Users/nguyenquangvinh/Desktop/Lab/HinhAnh \
  --image-rel-path "DSC01275__roi5.png,DSC01275__roi6.png"
```

## Current algorithm

1. Group boxes by image and predicted label.
2. Treat `crack`, `mold`, and `spall` as spatial-only labels by default.
3. Keep the highest quality box when boxes overlap:
   - `IoU >= 0.30`, or
   - `containment >= 0.70`.
4. Save dropped boxes with `drop_reason = final_spatial_overlap`.

Quality score:

```text
0.55 * OpenCLIP confidence + 0.25 * GroundingDINO score + 0.20 * geometry score
```

DINOv2 fallback code is still present for future non-spatial-only labels, but it is not used by the current default rule.
