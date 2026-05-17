# Semi-labeling Pipeline

Code nam o `DamageDetector/semi-labeling`. Ket qua nam o `infer_results/semi-labeling` va duoc chia theo step, moi step chua DB output cua step do.

## Layout Ket Qua

```text
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/
  step1_grounding_dino/
    damage_scan.sqlite3

  step2_sematic/
    damage_scan.sqlite3
    step2_shard0.sqlite3
    step2_shard1.sqlite3
    step2_merged.sqlite3

  step3_spatial_filter/
    filtered.sqlite3
    suspect.sqlite3
    dryrun_filtered.sqlite3
    dryrun_suspect.sqlite3

  step4_feature_grouping/
    feature_groups.sqlite3
    previous_feature_groups.sqlite3
```

`step2_sematic` giu nguyen typo theo code hien tai.

## Step 1: GroundingDINO

Output DB:

```text
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step1_grounding_dino/damage_scan.sqlite3
```

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
python semi-labeling/step1_gdino_labeling/run_damage_scan.py \
  --input-dir /Users/nguyenquangvinh/Desktop/Lab/HinhAnh \
  --db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step1_grounding_dino/damage_scan.sqlite3
```

## Step 2: OpenCLIP Semantic

Input DB la copy tu step 1. Output DB chuan:

```text
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3
```

```bash
cp /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step1_grounding_dino/damage_scan.sqlite3 \
  /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3

cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
python semi-labeling/step2_sematic/run_openclip_semantic.py \
  --db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3 \
  --image-root /Users/nguyenquangvinh/Desktop/Lab/HinhAnh \
  --source-run-id latest \
  --stage final
```

## Step 3: Spatial Filter

Input DB:

```text
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3
```

Output DB:

```text
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step3_spatial_filter/filtered.sqlite3
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step3_spatial_filter/suspect.sqlite3
```

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
python semi-labeling/step3_spatial_filter/filter_duplicates.py \
  --db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3 \
  --image-root /Users/nguyenquangvinh/Desktop/Lab/HinhAnh \
  --output-dir /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step3_spatial_filter \
  --semantic-run-id latest
```

## Step 4: Feature Grouping

Input DB:

```text
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step3_spatial_filter/filtered.sqlite3
```

Output DB:

```text
/Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step4_feature_grouping/feature_groups.sqlite3
```

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
python semi-labeling/step4_feature_grouping/run_feature_grouping.py \
  --source-db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3 \
  --filtered-db /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step3_spatial_filter/filtered.sqlite3 \
  --image-root /Users/nguyenquangvinh/Desktop/Lab/HinhAnh \
  --output-dir /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step4_feature_grouping
```

## App Viewer

Default path trong `DamageDetector/app`:

```text
Source DB:  /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3
Feature DB: /Users/nguyenquangvinh/Desktop/Lab/infer_results/semi-labeling/step4_feature_grouping/feature_groups.sqlite3
Image root: /Users/nguyenquangvinh/Desktop/Lab/HinhAnh
```
