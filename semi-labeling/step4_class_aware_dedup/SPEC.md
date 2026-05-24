# Step 4 — Class-Aware Learned Deduplication

## 1. Mục đích

Từ 87k detection (Step 2 output) + embedding cache (Step 3) → ra **~25k detection sạch, non-duplicate**.

Thay thế NMS heuristic (IoU threshold cứng) bằng **learned class-aware duplicate classifier + quality calibration + optional Weighted Box Fusion**.

## 2. Cải tiến so với version cũ

| Aspect | Step 3 cũ (đã archive) | Step 4 mới |
|---|---|---|
| Số hyperparameter | 8 ngưỡng tay | 2 ngưỡng (`τ_dup`, `quality_min`) + class table |
| Embedding | Re-compute mỗi run | Đọc Step 3 cache |
| NMS | Hard binary keep/drop | Soft probability (Soft-NMS family) |
| Crack exception | Hardcoded `if both_crack: continue` | Class-pair feature trong classifier |
| Quality score weights | `0.55/0.25/0.20` arbitrary | LogReg học từ labeled box |
| Keeper selection | Argmax quality | Keep-one cho crack, WBF cho spall/mold |
| Band-aid passes | 2 pass thêm | 1 pass duy nhất |
| Code | 929 dòng | ~1300 dòng (chia 7 module) |

## 3. Pipeline 6 stage

```
[Step 2 detections + Step 3 cached embeddings]
        │
        ▼
①  Candidate pair generation per image
   Pre-filter: IoU > 0.05 OR center_distance < 2·sqrt(area_union)
        │
        ▼
②  Pair feature extraction (13 features)
   IoU, cosine_sim, center_sim, size_sim, aspect_sim,
   same_class, class_pair onehot,
   crack-only: angle_diff, elongation_diff
        │
        ▼
③  Duplicate classifier (LogReg)
   p_dup = σ(W·features + b)
        │
        ▼
④  Union-Find cluster
   merge nếu p_dup ≥ τ_dup
        │
        ▼
⑤  Quality classifier (LogReg)
   p_good = σ(W'·box_features + b')
        │
        ▼
⑥  Representative selection per component
   if class == "crack":         keep_one(argmax p_good)
   else (spall, mold):          weighted_box_fusion(weights = p_good × detector_score)
        │
        ▼
⑦  Hard guards (post-filters, §17)
   - oversized: keeper với area_ratio ≥ 0.70 → drop
   - multi-feature container: keeper area ≥ 0.30 chứa ≥ 2 keeper khác cùng class → drop
        │
        ▼
[dedup.sqlite3]
```

## 4. Feature engineering

### 4.1 Pair features (13-dim)

```python
def pair_features(box_a, box_b, emb_a, emb_b):
    iou_value = iou(box_a, box_b)
    center_dist = center_distance(box_a, box_b)
    union_area = box_a.area + box_b.area - iou_value * min(box_a.area, box_b.area)
    both_crack = (box_a.label == "crack" and box_b.label == "crack")
    return {
        "iou":                iou_value,
        "cos_sim":            float(emb_a @ emb_b),
        "center_sim":         exp(-center_dist / sqrt(max(1.0, union_area))),
        "size_sim":           exp(-abs(log(box_a.area / box_b.area))),
        "aspect_sim":         exp(-abs(log(aspect(box_a) / aspect(box_b)))),
        "same_class":         1.0 if box_a.label == box_b.label else 0.0,
        "class_a_crack":      1.0 if box_a.label == "crack" else 0.0,
        "class_b_crack":      1.0 if box_b.label == "crack" else 0.0,
        "class_pair_spall_mold":  1.0 if {box_a.label, box_b.label} == {"spall", "mold"} else 0.0,
        "class_pair_crack_other": 1.0 if "crack" in {box_a.label, box_b.label} and not both_crack else 0.0,
        "detector_score_diff":    abs(box_a.detector_score - box_b.detector_score),
        "angle_diff":             abs(orientation(box_a) - orientation(box_b)) if both_crack else 0.0,
        "elongation_diff":        abs(log(elongation(box_a) / elongation(box_b))) if both_crack else 0.0,
    }
```

`orientation` và `elongation` xem [§4.3](#43-crack-topology-features).

### 4.2 Single-box features (10-dim) cho quality classifier

```python
def box_features(box, emb, crop):
    return {
        "semantic_prob":   box.predicted_probability_pct / 100,
        "detector_score":  box.detector_score,
        "log_area_ratio":  log(max(1e-6, box.area_ratio)),
        "log_aspect":      log(aspect(box)),
        "edge_density":    sobel_edge_density(crop),      # Sobel magnitude > threshold
        "blur_score":      laplacian_variance(crop),       # cv2.Laplacian variance
        "is_crack":        1.0 if box.label == "crack" else 0.0,
        "is_spall":        1.0 if box.label == "spall" else 0.0,
        "emb_pca_1":       float(pca_components[0] @ emb),
        "emb_pca_2":       float(pca_components[1] @ emb),
    }
```

`pca_components` tính 1 lần khi train quality classifier từ toàn bộ Step 3 embeddings.

### 4.3 Crack topology features

```python
def orientation(box) -> float:
    """Góc (radian) của trục dài bounding box.
       Nếu có crop mask sẵn → dùng PCA trên pixel edges.
       Fallback: arctan2(height, width) — dùng aspect của bbox."""

def elongation(box) -> float:
    """max(w/h, h/w). Crack thường >> 1."""
```

Phase 1: dùng fallback (chỉ bbox). Phase 2 (optional): tích hợp với SAM mask từ pipeline khác.

## 5. Phased rollout

### Phase 1 — Default config (chạy được ngay, không cần label)

| Component | Default |
|---|---|
| `duplicate_classifier` | LogReg với hand-set weights (xem §6) |
| `quality_classifier` | LogReg với hand-set weights |
| `τ_dup` | 0.5 |
| `quality_min` | 0.3 |
| `keep_mode` | `{"crack": "keep_one", "spall": "wbf", "mold": "wbf"}` |

→ Step 4 chạy được full pipeline ngay, kết quả chưa tối ưu nhưng đã có data để label.

### Phase 2 — Learned config (sau khi label)

User label:
- **300-500 pair** từ output Phase 1 (`duplicate` / `not_duplicate`)
- **100-200 box** từ output Phase 1 (`good` / `bad`)

Retrain LogReg → replace default. Lưu coef vào `dedup_runs.duplicate_classifier_json` / `quality_classifier_json`.

## 6. Default classifier weights (Phase 1)

### Duplicate classifier default

Sigmoid intercept và coefficient nắm ý nghĩa rule cũ:

```python
DUPLICATE_DEFAULT_WEIGHTS = {
    "intercept":            -2.0,
    "iou":                  +4.0,
    "cos_sim":              +3.0,
    "center_sim":           +1.5,
    "size_sim":             +1.0,
    "aspect_sim":           +0.5,
    "same_class":           +0.5,
    "class_a_crack":         0.0,
    "class_b_crack":         0.0,
    "class_pair_spall_mold": -0.5,
    "class_pair_crack_other": -1.0,
    "detector_score_diff":  -0.5,
    "angle_diff":           -2.0,           # crack: góc khác → ít likely duplicate
    "elongation_diff":      -1.0,
}
```

→ `p_dup ≥ 0.5` khi (xấp xỉ) IoU ≥ 0.3 + cos_sim ≥ 0.5 + same_class.

### Quality classifier default

```python
QUALITY_DEFAULT_WEIGHTS = {
    "intercept":        -0.5,
    "semantic_prob":    +2.0,
    "detector_score":   +1.5,
    "geometry_score":   +1.0,    # U-curve theo area_ratio, peak ∈ [0.001, 0.30]
    "log_aspect":       -0.2,
    "edge_density":     +0.8,    # crack thật có edges
    "blur_score":       +0.3,
    "is_crack":          0.0,
    "is_spall":          0.0,
    "emb_pca_1":         0.0,
    "emb_pca_2":         0.0,
}
```

`geometry_score(area_ratio)`:
- `0` nếu `ratio ≤ 0`
- `0.3` nếu `< 0.001` (box quá nhỏ)
- `1.0` nếu `0.001 ≤ ratio ≤ 0.30` (sweet spot)
- `0.5` nếu `0.30 < ratio ≤ 0.70`
- `0.1` nếu `> 0.70` (box quá to)

## 7. DB Schema

```sql
CREATE TABLE dedup_runs (
    dedup_run_id              TEXT PRIMARY KEY,
    created_at_utc            TEXT NOT NULL,
    source_db_path            TEXT NOT NULL,
    embedding_db_path         TEXT NOT NULL,
    embedding_run_id          TEXT NOT NULL,
    duplicate_classifier_json TEXT NOT NULL,    -- mode + weights, hoặc "default"
    quality_classifier_json   TEXT NOT NULL,
    options_json              TEXT NOT NULL,    -- τ_dup, keeper_mode_table, quality_min, ...
    total_detections          INTEGER NOT NULL,
    kept_count                INTEGER NOT NULL,
    fused_count               INTEGER NOT NULL,
    dropped_count             INTEGER NOT NULL
);

CREATE TABLE dedup_results (
    dedup_run_id        TEXT NOT NULL,
    result_id           INTEGER NOT NULL,         -- FK -> step2.detections.result_id
    image_rel_path      TEXT NOT NULL,
    predicted_label     TEXT NOT NULL,
    keep                INTEGER NOT NULL,         -- 1/0
    fused               INTEGER NOT NULL,         -- 1 nếu là fused box mới, 0 nếu nguyên gốc
    duplicate_group_id  TEXT NOT NULL,            -- uuid của component
    representative_id   INTEGER NOT NULL,         -- result_id của keeper; -1 nếu fused
    p_dup_max           REAL NOT NULL,            -- max p_dup với neighbor bất kỳ
    p_good              REAL NOT NULL,            -- quality score
    drop_reason         TEXT NOT NULL,            -- "" | "duplicate" | "low_quality"
    fused_bbox_json     TEXT,                     -- "[x1,y1,x2,y2]" nếu fused
    PRIMARY KEY (dedup_run_id, result_id)
);

CREATE INDEX idx_dedup_image ON dedup_results (dedup_run_id, image_rel_path, keep);
CREATE INDEX idx_dedup_group ON dedup_results (dedup_run_id, duplicate_group_id);

-- Audit pair scores (để debug + UI labeling)
CREATE TABLE dedup_pair_scores (
    dedup_run_id    TEXT NOT NULL,
    result_id_a     INTEGER NOT NULL,
    result_id_b     INTEGER NOT NULL,
    p_dup           REAL NOT NULL,
    features_json   TEXT NOT NULL,                -- 13-dim pair features dạng JSON
    PRIMARY KEY (dedup_run_id, result_id_a, result_id_b)
);

CREATE INDEX idx_dedup_pairs_score ON dedup_pair_scores (dedup_run_id, p_dup);
```

`dedup_pair_scores` cần thiết cho Phase 2 labeling — UI đọc và show pair side-by-side.

## 8. CLI

```bash
python dedup_detections.py \
  --source-db ../../infer_results/semi-labeling/step2_sematic/damage_scan.sqlite3 \
  --embedding-db ../../infer_results/semi-labeling/step3_embedding/embeddings.sqlite3 \
  --output-db ../../infer_results/semi-labeling/step4_class_aware_dedup/dedup.sqlite3 \
  --semantic-run-id latest \
  --embedding-run-id latest \
  --dedup-threshold 0.5 \
  --quality-min 0.3 \
  --duplicate-classifier default \
  --quality-classifier default \
  --keeper-mode-table '{"crack":"keep_one","spall":"wbf","mold":"wbf"}' \
  --save-pair-scores true \
  --pair-prefilter-iou 0.05
```

| Flag | Default | Mô tả |
|---|---|---|
| `--dedup-threshold` | `0.5` | τ_dup |
| `--quality-min` | `0.3` | drop nếu p_good < quality_min |
| `--duplicate-classifier` | `default` | hoặc path tới JSON weights đã train |
| `--quality-classifier` | `default` | tương tự |
| `--keeper-mode-table` | `{...}` | per-class keeper mode |
| `--save-pair-scores` | `true` | ghi `dedup_pair_scores` (cần để label Phase 2) |
| `--pair-prefilter-iou` | `0.05` | bỏ pair có IoU thấp hơn để tăng tốc |
| `--image-root` | `HinhAnh/` | dùng cho `edge_density` / `blur_score` |
| `--limit-images` | `0` | 0 = full; > 0 để test trên N image đầu |

## 9. Pseudo-code chính

```python
def main(args):
    detections = read_detections(args.source_db, args.semantic_run_id)
    embeddings, idx_map = load_embeddings(args.embedding_db, embedding_run_id=args.embedding_run_id)

    dup_clf = load_classifier(args.duplicate_classifier, default=DUPLICATE_DEFAULT_WEIGHTS)
    qual_clf = load_classifier(args.quality_classifier, default=QUALITY_DEFAULT_WEIGHTS)

    ensure_schema(args.output_db)
    run_id = uuid4().hex
    write_run_metadata(args.output_db, run_id, args, dup_clf, qual_clf)

    all_decisions = []
    for image_rel_path, dets in groupby_image(detections):
        if len(dets) == 0: continue
        ids = [d.result_id for d in dets]
        embs = embeddings[[idx_map[i] for i in ids]]

        # ① + ② Candidate pairs + features
        pair_feats, pair_keys = build_pair_features(dets, embs, prefilter_iou=args.pair_prefilter_iou)

        # ③ Duplicate classifier
        p_dups = dup_clf.predict_proba(pair_feats)        # (M,) probabilities

        # Save pair scores (optional)
        if args.save_pair_scores:
            persist_pair_scores(args.output_db, run_id, pair_keys, p_dups, pair_feats)

        # ④ Union-Find cluster
        components = union_find_clusters(ids, pair_keys, p_dups, threshold=args.dedup_threshold)

        # ⑤ Quality classifier per box
        box_feats = build_box_features(dets, embs, image_rel_path, args.image_root)
        p_goods = qual_clf.predict_proba(box_feats)

        # ⑥ Representative per component
        for comp in components:
            comp_dets = [dets[ids.index(i)] for i in comp]
            comp_p_good = [p_goods[ids.index(i)] for i in comp]
            mode = args.keeper_mode_table.get(comp_dets[0].label, "keep_one")

            if mode == "keep_one" or len(comp) == 1:
                keeper_idx = int(np.argmax(comp_p_good))
                keeper = comp_dets[keeper_idx]
                for d, q in zip(comp_dets, comp_p_good):
                    all_decisions.append(make_decision(
                        d, keeper, keep=(d is keeper and q >= args.quality_min),
                        fused=False, p_good=q, p_dup_max=max_pdup_for_box(d, pair_keys, p_dups),
                    ))
            elif mode == "wbf":
                weights = [q * d.detector_score for d, q in zip(comp_dets, comp_p_good)]
                fused_bbox = weighted_box_fusion(comp_dets, weights)
                all_decisions.extend(make_wbf_decisions(comp_dets, fused_bbox, comp_p_good))

    write_decisions(args.output_db, run_id, all_decisions)
    finalize_run_counts(args.output_db, run_id)
```

## 10. Evaluation harness (`eval_dedup.py`)

Tính metrics trên holdout labeled pairs:

```python
metrics = {
    "duplicate_pair_precision":  ...,
    "duplicate_pair_recall":     ...,
    "duplicate_pair_f1":         ...,
    "auc_roc_duplicate":         ...,
    "kept_precision":            ...,
    "false_merge_rate":          ...,
    "false_drop_rate":           ...,
    "crack_recall_after_dedup":  ...,
    "crack_false_merge_rate":    ...,
    "crack_false_drop_rate":     ...,
}
```

Output:
- `eval_dedup.json` cho mỗi run
- `eval_dedup_ablation.md` tổng hợp ablation table cho thesis

### Ablation variants (cho thesis)

| Variant | Cite | Note |
|---|---|---|
| IoU-only NMS | (baseline NMS) | baseline classical |
| IoU + cosine | Soft-NMS, Bodla 2017 | thêm embedding |
| + class bias | (ours) | per-class prior |
| + geometric features | Zheng 2020 (CIoU) | center/size/aspect |
| + crack topology guard | bridge crack [Springer] | angle + elongation |
| learned classifier (Phase 2) | Tan 2019 + sklearn calibration | weights học |
| WBF vs keep-one | Solovyev 2019 (WBF) | keeper strategy |

## 11. File structure

```
DamageDetector/semi-labeling/step4_class_aware_dedup/
├── __init__.py
├── SPEC.md                       # file này
├── README.md
├── dedup_detections.py           # main entry (~300 dòng)
├── pair_features.py              # 13 pair features (~120 dòng)
├── box_features.py               # 10 box features (~100 dòng)
├── crack_topology.py             # angle + elongation (~80 dòng)
├── classifiers.py                # LogReg load/train/predict + defaults (~150 dòng)
├── nms_cluster.py                # Union-Find + WBF (~150 dòng)
├── output_store.py               # SQLite writer (~120 dòng)
├── source_store.py               # read detections từ Step 2 (~60 dòng)
└── eval_dedup.py                 # evaluation harness (~150 dòng)

infer_results/semi-labeling/step4_class_aware_dedup/
├── dedup.sqlite3
├── classifier_weights/
│   ├── duplicate_clf.json        # Phase 2 trained weights
│   └── quality_clf.json
├── labeled_pairs/
│   ├── pairs_to_label.csv        # generate từ Phase 1 output
│   └── boxes_to_label.csv
└── eval/
    ├── eval_dedup.json
    └── eval_dedup_ablation.md
```

Tổng ~1300 dòng, chia 9 module Python.

## 12. Test plan

### Unit tests

1. **Pair features**: 2 box giống nhau hoàn toàn → `iou=1, cos_sim=1, center_sim=1, size_sim=1, aspect_sim=1`.
2. **Crack topology**: 2 crack vuông góc nhau → `angle_diff ≈ π/2`.
3. **WBF**: 3 box overlap → fused bbox nằm trong convex hull của 3 box.

### Integration tests

4. **Crack overlap preservation**: 2 crack box IoU=0.5 nhưng góc khác → cả 2 phải `keep=1` (verify rule).
5. **Exact duplicate**: copy 1 box → 1 trong 2 phải `keep=0`, `drop_reason=duplicate`.
6. **Low quality drop**: box có `semantic_prob=0.1` + `area_ratio=0.95` → `keep=0`, `drop_reason=low_quality`.

### End-to-end

7. **Full run với default classifier**: tổng `kept + dropped + fused = total_detections`.
8. **Resume idempotent**: chạy 2 lần với cùng args → 2 run khác run_id nhưng kết quả giống nhau.
9. **Pair scores integrity**: Phase 1 output có `dedup_pair_scores` non-empty cho mỗi image có > 1 detection.

## 13. Workflow user

```
1. Run Step 3 (embedding cache)              [đã có spec riêng]
2. Run Step 4 Phase 1 (default classifiers)  [~5-10 phút trên CPU]
3. Inspect dedup_pair_scores DB
4. Label 300-500 pair (UI hoặc CSV)
5. Label 100-200 box quality (UI hoặc CSV)
6. Train Phase 2 classifiers (LogReg fit + save JSON weights)
7. Run Step 4 Phase 2 (learned classifiers)
8. Run eval_dedup → ablation table cho thesis
```

## 14. Section thesis tương ứng

| Section | Topic | Citation |
|---|---|---|
| 4.1 | Bài toán deduplication, vấn đề NMS | Bodla 2017, Hosang 2017 |
| 4.2 | Pair feature design | Zheng 2020 (CIoU) |
| 4.3 | Learned duplicate classifier | Tan 2019 (Learning to Rank Proposals) |
| 4.4 | Crack topology guard | bridge crack literature |
| 4.5 | Quality calibration | Küppers 2020, sklearn calibration |
| 4.6 | Keeper selection: keep-one vs WBF | Solovyev 2019 (WBF) |
| 4.7 | Evaluation + ablation | (ours) |

## 15. Out of scope

- Multi-image dedup (chỉ dedup trong cùng 1 image).
- Cross-class merging beyond class-pair bias (vd: crack ⇒ spall không bao giờ merge).
- Learnable angle estimator (Phase 1 dùng bbox aspect; Phase 2+ có thể thêm SAM mask).
- Active learning loop (cycle Phase 1 → label → Phase 2 chỉ chạy 1 lần).

## 16. Hard guards (post-filters)

Run đầu tiên với LogReg classifier monotonic theo `log_area_ratio` cho ra box phủ 99% ảnh thắng quality, nuốt tất cả box nhỏ đúng vị trí. Port lại 4 guard từ bản heuristic cũ (`_archive_step3_2026-05-22/code_step3_spatial_filter/filter_duplicates.py`):

| Guard | Vị trí | Default | Cờ CLI |
|---|---|---|---|
| Same-class hard gate | `build_pair_features` | bật | `--allow-cross-class-pairs` |
| `geometry_score` U-curve | `box_features` | thay `log_area_ratio` | — |
| Oversized auto-drop | post-filter ⑦ | `area_ratio ≥ 0.70` | `--oversized-area-ratio` |
| Multi-feature container suppression | post-filter ⑦ | `area_ratio ≥ 0.30` + ≥ 2 keeper con cùng class, mỗi cái nằm ≥ 85% trong parent | `--container-area-ratio`, `--container-overlap-threshold`, `--container-min-children` |

Drop reason mới trong `dedup_results.drop_reason`: `"oversized"`, `"multi_feature_container"`.

Tắt toàn bộ post-filter bằng `--disable-post-filters true` (chỉ dùng cho ablation thesis).

## 17. References

- Bodla et al., "Soft-NMS — Improving Object Detection with One Line of Code", ICCV 2017.
- Solovyev et al., "Weighted Boxes Fusion: Ensembling boxes from different object detection models", arXiv 2019.
- Hosang et al., "Learning Non-Maximum Suppression", arXiv 2017.
- Tan et al., "Learning to Rank Proposals for Object Detection", ICCV 2019.
- Zheng et al., "Enhancing Geometric Factors in Model Learning and Inference for Object Detection (CIoU / Cluster-NMS)", arXiv 2020.
- Küppers et al., "Multivariate Confidence Calibration for Object Detection", CVPRW 2020.
- scikit-learn, "Probability calibration", official documentation.
- Bridge crack detection (DOAJ), Springer "Automated bridge crack detection method based on lightweight vision models".
