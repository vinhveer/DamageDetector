# sam_gdino pipeline

End-to-end pipeline: detect bridge → crop → damage detect → semantic label → embed → segment.

Steps:
1. `step1_gdino_bridge` — GroundingDINO detect "bridge" với dynamic threshold (floor + top-k).
2. `step2_sam_bridge_crop` — Lấy điểm trong box → SAM point-prompt → mask → crop ảnh cầu.
3. `step3_gdino_damage` — Multi-prompt damage detect trên ảnh đã crop (copy từ semi-labeling/step1).
4. `step4_openclip_semantic` — Gán top-1 semantic label cho mỗi box (copy từ semi-labeling/step2).
5. `step5_embedding` — Embed detections (copy từ semi-labeling/step3).
6. `step6_route_segment` — Routing: top-1 label ∈ CRACK_LABELS → SAM-LoRA, ngược lại → SAM zero-shot.

Output mặc định: `/Users/nguyenquangvinh/Desktop/Lab/infer_results/pineline/sam_gdino/`.

## Step 1: GDINO bridge

```bash
cd /Users/nguyenquangvinh/Desktop/Lab/DamageDetector
python -m pineline.sam_gdino.step1_gdino_bridge.run \
  --input-dir /Users/nguyenquangvinh/Desktop/Lab/HinhAnh \
  --score-floor 0.20 \
  --top-k 3
```

Dynamic threshold:
- `--score-floor`: ngưỡng tối thiểu (default 0.20). Box nào dưới ngưỡng này bỏ. Ảnh nào không còn box nào ≥ ngưỡng → bỏ ảnh.
- `--top-k`: giữ tối đa N box mỗi ảnh sau khi sort theo score giảm dần (default 3).
- `--box-threshold`: ngưỡng GDINO raw (default 0.10 — thấp để có pool candidates đủ rộng cho post-filter).

## Step 2: SAM bridge crop (sẽ làm sau)
## Step 3-5: Copy từ semi-labeling (sẽ làm sau)
## Step 6: Route + Segment (sẽ làm sau)
