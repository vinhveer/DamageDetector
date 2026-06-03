# Spec: Xử lý "spall ít" ở semi-labeling (run_id=myrun)

> Trạng thái: **ĐỀ XUẤT — chưa thực thi.** Viết để duyệt trước khi đổi pipeline.
> Ngày: 2026-06-03. DB: `model_with_inference/semi_labeling/resemi.sqlite3`.

## 0. Kết luận điều tra (đã xác minh)

**Step09 KHÔNG bị interrupt.** Cả 2 run đều xử lý trọn 78,328 candidate
(`promote + defer = candidate`, `rejected = 0`):

| run | conf thr | candidate | promote | defer |
|---|---|---|---|---|
| `selftrain_myrun_clf_myrun_em_r1` | 0.90 | 78,328 | 1,816 | 76,512 |
| `selftrain_myrun_conf075_audit` | 0.75 | 78,328 | 1,945 | 76,383 |

**Spall ở step09 không hề ít** — promote spall=585, đứng **thứ 2** sau crack (783),
trên cả mold (448); tỉ lệ promote spall 2.6% > mold 1.5%.

Spall ít là hiện tượng **toàn cục, có gốc từ step01**, không phải lỗi step09:

| nhãn | step01 semantic | review_queue | cleaned_labels |
|---|---|---|---|
| mold | 51,475 | 53,732 | 15,880 |
| crack | 47,926 | 39,791 | 11,282 |
| **spall** | **11,217** | **9,660** | **2,040** |

→ Hai vấn đề tách biệt: **(A)** tổng promote step09 quá thấp (2.3%) vì cổng quá chặt;
**(B)** spall hiếm từ đầu nguồn (step01).

---

## Phần A — Nới cổng step09 (để promote nhiều hơn, gồm spall)

### Nguyên nhân tổng promote thấp
Blocker trên 76,512 item bị defer (run r1):

| blocker | #item | nguồn |
|---|---|---|
| `missing_core_or_prototype_agreement` | 50,082 | `self_training.py:291` — đòi prototype HOẶC core đồng ý với classifier, sim ≥ 0.70 |
| `geometry_conflict` | 49,733 | `self_training.py:293` — `suspect_composite_box` (44,460 box, keep_for_cleaned=0) |
| `near_reject_prototype` | 16,857 | `self_training.py:295` |
| `classifier_confidence_low` | 13,092 | conf < 0.90 |

### Mô phỏng tác động (đếm lại từ promotions hiện có, không chạy lại)

| Thay đổi | total promote | spall promote |
|---|---|---|
| Baseline (hiện tại) | 1,816 | 585 |
| Bỏ chặn `geometry_conflict` | 13,073 | 3,729 |
| Bỏ chặn `missing_core_or_prototype` | 14,324 | 4,473 |
| Bỏ cả hai | 50,172 | 16,406 |
| Bỏ cả 3 (+ near_reject) | 65,236 | 17,388 |

### Đề xuất A (an toàn, gồm spall tăng ~6×)
Chạy lại step09 với cổng nới vừa phải — **giữ `geometry_conflict` (vì composite box
thật sự là box xấu), nhưng hạ ngưỡng prototype/core** để bớt `missing_agreement`:

```
python -m steps.step09_self_train.main \
  --db model_with_inference/semi_labeling/resemi.sqlite3 \
  --run-id myrun --classifier-run-id clf_myrun_emb_c6ec_28c706fb \
  --self-training-run-id selftrain_myrun_loosened_audit --round-index 3 \
  --classifier-confidence-threshold 0.85 \
  --prototype-min-similarity 0.60 --core-min-similarity 0.60
  # KHÔNG --apply-promotions ở lần đầu (audit-only)
```

Đầu tiên audit, xem số liệu; nếu ổn thì thêm `--apply-promotions`.

> ⚠️ `geometry_conflict` là cổng cứng trong code (`self_training.py:293`), KHÔNG có flag CLI.
> Nếu muốn bỏ nó cần sửa code (thêm flag `--ignore-composite-box`) — hỏi trước.

---

## Phần B — Tăng spall từ đầu nguồn (step01 semantic)

### Cơ chế gán nhãn
Step01 gán `final_label` theo prototype similarity, chia bằng `--accept-threshold 0.75`
/ `--suspect-threshold 0.50` / margin. Spall ra ít vì **bản thân spall prototype ít/yếu**
ở step05, hoặc spall hiếm trong source detections.

### Đề xuất B (cần xác minh thêm trước khi làm)
1. Kiểm tra số spall prototype ở `prototype_items` của version mới nhất.
2. Nếu spall prototype < crack/mold đáng kể → bổ sung spall prototype ở step05 (tab
   Prototype của app, hoặc step05 với nhiều spall pick hơn) rồi chạy lại step01→07.
3. KHÔNG hạ threshold toàn cục (sẽ làm noise mold/crack tăng theo).

> Phần B đụng nhiều bước (01→07), tốn thời gian. Chỉ làm nếu mục tiêu là tăng spall
> thật trong tập train cuối, không chỉ ở step09.

---

## Phần C — "Không đổi pipeline" (mặc định nếu chưa duyệt A/B)
Giữ nguyên DB. Quay lại hoàn thiện app review. Việc đã xong ở app: overlay tất cả box
trên ảnh (tab Review). Spall ít trong app chỉ phản ánh đúng dữ liệu, không phải bug app.

---

## Câu hỏi cần bạn quyết
1. Phần A: chạy audit-only với cổng nới (0.85/0.60/0.60) trước? Có cho phép tôi thêm
   flag `--ignore-composite-box` vào code step09 không?
2. Phần B: có muốn tôi xác minh số spall prototype (chỉ đọc DB, không sửa) để xem có
   đáng bổ sung không?
3. Sau khi promote, có `--apply-promotions` để đổ vào cleaned_labels rồi xem trong app?
