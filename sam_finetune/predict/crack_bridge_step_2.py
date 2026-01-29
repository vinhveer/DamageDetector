# MỤC ĐÍCH: chạy lần 2 để xác định hư hỏng trên cầu (sử dụng hình đã tách ở step 1)
# chạy ở chế độ X để chọn ROI, sau đó chọn C để chạy GroundDINO.
import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# ====== CẤU HÌNH ======
sam_checkpoint = r"sam_vit_b_01ec64.pth"
model_type = "vit_b"
image_path = r"D:\Reseach\19_NCS\SAM GroundDINO\pics\GDINO_SAM_TranPhuBridge\TranPhuBridge_isolate_bridge_ALL.png"

# Thư mục lưu mặc định: GDINO_SAM_<tên ảnh> cạnh file ảnh
IMG_DIR = os.path.dirname(image_path)
IMG_NAME = os.path.splitext(os.path.basename(image_path))[0]  # "columnCrack"
DEFAULT_SAVE_DIR = os.path.join(IMG_DIR, f"GDINO_SAM_{IMG_NAME}")

# GroundingDINO (Tiny cho CPU/GPU trung bình; có thể đổi sang 'IDEA-Research/grounding-dino-base')
# GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Danh sách prompt mặc định
TEXT_QUERIES = [
    # "bridge","crack", "mold", "water"
    # "bridge", "bridge beam","bridge girder", "beam", "girder"
    "crack", "mold", "stain", "spall", "damage", "corrosion", "rust"
]


# Ngưỡng cho GroundingDINO
BOX_THRESHOLD = 0.1     # lọc box yếu
TEXT_THRESHOLD = 0.1    # lọc logit văn bản yếu

# ====== LOAD ẢNH ======
bgr = cv2.imread(image_path)
if bgr is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H_full, W_full = img.shape[:2]

# ====== THIẾT BỊ ======
device = "cpu"
print("Device:", device)

# ====== LOAD SAM ======
if not os.path.isfile(sam_checkpoint):
    raise FileNotFoundError(f"Không tìm thấy checkpoint: {sam_checkpoint}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(img)

# ====== SAM AUTO GENERATOR ======
def make_mask_generator(profile="QUALITY"):
    up = profile.upper()
    if up == "QUALITY":
        return SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=24,
            points_per_batch=64,
            pred_iou_thresh=0.92,
            stability_score_thresh=0.92,
            stability_score_offset=0.95,
            box_nms_thresh=0.7,
            crop_n_layers=1,
            crop_overlap_ratio=0.5,
            crop_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=80,
            output_mode="binary_mask",
        )
    elif up == "ULTRA":
        return SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            points_per_batch=128,
            pred_iou_thresh=0.94,
            stability_score_thresh=0.94,
            stability_score_offset=0.95,
            box_nms_thresh=0.7,
            crop_n_layers=2,
            crop_overlap_ratio=0.55,
            crop_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=40,
            output_mode="binary_mask",
        )
    elif up == "FINE":  # độ mịn cao để bắt vết nứt nhỏ
        return SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=48,          # dày hơn
            points_per_batch=128,
            pred_iou_thresh=0.88,        # nới nhẹ để không bỏ lỡ
            stability_score_thresh=0.90,
            stability_score_offset=0.95,
            box_nms_thresh=0.7,
            crop_n_layers=3,             # cắt nhiều lớp hơn
            crop_overlap_ratio=0.6,
            crop_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=10,     # cho phép vùng rất nhỏ
            output_mode="binary_mask",
        )
    else:
        raise ValueError("Unknown profile")

current_profile = "FINE"
mask_generator = make_mask_generator(current_profile)

def print_current_params():
    mg = mask_generator
    attrs = {
        "profile": current_profile,
        "points_per_side": mg.points_per_side,
        "points_per_batch": mg.points_per_batch,
        "pred_iou_thresh": mg.pred_iou_thresh,
        "stability_score_thresh": mg.stability_score_thresh,
        "stability_score_offset": mg.stability_score_offset,
        "box_nms_thresh": mg.box_nms_thresh,
        "crop_n_layers": mg.crop_n_layers,
        "crop_overlap_ratio": mg.crop_overlap_ratio,
        "crop_nms_thresh": mg.crop_nms_thresh,
        "crop_n_points_downscale_factor": mg.crop_n_points_downscale_factor,
        "min_mask_region_area": mg.min_mask_region_area,
        "output_mode": mg.output_mode,
    }
    print("🔧 SAM params:", attrs)

# ====== TRẠNG THÁI UI ======
points, labels = [], []
overlay = bgr.copy()
mask_best = None
score_best = None
all_masks = []   # danh sách mask auto (full ảnh hoặc ROI)
WIN_NAME = "SAM (L=FG, R=BG | Enter=Run | A=Auto | M=Multi-mask | T=Text | C/B=ROI | X=ROI mode | H=Fine | R=Reset | S=Save | Q/U profile | Esc=Quit)"

# ROI interaction
ROI_MODE = False
roi_start = None
roi_end = None
roi_rect = None  # (x1,y1,x2,y2) đã chốt

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def get_rect_xyxy(p1, p2):
    x1,y1 = p1; x2,y2 = p2
    x1,x2 = sorted([clamp(x1,0,W_full-1), clamp(x2,0,W_full-1)])
    y1,y2 = sorted([clamp(y1,0,H_full-1), clamp(y2,0,H_full-1)])
    return x1,y1,x2,y2

def draw_roi(display):
    if roi_start is not None and roi_end is not None:
        x1,y1,x2,y2 = get_rect_xyxy(roi_start, roi_end)
        cv2.rectangle(display, (x1,y1), (x2,y2), (255, 180, 60), 2)
        cv2.putText(display, f"ROI: {x1},{y1} -> {x2},{y2}",
                    (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,180,60), 2)

def redraw():
    global overlay
    display = bgr.copy()
    if mask_best is not None:
        alpha = 0.45
        color = np.array([255, 0, 255], dtype=np.uint8)
        m = mask_best.astype(bool)
        display[m] = (alpha * color + (1 - alpha) * display[m]).astype(np.uint8)
        if score_best is not None:
            cv2.putText(display, f"Score: {score_best:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 200, 50), 3)
    for (x, y), lb in zip(points, labels):
        if lb == 1:
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        else:
            cv2.drawMarker(display, (x, y), (0, 0, 255),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
    draw_roi(display)
    if ROI_MODE:
        cv2.putText(display, "ROI MODE: drag to select, press C/B to run in ROI",
                    (10, H_full-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,220,255), 2)
    overlay[:] = display

def run_sam_interactive():
    global mask_best, score_best
    if len(points) == 0:
        print("⚠️ Chưa có điểm. Click chuột để thêm.")
        return
    pt = np.array(points, dtype=np.float32)
    lb = np.array(labels, dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=pt,
        point_labels=lb,
        multimask_output=True,
    )
    idx = int(np.argmax(scores))
    mask_best = masks[idx]
    score_best = float(scores[idx])
    print(f"✅ SAM Interactive: best score={score_best:.3f}, candidates={len(scores)}")

def run_sam_auto():
    global mask_best, score_best
    print("🚀 SAM Auto (1 mask lớn nhất)...")
    masks = mask_generator.generate(img)
    if len(masks) == 0:
        print("⚠️ Không tìm thấy mask nào.")
        return
    largest = max(masks, key=lambda x: x['area'])
    mask_best = largest['segmentation']
    score_best = largest.get('stability_score', None)
    print(f"✅ SAM Auto: lấy mask lớn nhất, area={largest['area']}")

def run_sam_auto_all(save_suffix=""):
    global all_masks
    print("🚀 SAM Auto (nhiều mask, full ảnh)...")
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
    all_masks = mask_generator.generate(img)
    if len(all_masks) == 0:
        print("⚠️ Không tìm thấy mask nào.")
        return

    display = bgr.copy()
    for m in all_masks:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        alpha = 0.5
        region = m["segmentation"].astype(bool)
        display[region] = (alpha * color + (1 - alpha) * display[region]).astype(np.uint8)

    base = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(DEFAULT_SAVE_DIR, f"{base}_all_masks_overlay{save_suffix}.png")
    cv2.imshow("All Masks", display)
    cv2.imwrite(overlay_path, display)
    print(f"✅ Đã lưu {overlay_path} với {len(all_masks)} mask")

    for i, m in enumerate(all_masks):
        mask = (m["segmentation"].astype(np.uint8) * 255)
        out_path = os.path.join(DEFAULT_SAVE_DIR, f"{base}_mask_{i+1:02d}{save_suffix}.png")
        cv2.imwrite(out_path, mask)
    print(f"💾 Đã lưu {len(all_masks)} mask nhị phân vào {DEFAULT_SAVE_DIR}")

def run_sam_auto_all_roi(x1,y1,x2,y2):
    """Chạy auto masks trong ROI (độ mịn theo profile hiện tại), ghép về khung full."""
    global all_masks, mask_best, score_best, overlay
    print(f"🚀 SAM Auto (ROI {x1},{y1}→{x2},{y2})...")
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

    roi_rgb = img[y1:y2, x1:x2].copy()
    roi_bgr = bgr[y1:y2, x1:x2].copy()
    masks_roi = mask_generator.generate(roi_rgb)

    if len(masks_roi) == 0:
        print("⚠️ ROI: Không tìm thấy mask nào.")
        return

    # vẽ overlay lên ROI rồi ghép vào ảnh lớn
    disp_full = bgr.copy()
    disp_roi = roi_bgr.copy()
    for m in masks_roi:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        alpha = 0.5
        region = m["segmentation"].astype(bool)
        disp_roi[region] = (alpha * color + (1 - alpha) * disp_roi[region]).astype(np.uint8)
    disp_full[y1:y2, x1:x2] = disp_roi
    overlay[:] = disp_full  # cập nhật cửa sổ chính

    base = os.path.splitext(os.path.basename(image_path))[0]
    suffix = f"_ROI_{x1}_{y1}_{x2}_{y2}"
    overlay_path = os.path.join(DEFAULT_SAVE_DIR, f"{base}_all_masks_overlay{suffix}.png")
    cv2.imwrite(overlay_path, disp_full)
    print(f"✅ Đã lưu {overlay_path} với {len(masks_roi)} mask (ROI)")

    # lưu mask riêng theo hệ quy chiếu full ảnh (dịch ROI về full)
    for i, m in enumerate(masks_roi):
        mask_roi = (m["segmentation"].astype(np.uint8) * 255)
        # dán vào khung full
        mask_full = np.zeros((H_full, W_full), dtype=np.uint8)
        mask_full[y1:y2, x1:x2] = mask_roi
        out_path = os.path.join(DEFAULT_SAVE_DIR, f"{base}_mask_{i+1:02d}{suffix}.png")
        cv2.imwrite(out_path, mask_full)

    # cập nhật mask_best tạm thời bằng mask lớn nhất trong ROI
    largest = max(masks_roi, key=lambda x: x['area'])
    mask_best = np.zeros((H_full, W_full), dtype=bool)
    mask_best[y1:y2, x1:x2] = largest['segmentation'].astype(bool)
    score_best = largest.get("stability_score", None)
    print(f"💾 Đã lưu {len(masks_roi)} mask nhị phân ROI vào {DEFAULT_SAVE_DIR}")

def reset_points():
    global points, labels, mask_best, score_best, all_masks
    points.clear(); labels.clear()
    mask_best = None; score_best = None
    all_masks = []
    print("🔄 Reset points & masks.")

def in_roi(x,y,x1,y1,x2,y2):
    return (x1 <= x < x2) and (y1 <= y < y2)

def mouse_cb(event, x, y, flags, param):
    global mask_best, score_best, roi_start, roi_end, roi_rect
    if ROI_MODE:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_start = (x, y)
            roi_end = (x, y)
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and roi_start is not None:
            roi_end = (x, y)
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and roi_start is not None:
            roi_end = (x, y)
            x1,y1,x2,y2 = get_rect_xyxy(roi_start, roi_end)
            if x2-x1 >= 5 and y2-y1 >= 5:
                roi_rect = (x1,y1,x2,y2)
                print(f"📦 ROI set: {roi_rect}")
            else:
                roi_rect = None
                print("⚠️ ROI quá nhỏ, bỏ qua.")
            redraw()
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(all_masks) > 0:
                # nếu đã có all_masks (full), cho phép chọn mask theo click
                # lưu ý: với ROI masks, đã dán về full nên cũng chọn được
                for m in all_masks:
                    seg = m.get("segmentation", None)
                    if seg is None:
                        continue
                    if seg.shape != (H_full, W_full):
                        continue
                    if seg[y, x]:
                        mask_best = seg
                        score_best = m.get("stability_score", None)
                        print(f"🎯 Chọn mask (click) với stability={score_best}")
                        redraw()
                        return
            else:
                points.append([x, y]); labels.append(1)
                print(f"➕ FG: ({x},{y})"); redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y]); labels.append(0)
            print(f"➕ BG: ({x},{y})"); redraw()

# ====== GROUNDINGDINO: TEXT -> BOXES ======
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image

processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
gdino = GroundingDinoForObjectDetection.from_pretrained(GDINO_MODEL_ID)
gdino.to(device)
gdino.eval()

def run_text_boxes(pil_image, text_queries, box_threshold=0.25, text_threshold=0.25):
    """
    Trả về list dict: { 'label': str, 'box_xyxy': np.ndarray(float, shape=(4,)), 'score': float }
    Hỗ trợ transformers 4.39–4.42 với API post_process_grounded_object_detection.
    """
    results_all = []
    W, H = pil_image.size
    caption = ". ".join(text_queries)

    with torch.no_grad():
        inputs = processor(images=pil_image, text=caption, return_tensors="pt").to(device)
        outputs = gdino(**inputs)
        target_sizes = torch.tensor([[H, W]], device=device)
        processed = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes
        )
    if len(processed) == 0:
        return results_all
    p = processed[0]
    boxes = p["boxes"].detach().cpu().numpy()
    scores = p["scores"].detach().cpu().numpy()
    labels = p["labels"]
    for b, s, lab in zip(boxes, scores, labels):
        results_all.append({"label": lab, "box_xyxy": b.astype(np.float32), "score": float(s)})
    return results_all

# ====== TEXT->BOX->SAM (FULL) ======
def run_text_then_sam(text_queries=None, save_dir=DEFAULT_SAVE_DIR):
    global mask_best, score_best, overlay
    if text_queries is None:
        text_queries = TEXT_QUERIES
    os.makedirs(save_dir, exist_ok=True)
    pil_img = Image.fromarray(img)

    print(f"📝 Text prompts: {text_queries}")
    dets = run_text_boxes(pil_img, text_queries, BOX_THRESHOLD, TEXT_THRESHOLD)
    if len(dets) == 0:
        print("⚠️ GroundingDINO không tìm thấy đối tượng phù hợp.")
        return

    disp = bgr.copy()
    base = os.path.splitext(os.path.basename(image_path))[0]
    per_label_count = {}

    for det in dets:
        x1, y1, x2, y2 = det["box_xyxy"]
        label = det["label"]; score = det["score"]

        cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
        cv2.putText(disp, f"{label} {score:.2f}", (int(x1), int(max(0, y1-5))),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 200, 255), 3)

        input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        idx = int(np.argmax(scores))
        chosen_mask = masks[idx].astype(np.uint8)

        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        mbool = chosen_mask.astype(bool)
        disp[mbool] = (0.45 * color + 0.55 * disp[mbool]).astype(np.uint8)

        per_label_count[label] = per_label_count.get(label, 0) + 1
        out_mask = (chosen_mask * 255)
        out_path = os.path.join(save_dir, f"{base}_{label}_{per_label_count[label]:02d}.png")
        cv2.imwrite(out_path, out_mask)

        mask_best = chosen_mask
        score_best = float(np.max(scores))

    out_overlay = os.path.join(save_dir, f"{base}_gdino_sam_overlay.png")
    cv2.imwrite(out_overlay, disp)
    overlay[:] = disp
    print(f"✅ Text→Box→SAM xong. Đã lưu overlay: {out_overlay}")
    print("📦 Lưu từng mask theo nhãn trong thư mục:", save_dir)

# ====== TEXT->BOX->SAM (ROI) ======
def run_text_then_sam_roi(x1,y1,x2,y2, text_queries=None, save_dir=DEFAULT_SAVE_DIR):
    """
    1) GroundingDINO trong ROI (ảnh cắt)
    2) SAM từ box (tọa độ full = box_roi + offset)
    3) Lưu overlay & mask (trả về full-size)
    """
    global mask_best, score_best, overlay
    if text_queries is None:
        text_queries = TEXT_QUERIES
    os.makedirs(save_dir, exist_ok=True)

    roi_rgb = img[y1:y2, x1:x2].copy()
    roi_bgr = bgr[y1:y2, x1:x2].copy()
    pil_roi = Image.fromarray(roi_rgb)
    dets = run_text_boxes(pil_roi, text_queries, BOX_THRESHOLD, TEXT_THRESHOLD)
    if len(dets) == 0:
        print("⚠️ ROI: GDINO không tìm thấy đối tượng.")
        return

    disp_full = bgr.copy()
    base = os.path.splitext(os.path.basename(image_path))[0]
    per_label_count = {}
    suffix = f"_ROI_{x1}_{y1}_{x2}_{y2}"

    for det in dets:
        rx1, ry1, rx2, ry2 = det["box_xyxy"]
        label = det["label"]; score = det["score"]

        # chuyển box ROI -> box full
        x1f = rx1 + x1; y1f = ry1 + y1; x2f = rx2 + x1; y2f = ry2 + y1
        cv2.rectangle(disp_full, (int(x1f), int(y1f)), (int(x2f), int(y2f)), (0, 200, 255), 2)
        cv2.putText(disp_full, f"{label} {score:.2f}", (int(x1f), int(max(0, y1f-5))),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 255), 2)

        input_box = np.array([[x1f, y1f, x2f, y2f]], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        idx = int(np.argmax(scores))
        chosen_mask = masks[idx].astype(np.uint8)

        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        mbool = chosen_mask.astype(bool)
        disp_full[mbool] = (0.45 * color + 0.55 * disp_full[mbool]).astype(np.uint8)

        per_label_count[label] = per_label_count.get(label, 0) + 1
        out_mask = (chosen_mask * 255)
        out_path = os.path.join(save_dir, f"{base}_{label}_{per_label_count[label]:02d}{suffix}.png")
        cv2.imwrite(out_path, out_mask)

        mask_best = chosen_mask
        score_best = float(np.max(scores))

    out_overlay = os.path.join(save_dir, f"{base}_gdino_sam_overlay{suffix}.png")
    cv2.imwrite(out_overlay, disp_full)
    overlay[:] = disp_full
    print(f"✅ ROI Text→Box→SAM xong. Đã lưu overlay: {out_overlay}")

# ====== HƯỚNG DẪN UI ======
os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
redraw()
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN_NAME, mouse_cb)

print("👉 Hướng dẫn:")
print(" - Chuột trái: Foreground (xanh) hoặc chọn mask (nếu đã có all-masks)")
print(" - Chuột phải: Background (đỏ)")
print(" - Enter: SAM Interactive (click)")
print(" - A: SAM Auto (mask lớn nhất toàn ảnh)")
print(" - M: SAM Auto (nhiều mask toàn ảnh)")
print(" - T: Text→Box→SAM toàn ảnh (GroundingDINO + SAM)")
print(" - X: bật/tắt ROI mode (kéo thả để vẽ vùng)")
print(" - C: Text→Box→SAM trong ROI đã vẽ")
print(" - B: SAM Auto (nhiều mask) trong ROI đã vẽ")
print(" - H: chuyển profile FINE (độ mịn cao)")
print(" - Q: QUALITY profile | U: ULTRA profile | P: in thông số hiện tại")
print(" - R: reset points/masks (không xóa ROI)")
print(" - S: lưu mask hiện tại + overlay hiện tại vào GDINO_SAM_<ten_anh>")
print(" - Esc: thoát")

while True:
    cv2.imshow(WIN_NAME, overlay)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:   # Esc
        break
    elif k == 13:  # Enter
        run_sam_interactive(); redraw()
    elif k in (ord('a'), ord('A')):
        run_sam_auto(); redraw()
    elif k in (ord('m'), ord('M')):
        run_sam_auto_all(); redraw()
    elif k in (ord('t'), ord('T')):
        run_text_then_sam(TEXT_QUERIES); redraw()
    elif k in (ord('x'), ord('X')):
        ROI_MODE = not ROI_MODE
        if ROI_MODE:
            print("✅ ROI MODE ON: kéo chuột trái để vẽ vùng; nhấn C hoặc B để chạy trong ROI.")
        else:
            print("✅ ROI MODE OFF")
        redraw()
    elif k in (ord('c'), ord('C')):
        if roi_start is None or roi_end is None:
            print("⚠️ Chưa vẽ ROI (bật X rồi kéo thả).")
        else:
            x1,y1,x2,y2 = get_rect_xyxy(roi_start, roi_end)
            if x2-x1 < 5 or y2-y1 < 5:
                print("⚠️ ROI quá nhỏ.")
            else:
                run_text_then_sam_roi(x1,y1,x2,y2, TEXT_QUERIES); redraw()
    elif k in (ord('b'), ord('B')):
        if roi_start is None or roi_end is None:
            print("⚠️ Chưa vẽ ROI (bật X rồi kéo thả).")
        else:
            x1,y1,x2,y2 = get_rect_xyxy(roi_start, roi_end)
            if x2-x1 < 5 or y2-y1 < 5:
                print("⚠️ ROI quá nhỏ.")
            else:
                run_sam_auto_all_roi(x1,y1,x2,y2); redraw()
    elif k in (ord('h'), ord('H')):
        current_profile = "FINE"
        mask_generator = make_mask_generator(current_profile)
        print("✅ Switched to FINE (high-detail) profile")
        print_current_params()
    elif k in (ord('q'), ord('Q')):
        current_profile = "QUALITY"
        mask_generator = make_mask_generator(current_profile)
        print("✅ Switched to QUALITY profile")
        print_current_params()
    elif k in (ord('u'), ord('U')):
        current_profile = "ULTRA"
        mask_generator = make_mask_generator(current_profile)
        print("✅ Switched to ULTRA profile")
        print_current_params()
    elif k in (ord('r'), ord('R')):
        reset_points(); redraw()
    elif k in (ord('s'), ord('S')):
        if mask_best is not None:
            os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(DEFAULT_SAVE_DIR, f"{base}_mask.png")
            overlay_path = os.path.join(DEFAULT_SAVE_DIR, f"{base}_overlay.png")
            cv2.imwrite(mask_path, (mask_best.astype(np.uint8) * 255))
            cv2.imwrite(overlay_path, overlay)
            print(f"💾 Saved: {mask_path} + {overlay_path}")
        else:
            print("⚠️ Chưa có mask nào để lưu.")
    elif k in (ord('p'), ord('P')):
        print_current_params()

cv2.destroyAllWindows()
