# MỤC ĐÍCH: chạy lần 1 để tách riêng cầu ra khỏi nền. Chạy ở chế độ I (isolation đối tượng bridge beam).

import os
import cv2
import numpy as np
from torch_runtime import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# ====== CẤU HÌNH ======
sam_checkpoint = r"sam_vit_b_01ec64.pth"
model_type = "vit_b"
image_path = r"D:\Reseach\19_NCS\SAM GroundDINO\pics\TranPhuBridge.JPG"

# Thư mục lưu mặc định: GDINO_SAM_<tên ảnh> cạnh file ảnh
IMG_DIR = os.path.dirname(image_path)
IMG_NAME = os.path.splitext(os.path.basename(image_path))[0]  # => "TranPhuBridge"
DEFAULT_SAVE_DIR = os.path.join(IMG_DIR, f"GDINO_SAM_{IMG_NAME}")

# GroundingDINO (Tiny cho CPU/GPU trung bình; có thể đổi sang 'IDEA-Research/grounding-dino-base')
# GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Danh sách prompt mặc định (dùng để detect với GDINO)
TEXT_QUERIES = [
    "bridge", "bridge beam","bridge girder", "beam", "girder"
]

# Ngưỡng cho GroundingDINO
BOX_THRESHOLD = 0.25     # lọc box yếu
TEXT_THRESHOLD = 0.25    # lọc logit văn bản yếu

# === ISOLATE BY LABEL (tách vật thể theo nhãn GroundingDINO) ===
# Ví dụ muốn tách riêng "bridge beam". Bạn có thể ghi ["beam"] hay ["girder"] tùy ảnh.
TARGET_LABELS = ["bridge beam"]
OUTSIDE_VALUE_ISOLATE = 0     # 0 = đen phần ngoài mask; 255 = trắng
CROP_TO_BBOX_ISOLATE = False  # True để cắt khít theo bbox của union mask

# ====== LOAD ẢNH ======
bgr = cv2.imread(image_path)
if bgr is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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
    if profile.upper() == "QUALITY":
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
    elif profile.upper() == "ULTRA":
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
    else:
        raise ValueError("Unknown profile")

# current_profile = "QUALITY"
current_profile = "ULTRA"
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
all_masks = []   # danh sách mask auto
WIN_NAME = "SAM (L=FG, R=BG | Enter=Run | A=Auto | M=Multi-mask | T=Text | I=Isolate | R=Reset | S=Save | Q/U profile | Esc=Quit)"

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

def run_sam_auto_all():
    global all_masks
    print("🚀 SAM Auto (nhiều mask)...")
    all_masks = mask_generator.generate(img)
    if len(all_masks) == 0:
        print("⚠️ Không tìm thấy mask nào.")
        return

    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)  # đảm bảo thư mục tồn tại

    display = bgr.copy()
    for m in all_masks:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        alpha = 0.5
        region = m["segmentation"].astype(bool)
        display[region] = (alpha * color + (1 - alpha) * display[region]).astype(np.uint8)
    cv2.imshow("All Masks", display)

    # lưu overlay tổng hợp
    base = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(DEFAULT_SAVE_DIR, f"{base}_all_masks_overlay.png")
    cv2.imwrite(overlay_path, display)
    print(f"✅ Đã lưu {overlay_path} với {len(all_masks)} mask")

    # lưu từng mask nhị phân
    for i, m in enumerate(all_masks):
        mask = (m["segmentation"].astype(np.uint8) * 255)
        out_path = os.path.join(DEFAULT_SAVE_DIR, f"{base}_mask_{i+1:02d}.png")
        cv2.imwrite(out_path, mask)
    print(f"💾 Đã lưu {len(all_masks)} mask nhị phân vào {DEFAULT_SAVE_DIR}")

def reset_points():
    global points, labels, mask_best, score_best, all_masks
    points.clear(); labels.clear()
    mask_best = None; score_best = None
    all_masks = []
    print("🔄 Reset points & masks.")

def mouse_cb(event, x, y, flags, param):
    global mask_best, score_best
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(all_masks) > 0:
            for m in all_masks:
                if m["segmentation"][y, x]:
                    mask_best = m["segmentation"]
                    score_best = m.get("stability_score", None)
                    print(f"🎯 Chọn mask area={m['area']}, stability={score_best}")
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
    Hỗ trợ các phiên bản transformers 4.39–4.42 với API post_process_grounded_object_detection.
    """
    results_all = []
    W, H = pil_image.size

    # Gom các query thành 1 câu, GroundingDINO sẽ tự matching
    caption = ". ".join(text_queries)

    with torch.no_grad():
        inputs = processor(images=pil_image, text=caption, return_tensors="pt").to(device)
        outputs = gdino(**inputs)

        # API đúng: post_process_grounded_object_detection(outputs, input_ids, box_threshold, text_threshold, target_sizes)
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
    boxes = p["boxes"].detach().cpu().numpy()     # (N, 4) xyxy theo kích thước ảnh
    scores = p["scores"].detach().cpu().numpy()   # (N,)
    labels = p["labels"]                           # list[str]

    for b, s, lab in zip(boxes, scores, labels):
        results_all.append({
            "label": lab,
            "box_xyxy": b.astype(np.float32),
            "score": float(s)
        })
    return results_all

# ====== TEXT->BOX->SAM ======
def run_text_then_sam(text_queries=None, save_dir=DEFAULT_SAVE_DIR):
    """
    1) GroundingDINO tìm bbox theo text
    2) Dùng SAM cắt mask từ bbox (chuẩn SAM: xyxy theo ảnh gốc)
    3) Lưu overlay & từng mask theo nhãn
    """
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

    # Vẽ bbox lên ảnh hiển thị, đồng thời gọi SAM theo từng box
    disp = bgr.copy()
    base = os.path.splitext(os.path.basename(image_path))[0]

    per_label_count = {}

    for i, det in enumerate(dets, 1):
        x1, y1, x2, y2 = det["box_xyxy"]
        label = det["label"]
        score = det["score"]

        # Vẽ box
        cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
        cv2.putText(disp, f"{label} {score:.2f}", (int(x1), int(max(0, y1-5))),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 200, 255), 3)

        # SAM từ box: predictor.predict nhận box ở toạ độ ảnh gốc, định dạng XYXY
        input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=True,
        )
        idx = int(np.argmax(scores))
        chosen_mask = masks[idx].astype(np.uint8)  # (H,W) 0/1

        # Tô màu overlay riêng theo mask
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        mbool = chosen_mask.astype(bool)
        disp[mbool] = (0.45 * color + 0.55 * disp[mbool]).astype(np.uint8)

        # Lưu mask riêng theo nhãn
        per_label_count[label] = per_label_count.get(label, 0) + 1
        out_mask = (chosen_mask * 255)
        out_path = os.path.join(save_dir, f"{base}_{label}_{per_label_count[label]:02d}.png")
        cv2.imwrite(out_path, out_mask)

        # Ghi nhớ mask cuối cùng để hiển thị ở cửa sổ chính
        mask_best = chosen_mask
        score_best = float(np.max(scores))

    # Lưu tổng hợp
    out_overlay = os.path.join(save_dir, f"{base}_gdino_sam_overlay.png")
    cv2.imwrite(out_overlay, disp)
    overlay[:] = disp
    print(f"✅ Text→Box→SAM xong. Đã lưu overlay: {out_overlay}")
    print("📦 Lưu từng mask theo nhãn trong thư mục:", save_dir)

# ====== ISOLATE OBJECT(S) BY LABEL USING GDINO + SAM ======
def isolate_objects_by_label(
    text_queries=None,
    target_labels=None,
    save_dir=DEFAULT_SAVE_DIR,
    outside_value=OUTSIDE_VALUE_ISOLATE,
    crop_to_bbox=CROP_TO_BBOX_ISOLATE,
    show=True
):
    """
    1) Dùng GroundingDINO (Text→Box) toàn ảnh
    2) Lọc detections có nhãn khớp target_labels (so khớp mềm, không phân biệt hoa/thường)
    3) Với mỗi box, gọi SAM để lấy mask tốt nhất -> union tất cả mask
    4) Áp union mask vào ảnh gốc (ngoài mask = outside_value), tuỳ chọn crop bbox -> Lưu & hiển thị
    """
    if text_queries is None:
        text_queries = TEXT_QUERIES
    if not target_labels:
        print("⚠️ Chưa đặt TARGET_LABELS. Mặc định dùng TEXT_QUERIES làm nhãn đích.")
        target_labels = TEXT_QUERIES

    os.makedirs(save_dir, exist_ok=True)
    pil_img = Image.fromarray(img)
    print(f"📝 Text prompts (detect): {text_queries}")
    dets = run_text_boxes(pil_img, text_queries, BOX_THRESHOLD, TEXT_THRESHOLD)
    if len(dets) == 0:
        print("❌ GroundingDINO không tìm thấy đối tượng nào.")
        return

    # so khớp nhãn mềm: chứa chuỗi mục tiêu (case-insensitive)
    def label_match(lbl, targets):
        l = lbl.lower()
        return any(t.lower() in l for t in targets)

    dets_keep = [d for d in dets if label_match(d["label"], target_labels)]
    if len(dets_keep) == 0:
        print(f"❌ Không tìm thấy nhãn thuộc {target_labels} trong detections.")
        labs = [d["label"] for d in dets]
        if len(labs) > 0:
            print("🔎 Các nhãn GDINO nhận ra:", sorted(set(labs)))
        return

    # union mask từ SAM theo từng box
    union_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    for det in dets_keep:
        x1, y1, x2, y2 = det["box_xyxy"]
        input_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        idx = int(np.argmax(scores))
        chosen_mask = masks[idx].astype(bool)
        union_mask |= chosen_mask

    if not union_mask.any():
        print("⚠️ Union mask rỗng (SAM không sinh được vùng hợp lệ).")
        return

    # áp union mask vào ảnh
    out = np.full_like(bgr, outside_value, dtype=np.uint8)
    out[union_mask] = bgr[union_mask]

    # tuỳ chọn cắt khít theo bbox
    if crop_to_bbox:
        ys, xs = np.where(union_mask)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        out = out[y1:y2+1, x1:x2+1]

    # lưu & hiển thị
    base = os.path.splitext(os.path.basename(image_path))[0]
    tgt_name = "_".join([t.replace(" ", "") for t in target_labels])  # ví dụ "bridgebeam"
    out_path = os.path.join(save_dir, f"{base}_isolate_{tgt_name}.png")
    cv2.imwrite(out_path, out)
    print(f"✅ Đã tách vật thể '{target_labels}' và lưu: {out_path}")

    if show:
        win_iso = f"Isolate: {tgt_name}"
        cv2.imshow(win_iso, out)

# ====== HƯỚNG DẪN UI ======
redraw()
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN_NAME, mouse_cb)

print("👉 Hướng dẫn:")
print(" - Chuột trái: Foreground (xanh) hoặc chọn mask nếu đã bấm M")
print(" - Chuột phải: Background (đỏ)")
print(" - Enter: SAM Interactive (click)")
print(" - A: SAM Auto (mask lớn nhất)")
print(" - M: SAM Auto (nhiều mask) + click chọn")
print(" - T: Text→Box→SAM (GroundingDINO + SAM) với prompts:", TEXT_QUERIES)
print(" - I: Isolate theo nhãn (TARGET_LABELS) → tách vật thể và hiển thị/lưu")
print(" - R: reset")
print(" - S: lưu mask PNG (mask hiện tại + overlay hiện tại vào thư mục GDINO_SAM_<tên ảnh>)")
print(" - Q: QUALITY profile | U: ULTRA profile | P: in thông số hiện tại")
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
        run_sam_auto_all()
    elif k in (ord('t'), ord('T')):
        run_text_then_sam(TEXT_QUERIES); redraw()
    elif k in (ord('i'), ord('I')):
        isolate_objects_by_label(
            text_queries=TEXT_QUERIES,
            target_labels=TARGET_LABELS,
            save_dir=DEFAULT_SAVE_DIR,
            outside_value=OUTSIDE_VALUE_ISOLATE,
            crop_to_bbox=CROP_TO_BBOX_ISOLATE,
            show=True
        )
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
    elif k in (ord('p'), ord('P')):
        print_current_params()

cv2.destroyAllWindows()
