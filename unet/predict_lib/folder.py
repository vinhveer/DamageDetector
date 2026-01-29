import os
from .core import predict_image


def _iter_images(input_dir, recursive, exts):
    if recursive:
        for root, _, files in os.walk(input_dir):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in exts:
                    yield os.path.join(root, name)
        return

    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            yield path


def _safe_basename(input_dir, image_path):
    rel = os.path.relpath(image_path, input_dir)
    rel_no_ext = os.path.splitext(rel)[0]
    safe = rel_no_ext.replace("\\", "__").replace("/", "__")
    safe = safe.replace(":", "_")
    return safe


def _find_gt_mask(gt_dir, input_dir, image_path):
    if not gt_dir:
        return None

    rel = os.path.relpath(image_path, input_dir)
    rel_no_ext = os.path.splitext(rel)[0]
    candidate_roots = [
        os.path.join(gt_dir, rel_no_ext),
        os.path.join(gt_dir, os.path.splitext(os.path.basename(image_path))[0]),
    ]

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
    suffixes = ["", "_mask", "_gt", "_label", "_labels", "_seg", "_segmentation"]

    for root in candidate_roots:
        for suffix in suffixes:
            for ext in exts:
                cand = f"{root}{suffix}{ext}"
                if os.path.exists(cand):
                    return cand
    return None


def predict_folder(
    model,
    input_dir,
    device,
    *,
    output_dir="results",
    threshold=0.5,
    apply_postprocessing=True,
    recursive=False,
    gt_dir=None,
    mode="tile",
    input_size=256,
    tile_overlap=0,
    tile_batch_size=4,
):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted(_iter_images(input_dir, recursive, exts))
    if not image_paths:
        print(f"No images found in: {input_dir}")
        return []

    print(f"Found {len(image_paths)} image(s).")
    overlap = (input_size // 2) if tile_overlap == 0 else tile_overlap
    results = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {image_path}")
        output_basename = _safe_basename(input_dir, image_path)
        gt_mask_path = _find_gt_mask(gt_dir, input_dir, image_path)
        if gt_dir and not gt_mask_path:
            print(f"  [WARN] GT mask not found for: {image_path}")
        details = predict_image(
            model,
            image_path,
            device,
            threshold=threshold,
            output_dir=output_dir,
            apply_postprocessing=apply_postprocessing,
            output_basename=output_basename,
            gt_mask_path=gt_mask_path,
            gt_expected=bool(gt_dir),
            mode=mode,
            input_size=input_size,
            tile_overlap=overlap,
            tile_batch_size=tile_batch_size,
            return_details=True,
        )
        results.append(details)
    return results
