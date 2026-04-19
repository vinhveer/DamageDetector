from PIL import Image


def load_image_rgb(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def normalize_roi_box(roi_box, full_size):
    if roi_box is None:
        return None
    left, top, right, bottom = [int(v) for v in roi_box]
    left = max(0, min(left, full_size[0]))
    right = max(0, min(right, full_size[0]))
    top = max(0, min(top, full_size[1]))
    bottom = max(0, min(bottom, full_size[1]))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid roi_box after clamping: {(left, top, right, bottom)}")
    return (left, top, right, bottom)


def crop_image_with_roi(img: Image.Image, roi_box):
    roi_box = normalize_roi_box(roi_box, img.size) if roi_box is not None else None
    if roi_box is None:
        return img, None
    return img.crop(roi_box), roi_box


def _letterbox_with_params(img: Image.Image, size: int, fill, interpolation):
    w, h = img.size
    if w == 0 or h == 0:
        return img, (0, 0, w, h)

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=interpolation)

    canvas = Image.new(img.mode, (size, size), color=fill)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas, (left, top, new_w, new_h)
