from PIL import Image


class LetterboxResize:
    """
    Resize while preserving aspect ratio, then pad to the target size.

    This avoids geometric distortion from Resize((H, W)) on non-square images.

    Notes:
    - For masks, use nearest-neighbor interpolation to avoid creating gray edges.
    - For RGB images, use bilinear/bicubic interpolation for smoother results.
    """

    def __init__(self, size, fill=0, interpolation=Image.BILINEAR):
        if isinstance(size, int):
            self.target_w = size
            self.target_h = size
        else:
            self.target_h, self.target_w = size  # torchvision convention (H, W)
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return img

        scale = min(self.target_w / w, self.target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = img.resize((new_w, new_h), resample=self.interpolation)

        if img.mode == "RGB":
            fill = self.fill if isinstance(self.fill, tuple) else (self.fill, self.fill, self.fill)
        else:
            fill = self.fill if not isinstance(self.fill, tuple) else self.fill[0]

        canvas = Image.new(img.mode, (self.target_w, self.target_h), color=fill)
        left = (self.target_w - new_w) // 2
        top = (self.target_h - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas
