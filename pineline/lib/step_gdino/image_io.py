from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Cho phép Pillow mở ảnh rất lớn (tif facade) mà không cảnh báo DecompressionBomb.
try:  # pragma: no cover - tuỳ môi trường
    from PIL import Image as _PILImage

    _PILImage.MAX_IMAGE_PIXELS = None
except Exception:  # pragma: no cover
    _PILImage = None


@dataclass(frozen=True)
class WorkingImage:
    """Ảnh RGB dùng để chạy GDINO + SAM.

    `scale = work_size / orig_size`. Nhân tọa độ working với (1/scale) để map về
    ảnh gốc nếu cần. Pipeline này lưu/cắt trong KHÔNG GIAN working nên hầu hết
    consumer không cần map ngược.
    """

    rgb: np.ndarray          # HxWx3 uint8, RGB
    work_path: Path          # nơi đã ghi working RGB (PNG/JPG)
    scale: float             # work / orig (<= 1.0 nếu có downscale)
    orig_width: int
    orig_height: int

    @property
    def work_width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def work_height(self) -> int:
        return int(self.rgb.shape[0])


def load_rgb_any(image_path: Path) -> np.ndarray:
    """Đọc bất kỳ ảnh nào (tif/jpg/png/...) thành RGB uint8.

    Thử OpenCV trước (nhanh, đọc tif tốt); fallback Pillow cho định dạng lạ.
    """
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if _PILImage is not None:
        with _PILImage.open(image_path) as pil:
            return np.array(pil.convert("RGB"))
    raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")


def to_working_rgb(
    image_path: Path,
    out_path: Path,
    *,
    max_side: int = 4096,
) -> WorkingImage:
    """Đọc ảnh gốc → RGB, downscale nếu cạnh dài > max_side, ghi working PNG.

    Trả về `WorkingImage` (mảng RGB + scale + kích thước gốc). Việc downscale giúp
    `SamPredictor.set_image` không nổ RAM với ảnh .tif rất lớn, đồng thời vẫn đủ
    chi tiết để khoanh vùng nhà.
    """
    rgb = load_rgb_any(image_path)
    orig_h, orig_w = rgb.shape[:2]

    scale = 1.0
    long_side = max(orig_w, orig_h)
    if max_side and max_side > 0 and long_side > int(max_side):
        scale = float(max_side) / float(long_side)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ghi PNG (BGR cho cv2) để GDINO đọc lại được bằng đường dẫn.
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return WorkingImage(
        rgb=rgb,
        work_path=out_path,
        scale=float(scale),
        orig_width=int(orig_w),
        orig_height=int(orig_h),
    )
