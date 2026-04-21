import cv2
import albumentations as A


def build_crack_profile_augment(profile: str):
    profile = str(profile or "balanced").strip().lower()
    if profile == "strong":
        profile = "aggressive"
    if profile == "advanced":
        profile = "outdomain"

    if profile == "outdomain":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.Affine(
                    scale=(0.80, 1.20),
                    translate_percent=(-0.08, 0.08),
                    rotate=(-40, 40),
                    shear=(-8, 8),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.Perspective(scale=(0.03, 0.08), keep_size=True, fit_output=False, p=1.0),
            ], p=0.65),
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.28, contrast_limit=0.28, p=1.0),
                A.RandomGamma(gamma_limit=(70, 135), p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=18, val_shift_limit=15, p=1.0),
                A.RGBShift(r_shift_limit=14, g_shift_limit=14, b_shift_limit=14, p=1.0),
            ], p=0.65),
            A.OneOf([
                A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), p=1.0),
                A.RandomToneCurve(scale=0.20, p=1.0),
                A.Sharpen(alpha=(0.15, 0.35), lightness=(0.7, 1.2), p=1.0),
            ], p=0.35),
            A.OneOf([
                A.ImageCompression(p=1.0),
                A.Downscale(p=1.0),
            ], p=0.40),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.Blur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.45),
            A.ToGray(p=0.08),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(0.04, 0.14),
                hole_width_range=(0.04, 0.14),
                fill=0,
                fill_mask=0,
                p=0.28,
            ),
        ], is_check_shapes=False)

    if profile == "aggressive":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.Affine(
                    scale=(0.85, 1.15),
                    translate_percent=(-0.06, 0.06),
                    rotate=(-35, 35),
                    shear=(-6, 6),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.Perspective(scale=(0.02, 0.05), keep_size=True, fit_output=False, p=1.0),
            ], p=0.55),
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),
            A.OneOf([
                A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), p=1.0),
                A.RandomToneCurve(scale=0.15, p=1.0),
            ], p=0.25),
            A.OneOf([
                A.ImageCompression(p=1.0),
                A.Downscale(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.Blur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.35),
            A.CoarseDropout(p=0.2),
        ], is_check_shapes=False)

    if profile == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.35),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.02, 0.02),
                rotate=(-10, 10),
                shear=(-2, 2),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.15,
            ),
            A.OneOf([
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=1.0),
                A.RandomGamma(gamma_limit=(92, 108), p=1.0),
            ], p=0.22),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.Blur(blur_limit=3, p=1.0),
            ], p=0.08),
        ], is_check_shapes=False)

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.04, 0.04),
                rotate=(-20, 20),
                shear=(-4, 4),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
            A.Perspective(scale=(0.01, 0.03), keep_size=True, fit_output=False, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.16, contrast_limit=0.16, p=1.0),
            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
        ], p=0.35),
        A.OneOf([
            A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), p=1.0),
            A.RandomToneCurve(scale=0.12, p=1.0),
        ], p=0.12),
        A.OneOf([
            A.ImageCompression(p=1.0),
            A.Downscale(p=1.0),
        ], p=0.12),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.Blur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.18),
        A.CoarseDropout(p=0.08),
    ], is_check_shapes=False)
def build_imagenet_normalize():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
    ])
