"""Generate synthetic crack images from existing masks with SD1.5 + ControlNet-Canny."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


PROMPTS = [
    "weathered concrete wall with crack, photorealistic, surface texture",
    "old wooden plank with crack, natural wood grain",
    "brick wall surface with crack, masonry texture",
    "stone surface with crack, rough texture",
    "rusty metal plate with crack, weathered",
    "tiled floor with crack, ceramic surface",
    "painted wall with crack, peeling paint texture",
    "asphalt road surface with crack, weathered tarmac",
]

NEGATIVE_PROMPT = "blurry, low quality, cartoon, drawing, watermark, text, letters"
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(binary)) == 0:
        return binary
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        return ximgproc.thinning(binary)

    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()
    while True:
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = eroded
        if int(cv2.countNonZero(work)) == 0:
            break
    return skeleton


def mask_to_canny(mask: np.ndarray) -> np.ndarray:
    skeleton = _skeletonize(mask)
    return cv2.dilate(skeleton, np.ones((2, 2), np.uint8), iterations=1)


def _load_pipeline(device: str):
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch_dtype,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if str(device).startswith("cuda"):
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
    else:
        pipe = pipe.to(device)
    return pipe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crack500_masks", required=True, help="Folder containing Crack500 binary masks")
    parser.add_argument("--output_dir", required=True, help="Output dataset root with images/ and masks/")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=0.9)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    mask_files = sorted(
        path for path in Path(args.crack500_masks).iterdir()
        if path.is_file() and path.suffix.lower() in MASK_EXTS
    )
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {args.crack500_masks}")

    out_img = Path(args.output_dir) / "images"
    out_mask = Path(args.output_dir) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    pipe = _load_pipeline(str(args.device))
    image_size = int(args.image_size)
    generator = torch.Generator(device=str(args.device) if str(args.device).startswith("cuda") else "cpu")
    generator.manual_seed(int(args.seed))

    for i in range(int(args.num_samples)):
        mask_path = random.choice(mask_files)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != (image_size, image_size):
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
        canny = mask_to_canny(mask)
        canny_pil = Image.fromarray(np.repeat(canny[:, :, None], 3, axis=2))
        prompt = random.choice(PROMPTS)
        result = pipe(
            prompt=prompt,
            image=canny_pil,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.guidance_scale),
            controlnet_conditioning_scale=float(args.controlnet_scale),
            generator=generator,
        ).images[0]
        out_name = f"synth_{i:05d}"
        result.save(out_img / f"{out_name}.jpg", quality=95)
        cv2.imwrite(str(out_mask / f"{out_name}.png"), mask)
        if i % 100 == 0:
            print(f"[{i}/{int(args.num_samples)}] {prompt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
