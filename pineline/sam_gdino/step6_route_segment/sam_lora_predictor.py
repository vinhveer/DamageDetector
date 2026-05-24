"""SAM-LoRA per-detection predictor.

The SAM-LoRA checkpoint in results_v2/sam-finetune-lora-hq was trained with the
tile-full-box pattern: each training tile is fed as the whole image and the box
prompt is the tile's own bounds [0,0,W,H]. So at inference we must crop the
detection box (with padding) and feed that crop as a tile via
`model_tile_prob_map`. SamPredictor + arbitrary box prompts will NOT work — that
isn't how the model was trained.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch_runtime import select_torch_device


class TileFullBoxSamLoraPredictor:
    """Per-detection SAM-LoRA predictor using the tile-full-box pattern."""

    def __init__(
        self,
        *,
        sam_checkpoint: Path,
        delta_checkpoint: Path,
        device: str = "auto",
        middle_dim: int = 32,
        scaling_factor: float = 0.2,
        rank: int = 4,
        sam_model_type: str = "auto",
        box_padding_ratio: float = 0.10,
        min_crop_size: int = 96,
    ) -> None:
        from segmentation.sam.backbones.segment_anything import sam_model_registry
        from segmentation.sam.finetune.runtime import (
            apply_delta_to_sam,
            infer_sam_model_type_from_state_dict,
            load_checkpoint_state_dict,
            load_inference_config,
            resolve_decoder_type,
            resolve_image_size,
            resolve_predict_threshold,
        )

        state_dict_for_inference = load_checkpoint_state_dict(str(sam_checkpoint))
        inferred = infer_sam_model_type_from_state_dict(state_dict_for_inference)
        req = (sam_model_type or "auto").strip().lower()
        vit_name = inferred if req == "auto" else (inferred or req)
        if vit_name not in sam_model_registry:
            raise ValueError(f"Unknown SAM model type: {vit_name!r}")

        cfg = load_inference_config(str(delta_checkpoint))
        image_size = resolve_image_size(str(delta_checkpoint), None)
        decoder_type = resolve_decoder_type(str(delta_checkpoint), None)
        try:
            threshold = float(resolve_predict_threshold(str(delta_checkpoint), None))
        except Exception:
            threshold = 0.5

        # Pass checkpoint=ckpt — sam_model_registry handles internal size
        # adaptation via build_sam.load_from() for vanilla SAM checkpoints.
        sam, _ = sam_model_registry[vit_name](
            image_size=int(image_size),
            num_classes=1,
            checkpoint=str(sam_checkpoint),
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
            decoder_type=decoder_type,
            centerline_head=bool(cfg.get("centerline_head", False)),
        )
        apply_delta_to_sam(
            sam=sam,
            delta_type="lora",
            delta_ckpt_path=str(delta_checkpoint),
            middle_dim=int(middle_dim),
            scaling_factor=float(scaling_factor),
            rank=int(rank),
        )
        torch_device = select_torch_device(device)
        sam.to(device=torch_device).eval()

        self.model = sam
        self.device = torch_device
        self.model_type = vit_name
        self.image_size = int(image_size)
        self.decoder_type = decoder_type
        self.threshold = float(threshold)
        self.box_padding_ratio = float(box_padding_ratio)
        self.min_crop_size = int(min_crop_size)

    def _pad_box(self, box: tuple[int, int, int, int], H: int, W: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(round(bw * self.box_padding_ratio))
        pad_y = int(round(bh * self.box_padding_ratio))
        # Enforce min crop size to give the encoder enough context.
        need_x = max(0, self.min_crop_size - (bw + 2 * pad_x))
        need_y = max(0, self.min_crop_size - (bh + 2 * pad_y))
        pad_x += (need_x + 1) // 2
        pad_y += (need_y + 1) // 2
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(W, x2 + pad_x)
        ny2 = min(H, y2 + pad_y)
        return nx1, ny1, nx2, ny2

    @torch.inference_mode()
    def predict_boxes(self, image_rgb: np.ndarray, boxes_xyxy: list) -> list[tuple[np.ndarray | None, float]]:
        """For each box: crop with padding, fit to model image_size, run as
        tile-full-box, threshold, upscale back, embed into full-image mask."""
        import cv2 as _cv2
        from segmentation.sam.finetune.tiled_inference import model_tile_prob_map

        H, W = image_rgb.shape[:2]
        # HQ decoder requires multimask_output=False when num_classes=1.
        multimask = False
        results: list[tuple[np.ndarray | None, float]] = []
        for box in boxes_xyxy:
            cx1, cy1, cx2, cy2 = self._pad_box(tuple(box), H, W)
            crop = image_rgb[cy1:cy2, cx1:cx2]
            ch, cw = crop.shape[:2]
            if ch == 0 or cw == 0:
                results.append((None, 0.0))
                continue
            # Fit crop to model image_size while preserving aspect ratio.
            scale = float(self.image_size) / float(max(ch, cw))
            if scale < 1.0:
                rw = max(1, int(round(cw * scale)))
                rh = max(1, int(round(ch * scale)))
                fit_crop = _cv2.resize(crop, (rw, rh), interpolation=_cv2.INTER_AREA)
            else:
                fit_crop = crop
            prob = model_tile_prob_map(
                self.model,
                fit_crop,
                image_size=self.image_size,
                multimask_output=multimask,
                use_amp=False,
            )
            prob = np.asarray(prob, dtype=np.float32)
            if prob.shape != (ch, cw):
                prob = _cv2.resize(prob, (cw, ch), interpolation=_cv2.INTER_LINEAR)
            mask_crop = prob >= float(self.threshold)
            full_mask = np.zeros((H, W), dtype=bool)
            full_mask[cy1:cy2, cx1:cx2] = mask_crop
            score = float(prob.max()) if prob.size else 0.0
            results.append((full_mask, score))
        return results


# Backward-compat alias for the previous name used in runner.py.
PatchedSamLoraPredictor = TileFullBoxSamLoraPredictor
