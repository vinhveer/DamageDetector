from __future__ import annotations

from typing import Any

import open_clip
import torch
from PIL import Image

from pineline.sam_gdino.step4_openclip_semantic.prompts import (
    DEFAULT_LABEL_PROMPTS,
    build_prompt_index,
)


def auto_device(requested: str = "auto") -> str:
    req = str(requested or "auto").strip().lower()
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenClipClassifier:
    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        prompt_groups: dict[str, list[str]] | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.device = auto_device(device)
        self.prompt_groups = dict(prompt_groups or DEFAULT_LABEL_PROMPTS)
        self.prompt_labels, self.prompt_texts = build_prompt_index(self.prompt_groups)
        if not self.prompt_texts:
            raise ValueError("Empty CLIP prompt index.")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        with torch.inference_mode():
            tokens = self.tokenizer(self.prompt_texts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        self._text_features = feats

    @torch.inference_mode()
    def classify_batch(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        if not images:
            return []
        tensors = [self.preprocess(im.convert("RGB")) for im in images]
        batch = torch.stack(tensors).to(self.device)
        img_feats = self.model.encode_image(batch)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_feats @ self._text_features.T
        probs = logits.softmax(dim=-1).detach().cpu().tolist()
        return [self._aggregate(p) for p in probs]

    def _aggregate(self, prompt_probs: list[float]) -> dict[str, Any]:
        class_scores: dict[str, float] = {label: 0.0 for label in self.prompt_groups}
        for idx, p in enumerate(prompt_probs):
            class_scores[self.prompt_labels[idx]] += float(p)
        ranked = sorted(class_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        return {
            "predicted_label": top_label,
            "predicted_probability": float(top_score),
            "class_scores": [
                {"label": label, "probability": float(score)} for label, score in ranked
            ],
        }
