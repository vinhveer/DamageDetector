from __future__ import annotations

from typing import Any

import open_clip
from PIL import Image
import torch

from prompts import POS_PROMPTS, build_prompt_index


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenClipSemanticClassifier:
    def __init__(
        self,
        *,
        model_name: str,
        pretrained: str,
        device: str = "auto",
        prompt_groups: dict[str, list[str]] | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.device = auto_device() if str(device or "auto") == "auto" else str(device)
        self.prompt_groups = dict(prompt_groups or POS_PROMPTS)
        self.prompt_labels, self.prompt_texts = build_prompt_index(self.prompt_groups)
        if not self.prompt_texts:
            raise ValueError("Prompt list is empty.")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self._text_features = self._encode_texts(self.prompt_texts)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(encoded)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def classify_images(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        if not images:
            return []
        tensors = [self.preprocess(image.convert("RGB")) for image in images]
        image_batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
            image_features = self.model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ self._text_features.T
            batch_prompt_probs = logits.softmax(dim=-1).detach().cpu().tolist()

        return [self._classification_from_prompt_probs(prompt_probs) for prompt_probs in batch_prompt_probs]

    def classify_image(self, image: Image.Image) -> dict[str, Any]:
        return self.classify_images([image])[0]

    def _classification_from_prompt_probs(self, prompt_probs: list[float]) -> dict[str, Any]:
        prompt_rows: list[dict[str, Any]] = []
        class_scores: dict[str, float] = {label: 0.0 for label in self.prompt_groups}
        for idx, probability in enumerate(prompt_probs):
            label = self.prompt_labels[idx]
            prompt = self.prompt_texts[idx]
            value = float(probability)
            class_scores[label] += value
            prompt_rows.append(
                {
                    "label": label,
                    "prompt": prompt,
                    "probability": value,
                    "probability_pct": value * 100.0,
                }
            )

        ranked_classes = [
            {"label": label, "probability": float(score), "probability_pct": float(score) * 100.0}
            for label, score in sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        ranked_prompts = sorted(prompt_rows, key=lambda item: item["probability"], reverse=True)
        predicted = ranked_classes[0]
        return {
            "device": self.device,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "predicted_label": predicted["label"],
            "predicted_probability": predicted["probability"],
            "predicted_probability_pct": predicted["probability_pct"],
            "class_scores": ranked_classes,
            "top_prompts": ranked_prompts[:5],
        }
