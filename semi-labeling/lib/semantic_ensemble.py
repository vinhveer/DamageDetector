from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

from .source_store import SourceDetection


@dataclass(frozen=True)
class SemanticModelOutput:
    result_id: int
    model_name: str
    source_type: str
    top1_label: str
    top1_score: float
    top2_label: str | None
    top2_score: float | None
    margin: float
    entropy: float | None
    raw_scores: dict[str, float | str | None]

    @property
    def raw_scores_json(self) -> str:
        return json.dumps(self.raw_scores, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class SemanticAgreement:
    result_id: int
    majority_label: str
    agreement_ratio: float
    strong_agreement_count: int
    conflict_labels: list[str]
    sources: list[dict[str, float | str | None]]

    @property
    def conflict_labels_json(self) -> str:
        return json.dumps(self.conflict_labels, ensure_ascii=False, sort_keys=True)

    @property
    def sources_json(self) -> str:
        return json.dumps(self.sources, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class SemanticEnsembleResult:
    outputs: list[SemanticModelOutput]
    agreements: list[SemanticAgreement]

    @property
    def agreements_by_result_id(self) -> dict[int, SemanticAgreement]:
        return {item.result_id: item for item in self.agreements}


def build_semantic_ensemble(detections: list[SourceDetection]) -> SemanticEnsembleResult:
    outputs: list[SemanticModelOutput] = []
    agreements: list[SemanticAgreement] = []
    for detection in detections:
        detection_outputs = [openclip_output(detection)]
        detector_output = detector_prompt_output(detection)
        if detector_output is not None:
            detection_outputs.append(detector_output)
        outputs.extend(detection_outputs)
        agreements.append(compute_agreement(detection.result_id, detection_outputs))
    return SemanticEnsembleResult(outputs=outputs, agreements=agreements)


def openclip_output(detection: SourceDetection) -> SemanticModelOutput:
    ranked = sorted(detection.scores.items(), key=lambda item: item[1], reverse=True)
    top1_label = str(ranked[0][0]) if ranked else detection.initial_label
    top1_score = float(ranked[0][1]) if ranked else float(detection.initial_probability)
    top2_label = str(ranked[1][0]) if len(ranked) > 1 else None
    top2_score = float(ranked[1][1]) if len(ranked) > 1 else None
    margin = max(0.0, top1_score - float(top2_score or 0.0))
    return SemanticModelOutput(
        result_id=detection.result_id,
        model_name="openclip_step2",
        source_type="vision_language_model",
        top1_label=top1_label,
        top1_score=top1_score,
        top2_label=top2_label,
        top2_score=top2_score,
        margin=margin,
        entropy=_entropy([float(score) for _, score in ranked]),
        raw_scores={label: float(score) for label, score in ranked},
    )


def detector_prompt_output(detection: SourceDetection) -> SemanticModelOutput | None:
    label = infer_detector_label(detection)
    if label is None:
        return None
    score = max(0.0, min(1.0, float(detection.detector_score)))
    return SemanticModelOutput(
        result_id=detection.result_id,
        model_name="groundingdino_prompt",
        source_type="detector_prompt_prior",
        top1_label=label,
        top1_score=score,
        top2_label=None,
        top2_score=None,
        margin=score,
        entropy=None,
        raw_scores={label: score, "detector_label": detection.detector_label, "prompt_key": detection.prompt_key},
    )


def compute_agreement(result_id: int, outputs: list[SemanticModelOutput]) -> SemanticAgreement:
    if not outputs:
        return SemanticAgreement(result_id, "unknown", 0.0, 0, [], [])
    counts = Counter(output.top1_label for output in outputs)
    majority_label, majority_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    agreement_ratio = float(majority_count) / float(len(outputs))
    conflict_labels = sorted(label for label in counts if label != majority_label)
    strong_agreement_count = sum(
        1
        for output in outputs
        if output.top1_label == majority_label and output.top1_score >= 0.60 and output.margin >= 0.05
    )
    sources = [
        {
            "model_name": output.model_name,
            "source_type": output.source_type,
            "top1_label": output.top1_label,
            "top1_score": output.top1_score,
            "margin": output.margin,
        }
        for output in outputs
    ]
    return SemanticAgreement(
        result_id=result_id,
        majority_label=majority_label,
        agreement_ratio=agreement_ratio,
        strong_agreement_count=strong_agreement_count,
        conflict_labels=conflict_labels,
        sources=sources,
    )


def infer_detector_label(detection: SourceDetection) -> str | None:
    haystack = f"{detection.detector_label} {detection.prompt_key}".lower()
    aliases = {
        "crack": ("crack", "fracture", "fissure"),
        "spall": ("spall", "spalling", "delamination", "flaking", "broken", "chipped"),
        "mold": ("mold", "mould", "mildew", "moss"),
        "stain": ("stain", "dirty", "discoloration", "moisture"),
        "efflorescence": ("efflorescence", "salt", "white"),
    }
    for label, words in aliases.items():
        if any(word in haystack for word in words):
            return label
    return None


def _entropy(values: list[float]) -> float:
    total = sum(max(0.0, value) for value in values)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in values:
        probability = max(0.0, value) / total
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy
