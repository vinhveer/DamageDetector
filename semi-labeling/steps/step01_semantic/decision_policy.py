from __future__ import annotations

import json
from dataclasses import dataclass

from .semantic_ensemble import SemanticAgreement
from shared.db.source_store import SourceDetection


@dataclass(frozen=True)
class DecisionConfig:
    accept_threshold: float = 0.75
    suspect_threshold: float = 0.50
    low_margin_threshold: float = 0.03
    strong_margin_threshold: float = 0.10
    detector_conflict_penalty: float = 0.08
    labels: tuple[str, ...] = ("crack", "spall", "mold", "stain", "efflorescence")
    reject_labels: tuple[str, ...] = ("reject", "unknown", "background", "shadow", "edge", "object")


@dataclass(frozen=True)
class SemanticDecision:
    result_id: int
    initial_label: str
    suggested_label: str
    final_label: str
    decision_type: str
    reliability_score: float
    model_agreement: float
    top1_top2_margin: float
    reason_codes: list[str]
    score_components: dict[str, float | str | bool]

    @property
    def reason_codes_json(self) -> str:
        return json.dumps(self.reason_codes, ensure_ascii=False, sort_keys=True)

    @property
    def score_components_json(self) -> str:
        return json.dumps(self.score_components, ensure_ascii=False, sort_keys=True)


def decide(detection: SourceDetection, config: DecisionConfig, agreement: SemanticAgreement | None = None) -> SemanticDecision:
    ranked = sorted(detection.scores.items(), key=lambda item: item[1], reverse=True)
    top_label = str(ranked[0][0]) if ranked else detection.initial_label
    top_score = float(ranked[0][1]) if ranked else float(detection.initial_probability)
    second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    margin = max(0.0, top_score - second_score)

    detector_agrees = _label_matches_detector(top_label, detection.detector_label, detection.prompt_key)
    detector_component = 1.0 if detector_agrees else 0.0
    reliability = (0.65 * top_score) + (0.25 * min(1.0, margin / max(config.strong_margin_threshold, 1e-6))) + (0.10 * detector_component)
    if not detector_agrees:
        reliability -= float(config.detector_conflict_penalty)
    agreement_ratio = 1.0 if agreement is None else float(agreement.agreement_ratio)
    majority_label = top_label if agreement is None else agreement.majority_label
    strong_agreement_count = 1 if agreement is None else int(agreement.strong_agreement_count)
    if agreement is not None:
        if agreement_ratio >= 0.75 and strong_agreement_count >= 1:
            reliability += 0.08
        elif agreement_ratio <= 0.50:
            reliability -= 0.18
    reliability = max(0.0, min(1.0, reliability))

    reason_codes: list[str] = []
    if top_score >= config.accept_threshold and margin >= config.strong_margin_threshold:
        reason_codes.append("high_consensus")
    if margin < config.low_margin_threshold:
        reason_codes.append("low_margin")
    if not detector_agrees:
        reason_codes.append("detector_semantic_conflict")
    if agreement is not None and agreement_ratio >= 0.75:
        reason_codes.append("model_agreement_high")
    if agreement is not None and agreement_ratio <= 0.50:
        reason_codes.append("model_disagreement")
    if top_label in set(config.reject_labels):
        reason_codes.append("near_reject_prototype")

    if top_label in set(config.reject_labels):
        decision_type = "reject"
        final_label = "reject"
    elif agreement is not None and majority_label != top_label and agreement_ratio >= 0.75:
        decision_type = "relabel_candidate"
        final_label = majority_label
        reason_codes.append("majority_relabels_openclip")
    elif agreement is not None and agreement_ratio <= 0.50:
        decision_type = "suspect"
        final_label = top_label
    elif reliability >= config.accept_threshold and margin >= config.low_margin_threshold:
        decision_type = "auto_accept"
        final_label = top_label
    elif reliability < config.suspect_threshold or margin < config.low_margin_threshold:
        decision_type = "suspect"
        final_label = top_label
        if "low_margin" not in reason_codes:
            reason_codes.append("low_semantic_reliability")
    else:
        decision_type = "suspect"
        final_label = top_label
        reason_codes.append("needs_core_or_prototype_evidence")

    if top_label != detection.initial_label:
        reason_codes.append("top_score_differs_from_initial_label")

    components: dict[str, float | str | bool] = {
        "semantic_confidence": top_score,
        "top1_top2_margin": margin,
        "detector_prompt_agreement": detector_component,
        "detector_agrees": detector_agrees,
        "top_label": top_label,
        "second_score": second_score,
        "majority_label": majority_label,
        "agreement_ratio": agreement_ratio,
        "strong_agreement_count": strong_agreement_count,
    }
    return SemanticDecision(
        result_id=detection.result_id,
        initial_label=detection.initial_label,
        suggested_label=top_label,
        final_label=final_label,
        decision_type=decision_type,
        reliability_score=reliability,
        model_agreement=agreement_ratio,
        top1_top2_margin=margin,
        reason_codes=sorted(set(reason_codes)),
        score_components=components,
    )


def _label_matches_detector(label: str, detector_label: str, prompt_key: str) -> bool:
    haystack = f"{detector_label} {prompt_key}".lower()
    normalized = str(label or "").lower().strip()
    if not normalized:
        return False
    aliases = {
        "spall": ("spall", "spalling", "delamination", "flaking"),
        "mold": ("mold", "mould", "stain", "moisture", "dirty"),
        "crack": ("crack", "fracture", "fissure"),
        "stain": ("stain", "mold", "mould", "moisture"),
        "efflorescence": ("efflorescence", "salt", "white"),
    }
    return any(alias in haystack for alias in aliases.get(normalized, (normalized,)))
