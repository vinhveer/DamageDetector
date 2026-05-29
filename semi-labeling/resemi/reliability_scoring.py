from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from .embedding_cache import load_embeddings


@dataclass(frozen=True)
class ReliabilityConfig:
    accept_threshold: float = 0.75
    suspect_threshold: float = 0.50
    strong_margin_threshold: float = 0.10
    prototype_reject_threshold: float = 0.80
    core_disagree_margin: float = 0.03
    require_external_evidence_for_auto: bool = True
    weights: dict[str, float] | None = None

    @property
    def effective_weights(self) -> dict[str, float]:
        return self.weights or {
            "semantic_confidence": 0.20,
            "model_agreement": 0.18,
            "top1_top2_margin": 0.14,
            "prototype": 0.16,
            "core": 0.14,
            "multi_crop_consistency": 0.06,
            "geometry_prior": 0.08,
            "detector_prompt_agreement": 0.04,
            "outlier_penalty": 0.20,
        }


@dataclass(frozen=True)
class ReliabilityScoreResult:
    result_id: int
    reliability_score: float
    decision_type: str
    model_agreement: float
    top1_top2_margin: float
    nearest_core_class: str | None
    nearest_core_similarity: float | None
    prototype_class: str | None
    prototype_similarity: float | None
    reason_codes: list[str]
    components: dict[str, object]

    @property
    def reason_codes_json(self) -> str:
        return json.dumps(sorted(set(self.reason_codes)), ensure_ascii=False, sort_keys=True)

    @property
    def components_json(self) -> str:
        return json.dumps(self.components, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class ReliabilityScoringResult:
    reliability_run_id: str
    run_id: str
    core_mining_run_id: str | None
    prototype_score_run_id: str | None
    options: dict[str, object]
    scores: list[ReliabilityScoreResult]

    @property
    def auto_accept_count(self) -> int:
        return sum(1 for item in self.scores if item.decision_type == "auto_accept")

    @property
    def suspect_count(self) -> int:
        return sum(1 for item in self.scores if item.decision_type in {"suspect", "relabel_candidate"})

    @property
    def reject_count(self) -> int:
        return sum(1 for item in self.scores if item.decision_type == "reject")


def run_reliability_scoring(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    config: ReliabilityConfig,
    core_mining_run_id: str | None = "latest",
    prototype_score_run_id: str | None = "latest",
) -> ReliabilityScoringResult:
    rows = _read_decision_rows(conn, run_id=run_id)
    if not rows:
        raise RuntimeError(f"No semantic_decisions found for run_id={run_id}")
    core_run_id = _resolve_core_run_id(conn, run_id=run_id, core_mining_run_id=core_mining_run_id)
    proto_run_id = _resolve_prototype_score_run_id(conn, run_id=run_id, prototype_score_run_id=prototype_score_run_id)
    core_features = _read_core_features(conn, run_id=run_id, core_mining_run_id=core_run_id, result_ids=[int(row["result_id"]) for row in rows])
    prototype_features = _read_prototype_features(conn, prototype_score_run_id=proto_run_id)
    consistency_features = _read_consistency_features(conn, run_id=run_id)
    geometry_features = _read_geometry_features(conn, run_id=run_id)
    outlier_features = _read_outlier_features(conn, core_mining_run_id=core_run_id)
    weights = config.effective_weights
    scores = [
        _score_row(
            row,
            config=config,
            weights=weights,
            core=core_features.get(int(row["result_id"])),
            prototype=prototype_features.get(int(row["result_id"])),
            consistency=consistency_features.get(int(row["result_id"])),
            geometry=geometry_features.get(int(row["result_id"])),
            outlier=outlier_features.get(int(row["result_id"])),
        )
        for row in rows
    ]
    suffix = f"{(core_run_id or 'nocore')[:12]}_{(proto_run_id or 'noproto')[:12]}"
    reliability_run_id = f"rel_{run_id}_{suffix}"
    return ReliabilityScoringResult(
        reliability_run_id=reliability_run_id,
        run_id=run_id,
        core_mining_run_id=core_run_id,
        prototype_score_run_id=proto_run_id,
        options={
            "accept_threshold": config.accept_threshold,
            "suspect_threshold": config.suspect_threshold,
            "strong_margin_threshold": config.strong_margin_threshold,
            "prototype_reject_threshold": config.prototype_reject_threshold,
            "core_disagree_margin": config.core_disagree_margin,
            "require_external_evidence_for_auto": config.require_external_evidence_for_auto,
            "weights": weights,
        },
        scores=scores,
    )


def persist_reliability_scoring_result(conn: sqlite3.Connection, result: ReliabilityScoringResult, *, created_at_utc: str, update_decisions: bool = True) -> None:
    conn.execute("DELETE FROM reliability_scoring_runs WHERE reliability_run_id = ?", (result.reliability_run_id,))
    conn.execute(
        """
        INSERT INTO reliability_scoring_runs (
            reliability_run_id, run_id, created_at_utc, core_mining_run_id,
            prototype_score_run_id, options_json, scored_count, auto_accept_count,
            suspect_count, reject_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.reliability_run_id,
            result.run_id,
            created_at_utc,
            result.core_mining_run_id,
            result.prototype_score_run_id,
            json.dumps(result.options, ensure_ascii=False, sort_keys=True),
            len(result.scores),
            result.auto_accept_count,
            result.suspect_count,
            result.reject_count,
        ),
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO reliability_scores (
            run_id, result_id, reliability_score, model_agreement, top1_top2_margin,
            nearest_core_class, nearest_core_similarity, prototype_class, prototype_similarity,
            reason_codes_json, score_components_json, created_at_utc, reliability_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                result.run_id,
                score.result_id,
                score.reliability_score,
                score.model_agreement,
                score.top1_top2_margin,
                score.nearest_core_class,
                score.nearest_core_similarity,
                score.prototype_class,
                score.prototype_similarity,
                score.reason_codes_json,
                score.components_json,
                created_at_utc,
                result.reliability_run_id,
            )
            for score in result.scores
        ],
    )
    if update_decisions:
        conn.executemany(
            """
            UPDATE semantic_decisions
            SET reliability_score = ?, decision_type = ?, nearest_core_class = ?, nearest_core_similarity = ?,
                prototype_class = ?, prototype_similarity = ?, reason_codes_json = ?, score_components_json = ?
            WHERE run_id = ? AND result_id = ?
            """,
            [
                (
                    score.reliability_score,
                    score.decision_type,
                    score.nearest_core_class,
                    score.nearest_core_similarity,
                    score.prototype_class,
                    score.prototype_similarity,
                    score.reason_codes_json,
                    score.components_json,
                    result.run_id,
                    score.result_id,
                )
                for score in result.scores
            ],
        )
    conn.commit()


def _read_decision_rows(conn: sqlite3.Connection, *, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT d.*, a.agreement_ratio, a.majority_label
        FROM semantic_decisions d
        LEFT JOIN semantic_agreements a ON a.run_id = d.run_id AND a.result_id = d.result_id
        WHERE d.run_id = ?
        ORDER BY d.result_id
        """,
        (run_id,),
    ).fetchall()


def _resolve_core_run_id(conn: sqlite3.Connection, *, run_id: str, core_mining_run_id: str | None) -> str | None:
    if core_mining_run_id in (None, "", "none"):
        return None
    if core_mining_run_id != "latest":
        return str(core_mining_run_id)
    row = conn.execute(
        "SELECT core_mining_run_id FROM core_mining_runs WHERE run_id = ? ORDER BY created_at_utc DESC, core_mining_run_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return str(row["core_mining_run_id"]) if row is not None else None


def _resolve_prototype_score_run_id(conn: sqlite3.Connection, *, run_id: str, prototype_score_run_id: str | None) -> str | None:
    if prototype_score_run_id in (None, "", "none"):
        return None
    if prototype_score_run_id != "latest":
        return str(prototype_score_run_id)
    row = conn.execute(
        "SELECT prototype_score_run_id FROM prototype_scoring_runs WHERE run_id = ? ORDER BY created_at_utc DESC, prototype_score_run_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return str(row["prototype_score_run_id"]) if row is not None else None


def _read_core_features(conn: sqlite3.Connection, *, run_id: str, core_mining_run_id: str | None, result_ids: list[int]) -> dict[int, dict[str, object]]:
    if not core_mining_run_id:
        return {}
    run = conn.execute("SELECT * FROM core_mining_runs WHERE core_mining_run_id = ?", (core_mining_run_id,)).fetchone()
    if run is None:
        return {}
    clusters = conn.execute(
        """
        SELECT core_cluster_id, label, centroid_blob, density_score, agreement_score
        FROM core_clusters
        WHERE core_mining_run_id = ? AND centroid_blob IS NOT NULL
        ORDER BY core_cluster_id
        """,
        (core_mining_run_id,),
    ).fetchall()
    if not clusters:
        return {}
    embeddings, ids, embedding_run = load_embeddings(
        conn,
        model_name=str(run["model_name"]),
        view_name=str(run["view_name"]),
        run_id=run_id,
        embedding_run_id=str(run["embedding_run_id"]),
        result_ids=result_ids,
    )
    dim = int(embedding_run["dim"])
    centroids = np.vstack([_decode_vector(bytes(row["centroid_blob"]), dim=dim) for row in clusters]).astype(np.float32, copy=False)
    features: dict[int, dict[str, object]] = {}
    for row_idx, result_id in enumerate(ids):
        sims = embeddings[row_idx] @ centroids.T
        best_idx = int(np.argmax(sims))
        best = clusters[best_idx]
        features[int(result_id)] = {
            "nearest_core_class": str(best["label"]),
            "nearest_core_similarity": float(sims[best_idx]),
            "nearest_core_cluster_id": str(best["core_cluster_id"]),
            "cluster_density": float(best["density_score"] or 0.0),
            "cluster_agreement": float(best["agreement_score"] or 0.0),
            "core_mining_run_id": core_mining_run_id,
        }
    return features


def _read_prototype_features(conn: sqlite3.Connection, *, prototype_score_run_id: str | None) -> dict[int, dict[str, object]]:
    if not prototype_score_run_id:
        return {}
    rows = conn.execute(
        """
        SELECT result_id, top_prototype_label, top_prototype_sim, second_prototype_label,
               second_prototype_sim, prototype_margin, top_prototype_result_id, is_reject_match
        FROM prototype_scores
        WHERE prototype_score_run_id = ?
        """,
        (prototype_score_run_id,),
    ).fetchall()
    return {
        int(row["result_id"]): {
            "prototype_score_run_id": prototype_score_run_id,
            "prototype_class": str(row["top_prototype_label"]),
            "prototype_similarity": float(row["top_prototype_sim"]),
            "second_prototype_label": row["second_prototype_label"],
            "second_prototype_similarity": row["second_prototype_sim"],
            "prototype_margin": float(row["prototype_margin"]),
            "top_prototype_result_id": row["top_prototype_result_id"],
            "is_reject_match": bool(int(row["is_reject_match"])),
        }
        for row in rows
    }


def _read_consistency_features(conn: sqlite3.Connection, *, run_id: str) -> dict[int, dict[str, object]]:
    rows = conn.execute("SELECT result_id, label_consistency, view_disagreement_count, status FROM crop_consistency_features WHERE run_id = ?", (run_id,)).fetchall()
    return {int(row["result_id"]): dict(row) for row in rows}


def _read_geometry_features(conn: sqlite3.Connection, *, run_id: str) -> dict[int, dict[str, object]]:
    graph = conn.execute("SELECT box_graph_run_id FROM box_graph_runs WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1", (run_id,)).fetchone()
    if graph is None:
        return {}
    rows = conn.execute(
        """
        SELECT q.result_id, q.box_quality_score, d.decision_type, d.keep_for_cleaned
        FROM box_quality_scores q
        LEFT JOIN box_cleanup_decisions d ON d.box_graph_run_id = q.box_graph_run_id AND d.result_id = q.result_id
        WHERE q.box_graph_run_id = ?
        """,
        (str(graph["box_graph_run_id"]),),
    ).fetchall()
    return {int(row["result_id"]): dict(row) for row in rows}


def _read_outlier_features(conn: sqlite3.Connection, *, core_mining_run_id: str | None) -> dict[int, dict[str, object]]:
    if not core_mining_run_id:
        return {}
    rows = conn.execute("SELECT result_id, outlier_type, similarity, reason_codes_json FROM core_outliers WHERE core_mining_run_id = ?", (core_mining_run_id,)).fetchall()
    return {int(row["result_id"]): dict(row) for row in rows}


def _score_row(
    row: sqlite3.Row,
    *,
    config: ReliabilityConfig,
    weights: dict[str, float],
    core: dict[str, object] | None,
    prototype: dict[str, object] | None,
    consistency: dict[str, object] | None,
    geometry: dict[str, object] | None,
    outlier: dict[str, object] | None,
) -> ReliabilityScoreResult:
    result_id = int(row["result_id"])
    final_label = str(row["final_label"])
    previous_reasons = [item for item in _parse_json_list(row["reason_codes_json"]) if item not in _DYNAMIC_REASON_CODES]
    previous_components = _parse_json_dict(row["score_components_json"])
    semantic_confidence = _clip01(float(previous_components.get("semantic_confidence", row["reliability_score"] or 0.0)))
    margin_raw = float(previous_components.get("top1_top2_margin", 0.0))
    margin_component = _clip01(margin_raw / max(config.strong_margin_threshold, 1e-6))
    agreement = _clip01(float(row["agreement_ratio"] if row["agreement_ratio"] is not None else row["model_agreement"] or 0.0))
    detector_prompt = _clip01(float(previous_components.get("detector_prompt_agreement", 0.0)))
    components: dict[str, object] = dict(previous_components)
    positive_parts: list[tuple[str, float, float]] = [
        ("semantic_confidence", semantic_confidence, weights["semantic_confidence"]),
        ("model_agreement", agreement, weights["model_agreement"]),
        ("top1_top2_margin", margin_component, weights["top1_top2_margin"]),
        ("detector_prompt_agreement", detector_prompt, weights["detector_prompt_agreement"]),
    ]
    reasons = set(previous_reasons)
    prototype_class: str | None = None
    prototype_similarity: float | None = None
    if prototype:
        prototype_class = str(prototype["prototype_class"])
        prototype_similarity = float(prototype["prototype_similarity"])
        prototype_margin = float(prototype["prototype_margin"])
        prototype_score = (0.75 * _cosine_to_unit(prototype_similarity)) + (0.25 * _clip01(prototype_margin / 0.10))
        if prototype_class != final_label:
            prototype_score *= 0.50
            reasons.add("prototype_label_disagree")
        if bool(prototype["is_reject_match"]):
            reasons.add("near_reject_prototype")
        positive_parts.append(("prototype", prototype_score, weights["prototype"]))
        components.update(prototype)
    else:
        reasons.add("missing_prototype_signal")
    nearest_core_class: str | None = None
    nearest_core_similarity: float | None = None
    if core:
        nearest_core_class = str(core["nearest_core_class"])
        nearest_core_similarity = float(core["nearest_core_similarity"])
        core_score = _cosine_to_unit(nearest_core_similarity)
        if nearest_core_class != final_label:
            core_score *= 0.40
            reasons.add("openclip_core_disagree")
        elif nearest_core_similarity < 0.40:
            reasons.add("far_from_class_core")
        positive_parts.append(("core", core_score, weights["core"]))
        if core.get("cluster_density") is not None:
            positive_parts.append(("cluster_density", _cosine_to_unit(float(core["cluster_density"])), 0.04))
        components.update(core)
    else:
        reasons.add("missing_core_signal")
    if consistency and consistency.get("label_consistency") is not None:
        consistency_score = _clip01(float(consistency["label_consistency"]))
        positive_parts.append(("multi_crop_consistency", consistency_score, weights["multi_crop_consistency"]))
        if consistency_score < 0.60:
            reasons.add("multi_crop_inconsistent")
        components["multi_crop_consistency"] = consistency_score
    else:
        components["multi_crop_consistency_status"] = str(consistency.get("status") if consistency else "missing")
    if geometry and geometry.get("box_quality_score") is not None:
        geometry_score = _clip01(float(geometry["box_quality_score"]))
        positive_parts.append(("geometry_prior_score", geometry_score, weights["geometry_prior"])
        )
        if str(geometry.get("decision_type") or "").startswith("suspect") or int(geometry.get("keep_for_cleaned") or 1) == 0:
            reasons.add("geometry_conflict")
        components.update({f"geometry_{key}": value for key, value in geometry.items() if key != "result_id"})
    outlier_score = 0.0
    if outlier:
        outlier_score = 1.0
        outlier_type = str(outlier["outlier_type"])
        reasons.add("rare_cluster" if outlier_type == "rare_cluster" else outlier_type)
        components.update({f"outlier_{key}": value for key, value in outlier.items() if key != "result_id"})
    total_weight = sum(weight for _, _, weight in positive_parts)
    weighted = sum(value * weight for _, value, weight in positive_parts) / max(total_weight, 1e-9)
    reliability = _clip01(weighted - (weights["outlier_penalty"] * outlier_score))
    components["reliability_v1_parts"] = {name: value for name, value, _ in positive_parts}
    components["reliability_v1_weights"] = {name: weight for name, _, weight in positive_parts}
    components["outlier_score"] = outlier_score
    has_external_evidence = core is not None or prototype is not None
    can_auto_accept = has_external_evidence or not config.require_external_evidence_for_auto
    if reliability >= config.accept_threshold and can_auto_accept and "near_reject_prototype" not in reasons and "geometry_conflict" not in reasons:
        decision_type = "auto_accept"
        reasons.add("high_consensus")
    elif "near_reject_prototype" in reasons and prototype_similarity is not None and prototype_similarity >= config.prototype_reject_threshold:
        decision_type = "reject"
    elif reliability < config.suspect_threshold or "openclip_core_disagree" in reasons or "geometry_conflict" in reasons:
        decision_type = "suspect"
        if margin_raw < config.strong_margin_threshold:
            reasons.add("low_margin")
    else:
        decision_type = "suspect"
        if not has_external_evidence:
            reasons.add("needs_core_or_prototype_evidence")
    return ReliabilityScoreResult(
        result_id=result_id,
        reliability_score=reliability,
        decision_type=decision_type,
        model_agreement=agreement,
        top1_top2_margin=margin_raw,
        nearest_core_class=nearest_core_class,
        nearest_core_similarity=nearest_core_similarity,
        prototype_class=prototype_class,
        prototype_similarity=prototype_similarity,
        reason_codes=sorted(reasons),
        components=components,
    )


def _parse_json_dict(raw: object) -> dict[str, object]:
    try:
        value = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _parse_json_list(raw: object) -> list[str]:
    try:
        value = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _cosine_to_unit(value: float) -> float:
    return _clip01((float(value) + 1.0) / 2.0)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _decode_vector(blob: bytes, *, dim: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype="<f4")
    if vector.size != int(dim):
        raise ValueError(f"Invalid core centroid size: got {vector.size}, expected {dim}")
    return vector.astype(np.float32, copy=True)


_DYNAMIC_REASON_CODES = {
    "high_consensus",
    "low_margin",
    "low_semantic_reliability",
    "needs_core_or_prototype_evidence",
    "missing_core_signal",
    "missing_prototype_signal",
    "prototype_label_disagree",
    "openclip_core_disagree",
    "near_reject_prototype",
    "far_from_class_core",
    "multi_crop_inconsistent",
    "geometry_conflict",
    "rare_cluster",
    "noise_cluster",
    "insufficient_class_samples",
}
