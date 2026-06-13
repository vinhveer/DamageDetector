from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from shared.taxonomy.label_taxonomy import LabelTaxonomy, build_label_taxonomy


@dataclass(frozen=True)
class DecisionPolicyConfig:
    accept_threshold: float = 0.75
    suspect_threshold: float = 0.50
    prototype_min_sim: float = 0.70
    relabel_margin: float = 0.05
    ambiguous_margin: float = 0.03
    view_name: str = "tight"
    allow_low_priority_cleaned: bool = False
    prototype_min_sim_by_class: dict[str, float] | None = None
    suspect_threshold_by_class: dict[str, float] | None = None
    accept_threshold_by_class: dict[str, float] | None = None
    allow_low_priority_by_class: dict[str, bool] | None = None

    def min_sim_for(self, label: object) -> float:
        """Per-class prototype/core similarity threshold; falls back to the global value."""
        by_class = self.prototype_min_sim_by_class or {}
        return float(by_class.get(str(label or ""), self.prototype_min_sim))

    def suspect_threshold_for(self, label: object) -> float:
        """Per-class reliability floor below which a box becomes suspect; falls back to the global value."""
        by_class = self.suspect_threshold_by_class or {}
        return float(by_class.get(str(label or ""), self.suspect_threshold))

    def accept_threshold_for(self, label: object) -> float:
        """Per-class auto-accept reliability threshold; falls back to the global value."""
        by_class = self.accept_threshold_by_class or {}
        return float(by_class.get(str(label or ""), self.accept_threshold))

    def allow_low_priority_for(self, label: object) -> bool:
        """Per-class low-priority-cleaned switch; falls back to the global value."""
        by_class = self.allow_low_priority_by_class or {}
        return bool(by_class.get(str(label or ""), self.allow_low_priority_cleaned))

    @property
    def thresholds(self) -> dict[str, float | bool | str]:
        return {
            "accept_threshold": self.accept_threshold,
            "accept_threshold_by_class": dict(self.accept_threshold_by_class or {}),
            "suspect_threshold": self.suspect_threshold,
            "suspect_threshold_by_class": dict(self.suspect_threshold_by_class or {}),
            "prototype_min_sim": self.prototype_min_sim,
            "prototype_min_sim_by_class": dict(self.prototype_min_sim_by_class or {}),
            "relabel_margin": self.relabel_margin,
            "ambiguous_margin": self.ambiguous_margin,
            "view_name": self.view_name,
            "allow_low_priority_cleaned": self.allow_low_priority_cleaned,
            "allow_low_priority_by_class": dict(self.allow_low_priority_by_class or {}),
        }


@dataclass(frozen=True)
class PolicyDecision:
    result_id: int
    initial_label: str
    suggested_label: str
    final_label: str
    previous_decision_type: str
    final_decision_type: str
    matched_rule: str
    reliability_score: float
    reason_codes: list[str]
    score_components: dict[str, object]

    @property
    def reason_codes_json(self) -> str:
        return json.dumps(sorted(set(self.reason_codes)), ensure_ascii=False, sort_keys=True)

    @property
    def score_components_json(self) -> str:
        return json.dumps(self.score_components, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class DecisionPolicyResult:
    decision_policy_run_id: str
    run_id: str
    reliability_run_id: str | None
    taxonomy_version_id: str
    config: DecisionPolicyConfig
    decisions: list[PolicyDecision]
    cleaned_rows: list[tuple]
    review_rows: list[tuple]

    @property
    def cleaned_count(self) -> int:
        return len(self.cleaned_rows)

    @property
    def review_count(self) -> int:
        return len(self.review_rows)

    @property
    def reject_count(self) -> int:
        return sum(1 for item in self.review_rows if str(item[6]) == "reject")


def apply_decision_policy(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    config: DecisionPolicyConfig,
    reliability_run_id: str | None = "latest",
    taxonomy: LabelTaxonomy | None = None,
) -> DecisionPolicyResult:
    reliability_run = _resolve_reliability_run(conn, run_id=run_id, reliability_run_id=reliability_run_id)
    rows = _read_policy_inputs(conn, run_id=run_id, reliability_run_id=reliability_run, view_name=config.view_name)
    if not rows:
        raise RuntimeError(f"No policy input rows found for run_id={run_id}")
    taxonomy = taxonomy or _resolve_taxonomy(conn, run_id=run_id)
    box_decisions = _read_box_decisions(conn, run_id=run_id)
    decisions: list[PolicyDecision] = []
    cleaned_rows: list[tuple] = []
    review_by_id: dict[int, tuple] = {}
    policy_run_id = f"policy_{run_id}_{(reliability_run or 'norel')[:12]}"
    for row in rows:
        decision = _decide_row(row, config=config)
        decisions.append(decision)
        box = box_decisions.get(decision.result_id, {})
        box_type = str(box.get("decision_type") or "keep_representative")
        box_keep = bool(int(box.get("keep_for_cleaned", 1)))
        box_drop = box_type == "drop_nested_duplicate"
        box_review = box_type in {"suspect_composite_box", "suspect_broad_box", "manual_box_review"}
        image_rel_path = str(row["image_rel_path"] or "")
        crop_path = row["crop_path"]
        if decision.final_decision_type in {"auto_accept", "auto_accept_low_priority"} and box_keep and not box_review:
            cleaned_rows.append(
                (
                    run_id,
                    decision.result_id,
                    image_rel_path,
                    crop_path,
                    decision.final_label,
                    taxonomy.export_label(decision.final_label),
                    decision.final_decision_type,
                    decision.reliability_score,
                    decision.reason_codes_json,
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                    policy_run_id,
                )
            )
        if not box_drop and decision.final_decision_type in {"suspect", "reject", "relabel_candidate"}:
            review_by_id[decision.result_id] = (
                run_id,
                decision.result_id,
                image_rel_path,
                crop_path,
                decision.initial_label,
                decision.suggested_label,
                decision.final_decision_type,
                decision.reliability_score,
                decision.reason_codes_json,
                policy_run_id,
            )
        if box_review:
            reasons = sorted(set([*decision.reason_codes, "bbox_quality_filter", box_type]))
            review_by_id[decision.result_id] = (
                run_id,
                decision.result_id,
                image_rel_path,
                crop_path,
                decision.initial_label,
                decision.suggested_label,
                box_type,
                min(decision.reliability_score, float(box.get("box_quality_score", decision.reliability_score))),
                json.dumps(reasons, ensure_ascii=False, sort_keys=True),
                policy_run_id,
            )
    return DecisionPolicyResult(
        decision_policy_run_id=policy_run_id,
        run_id=run_id,
        reliability_run_id=reliability_run,
        taxonomy_version_id=taxonomy.version_id,
        config=config,
        decisions=decisions,
        cleaned_rows=cleaned_rows,
        review_rows=list(review_by_id.values()),
    )


def persist_decision_policy_result(conn: sqlite3.Connection, result: DecisionPolicyResult, *, created_at_utc: str) -> None:
    conn.execute("DELETE FROM decision_policy_audit WHERE decision_policy_run_id = ?", (result.decision_policy_run_id,))
    conn.execute("DELETE FROM decision_policy_runs WHERE decision_policy_run_id = ?", (result.decision_policy_run_id,))
    conn.execute("DELETE FROM cleaned_labels WHERE run_id = ?", (result.run_id,))
    conn.execute("DELETE FROM review_queue WHERE run_id = ?", (result.run_id,))
    conn.execute(
        """
        INSERT INTO decision_policy_runs (
            decision_policy_run_id, run_id, created_at_utc, reliability_run_id,
            taxonomy_version_id, options_json, total_count, cleaned_count, review_count, reject_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.decision_policy_run_id,
            result.run_id,
            created_at_utc,
            result.reliability_run_id,
            result.taxonomy_version_id,
            json.dumps(result.config.thresholds, ensure_ascii=False, sort_keys=True),
            len(result.decisions),
            result.cleaned_count,
            result.review_count,
            result.reject_count,
        ),
    )
    conn.executemany(
        """
        INSERT INTO decision_policy_audit (
            decision_policy_run_id, result_id, matched_rule, previous_decision_type,
            final_decision_type, final_label, reliability_score, thresholds_json,
            score_components_json, reason_codes_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                result.decision_policy_run_id,
                decision.result_id,
                decision.matched_rule,
                decision.previous_decision_type,
                decision.final_decision_type,
                decision.final_label,
                decision.reliability_score,
                json.dumps(result.config.thresholds, ensure_ascii=False, sort_keys=True),
                decision.score_components_json,
                decision.reason_codes_json,
            )
            for decision in result.decisions
        ],
    )
    conn.executemany(
        """
        UPDATE semantic_decisions
        SET final_label = ?, decision_type = ?, reason_codes_json = ?, score_components_json = ?, matched_rule = ?
        WHERE run_id = ? AND result_id = ?
        """,
        [
            (
                decision.final_label,
                decision.final_decision_type,
                decision.reason_codes_json,
                decision.score_components_json,
                decision.matched_rule,
                result.run_id,
                decision.result_id,
            )
            for decision in result.decisions
        ],
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO cleaned_labels (
            run_id, result_id, image_rel_path, crop_path, final_label, export_label,
            decision_type, reliability_score, reason_codes_json, x1, y1, x2, y2,
            decision_policy_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        result.cleaned_rows,
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO review_queue (
            run_id, result_id, image_rel_path, crop_path, initial_label, suggested_label,
            queue_type, reliability_score, reason_codes_json, decision_policy_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        result.review_rows,
    )
    _update_counts(conn, run_id=result.run_id)
    conn.commit()


def _decide_row(row: sqlite3.Row, *, config: DecisionPolicyConfig) -> PolicyDecision:
    result_id = int(row["result_id"])
    reliability = float(row["reliability_score"])
    initial_label = str(row["initial_label"])
    suggested_label = str(row["suggested_label"])
    final_label = str(row["final_label"])
    previous_decision_type = str(row["decision_type"])
    reasons = set(_parse_json_list(row["reason_codes_json"]))
    components = _parse_json_dict(row["score_components_json"])
    prototype_class = row["prototype_class"]
    prototype_similarity = row["prototype_similarity"]
    prototype_margin = float(components.get("prototype_margin", 0.0) or 0.0)
    nearest_core_class = row["nearest_core_class"]
    nearest_core_similarity = row["nearest_core_similarity"]
    has_conflict = _has_conflict(reasons)

    if "near_reject_prototype" in reasons and _low_damage_similarity(row, config=config):
        decision_type = "reject"
        final_label = "reject"
        matched_rule = "reject_near_reject_prototype_low_damage_similarity"
    elif reliability >= config.accept_threshold_for(final_label) and "high_consensus" in reasons and not has_conflict:
        decision_type = "auto_accept"
        matched_rule = "auto_accept_high_consensus"
    elif _can_relabel_from_core(initial_label, nearest_core_class, nearest_core_similarity, prototype_class, prototype_similarity, prototype_margin, config=config):
        decision_type = "relabel_candidate"
        final_label = str(nearest_core_class)
        reasons.add("core_relabel_candidate")
        matched_rule = "relabel_candidate_core_margin"
    elif reliability < config.suspect_threshold_for(final_label) or has_conflict:
        decision_type = "suspect"
        matched_rule = "suspect_low_reliability_or_conflict"
    else:
        allow_low_priority = config.allow_low_priority_for(final_label)
        decision_type = "auto_accept_low_priority" if allow_low_priority else "suspect"
        matched_rule = "auto_accept_low_priority" if allow_low_priority else "suspect_low_priority_disabled"
        if decision_type == "auto_accept_low_priority":
            reasons.add("low_priority_accept")
    components["decision_policy_v1"] = {
        "matched_rule": matched_rule,
        "thresholds": config.thresholds,
        "previous_decision_type": previous_decision_type,
    }
    return PolicyDecision(
        result_id=result_id,
        initial_label=initial_label,
        suggested_label=suggested_label,
        final_label=final_label,
        previous_decision_type=previous_decision_type,
        final_decision_type=decision_type,
        matched_rule=matched_rule,
        reliability_score=reliability,
        reason_codes=sorted(reasons),
        score_components=components,
    )


def _read_policy_inputs(conn: sqlite3.Connection, *, run_id: str, reliability_run_id: str | None, view_name: str) -> list[sqlite3.Row]:
    clauses = ["d.run_id = ?", "cv.view_name = ?"]
    params: list[object] = [run_id, view_name]
    if reliability_run_id:
        clauses.append("COALESCE(r.reliability_run_id, '') = ?")
        params.append(reliability_run_id)
    return conn.execute(
        f"""
        SELECT d.run_id, d.result_id, d.initial_label, d.suggested_label, d.final_label,
               d.decision_type, d.reliability_score, d.nearest_core_class, d.nearest_core_similarity,
               d.prototype_class, d.prototype_similarity, d.reason_codes_json, d.score_components_json,
               cv.image_rel_path, cv.crop_path, cv.x1, cv.y1, cv.x2, cv.y2,
               r.reliability_run_id
        FROM semantic_decisions d
        LEFT JOIN reliability_scores r ON r.run_id = d.run_id AND r.result_id = d.result_id
        JOIN crop_views cv ON cv.run_id = d.run_id AND cv.result_id = d.result_id
        WHERE {' AND '.join(clauses)}
        ORDER BY d.result_id
        """,
        params,
    ).fetchall()


def _resolve_reliability_run(conn: sqlite3.Connection, *, run_id: str, reliability_run_id: str | None) -> str | None:
    if reliability_run_id in (None, "", "none"):
        return None
    if reliability_run_id != "latest":
        return str(reliability_run_id)
    row = conn.execute(
        "SELECT reliability_run_id FROM reliability_scoring_runs WHERE run_id = ? ORDER BY created_at_utc DESC, reliability_run_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return str(row["reliability_run_id"]) if row is not None else None


def _resolve_taxonomy(conn: sqlite3.Connection, *, run_id: str) -> LabelTaxonomy:
    row = conn.execute("SELECT taxonomy_version_id FROM resemi_runs WHERE run_id = ?", (run_id,)).fetchone()
    version_id = str(row["taxonomy_version_id"] or "label_taxonomy_v1") if row is not None else "label_taxonomy_v1"
    return build_label_taxonomy(version_id=version_id or "label_taxonomy_v1")


def _read_box_decisions(conn: sqlite3.Connection, *, run_id: str) -> dict[int, dict[str, object]]:
    graph = conn.execute("SELECT box_graph_run_id FROM box_graph_runs WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1", (run_id,)).fetchone()
    if graph is None:
        return {}
    rows = conn.execute(
        "SELECT result_id, decision_type, keep_for_cleaned, box_quality_score FROM box_cleanup_decisions WHERE box_graph_run_id = ?",
        (str(graph["box_graph_run_id"]),),
    ).fetchall()
    return {int(row["result_id"]): dict(row) for row in rows}


def _has_conflict(reasons: set[str]) -> bool:
    conflict_codes = {
        "openclip_core_disagree",
        "prototype_label_disagree",
        "near_reject_prototype",
        "geometry_conflict",
        "multi_crop_inconsistent",
        "rare_cluster",
        "noise_cluster",
        "insufficient_class_samples",
    }
    return bool(reasons & conflict_codes)


def _low_damage_similarity(row: sqlite3.Row, *, config: DecisionPolicyConfig) -> bool:
    damage_labels = {"crack", "mold", "spall"}
    prototype_class = str(row["prototype_class"] or "")
    prototype_similarity = row["prototype_similarity"]
    nearest_core_similarity = row["nearest_core_similarity"]
    min_sim = config.min_sim_for(row["final_label"])
    prototype_low = prototype_class not in damage_labels or prototype_similarity is None or float(prototype_similarity) < min_sim
    core_low = nearest_core_similarity is None or float(nearest_core_similarity) < min_sim
    return prototype_low and core_low


def _can_relabel_from_core(
    initial_label: str,
    nearest_core_class: object,
    nearest_core_similarity: object,
    prototype_class: object,
    prototype_similarity: object,
    prototype_margin: float,
    *,
    config: DecisionPolicyConfig,
) -> bool:
    if nearest_core_class is None or nearest_core_similarity is None:
        return False
    core_label = str(nearest_core_class)
    if not core_label or core_label == initial_label or core_label == "reject":
        return False
    core_sim = float(nearest_core_similarity)
    min_sim = config.min_sim_for(core_label)
    if core_sim < min_sim:
        return False
    if prototype_class is not None and str(prototype_class) == initial_label and prototype_similarity is not None:
        return (core_sim - float(prototype_similarity)) >= config.relabel_margin
    return prototype_margin >= config.relabel_margin or core_sim >= (min_sim + config.relabel_margin)


def _update_counts(conn: sqlite3.Connection, *, run_id: str) -> None:
    total = int(conn.execute("SELECT COUNT(*) FROM semantic_decisions WHERE run_id = ?", (run_id,)).fetchone()[0])
    cleaned = int(conn.execute("SELECT COUNT(*) FROM cleaned_labels WHERE run_id = ?", (run_id,)).fetchone()[0])
    suspect = int(conn.execute("SELECT COUNT(*) FROM review_queue WHERE run_id = ? AND queue_type != 'reject'", (run_id,)).fetchone()[0])
    reject = int(conn.execute("SELECT COUNT(*) FROM review_queue WHERE run_id = ? AND queue_type = 'reject'", (run_id,)).fetchone()[0])
    conn.execute(
        "UPDATE resemi_runs SET total_detections = ?, cleaned_count = ?, suspect_count = ?, reject_count = ? WHERE run_id = ?",
        (total, cleaned, suspect, reject, run_id),
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
