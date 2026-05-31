from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from shared.taxonomy.label_taxonomy import LabelTaxonomy, build_label_taxonomy


@dataclass(frozen=True)
class SelfTrainingConfig:
    round_index: int = 1
    classifier_confidence_threshold: float = 0.90
    classifier_margin_threshold: float = 0.10
    prototype_min_similarity: float = 0.70
    core_min_similarity: float = 0.70
    consistency_min_score: float = 0.80
    include_low_priority: bool = False
    apply_promotions: bool = False

    @property
    def options(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "classifier_confidence_threshold": self.classifier_confidence_threshold,
            "classifier_margin_threshold": self.classifier_margin_threshold,
            "prototype_min_similarity": self.prototype_min_similarity,
            "core_min_similarity": self.core_min_similarity,
            "consistency_min_score": self.consistency_min_score,
            "include_low_priority": self.include_low_priority,
            "apply_promotions": self.apply_promotions,
        }


@dataclass(frozen=True)
class PromotionDecision:
    result_id: int
    previous_label: str
    predicted_label: str
    action: str
    classifier_confidence: float
    classifier_margin: float
    prototype_class: str | None
    prototype_similarity: float | None
    nearest_core_class: str | None
    nearest_core_similarity: float | None
    geometry_decision: str | None
    consistency_score: float | None
    reason_codes: list[str]
    evidence: dict[str, object]

    @property
    def reason_codes_json(self) -> str:
        return json.dumps(sorted(set(self.reason_codes)), ensure_ascii=False, sort_keys=True)

    @property
    def evidence_json(self) -> str:
        return json.dumps(self.evidence, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class SelfTrainingResult:
    self_training_run_id: str
    run_id: str
    classifier_run_id: str
    config: SelfTrainingConfig
    promotions: list[PromotionDecision]
    cleaned_rows: list[tuple]

    @property
    def candidate_count(self) -> int:
        return len(self.promotions)

    @property
    def promoted_count(self) -> int:
        return sum(1 for item in self.promotions if item.action == "promote_clean")

    @property
    def rejected_count(self) -> int:
        return sum(1 for item in self.promotions if item.action == "reject_candidate")

    @property
    def deferred_count(self) -> int:
        return sum(1 for item in self.promotions if item.action == "defer_review")


def run_self_training(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    classifier_run_id: str,
    config: SelfTrainingConfig,
    taxonomy: LabelTaxonomy | None = None,
) -> SelfTrainingResult:
    taxonomy = taxonomy or _resolve_taxonomy(conn, run_id=run_id)
    classifier = conn.execute("SELECT * FROM classifier_runs WHERE classifier_run_id = ? AND run_id = ?", (classifier_run_id, run_id)).fetchone()
    if classifier is None:
        raise RuntimeError(f"Classifier run not found for run_id={run_id}: {classifier_run_id}")
    rows = _read_candidates(conn, run_id=run_id, classifier_run_id=classifier_run_id, include_low_priority=config.include_low_priority)
    box = _read_box_decisions(conn, run_id=run_id)
    consistency = _read_consistency(conn, run_id=run_id)
    decisions: list[PromotionDecision] = []
    cleaned_rows: list[tuple] = []
    self_training_run_id = f"selftrain_{run_id}_{classifier_run_id[:12]}_r{int(config.round_index)}"
    for row in rows:
        item = _decide_candidate(row, box=box.get(int(row["result_id"])), consistency=consistency.get(int(row["result_id"])), config=config)
        decisions.append(item)
        if item.action == "promote_clean":
            cleaned_rows.append(
                (
                    run_id,
                    item.result_id,
                    str(row["image_rel_path"] or ""),
                    row["crop_path"],
                    item.predicted_label,
                    taxonomy.export_label(item.predicted_label),
                    "self_training_promote",
                    float(row["reliability_score"] or 0.0),
                    item.reason_codes_json,
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                    self_training_run_id,
                    self_training_run_id,
                )
            )
    return SelfTrainingResult(
        self_training_run_id=self_training_run_id,
        run_id=run_id,
        classifier_run_id=classifier_run_id,
        config=config,
        promotions=decisions,
        cleaned_rows=cleaned_rows,
    )


def persist_self_training_result(conn: sqlite3.Connection, result: SelfTrainingResult, *, created_at_utc: str) -> None:
    conn.execute("DELETE FROM self_training_promotions WHERE self_training_run_id = ?", (result.self_training_run_id,))
    conn.execute("DELETE FROM self_training_runs WHERE self_training_run_id = ?", (result.self_training_run_id,))
    conn.execute(
        """
        INSERT INTO self_training_runs (
            self_training_run_id, run_id, classifier_run_id, created_at_utc,
            round_index, options_json, candidate_count, promoted_count,
            rejected_count, deferred_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.self_training_run_id,
            result.run_id,
            result.classifier_run_id,
            created_at_utc,
            int(result.config.round_index),
            json.dumps(result.config.options, ensure_ascii=False, sort_keys=True),
            result.candidate_count,
            result.promoted_count,
            result.rejected_count,
            result.deferred_count,
        ),
    )
    conn.executemany(
        """
        INSERT INTO self_training_promotions (
            self_training_run_id, result_id, previous_label, predicted_label,
            action, classifier_confidence, classifier_margin, prototype_class,
            prototype_similarity, nearest_core_class, nearest_core_similarity,
            geometry_decision, consistency_score, reason_codes_json, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                result.self_training_run_id,
                item.result_id,
                item.previous_label,
                item.predicted_label,
                item.action,
                item.classifier_confidence,
                item.classifier_margin,
                item.prototype_class,
                item.prototype_similarity,
                item.nearest_core_class,
                item.nearest_core_similarity,
                item.geometry_decision,
                item.consistency_score,
                item.reason_codes_json,
                item.evidence_json,
            )
            for item in result.promotions
        ],
    )
    if result.config.apply_promotions and result.cleaned_rows:
        conn.executemany(
            """
            UPDATE semantic_decisions
            SET final_label = ?, suggested_label = ?, decision_type = 'self_training_promote',
                self_training_run_id = ?, reason_codes_json = ?
            WHERE run_id = ? AND result_id = ?
            """,
            [
                (
                    item.predicted_label,
                    item.predicted_label,
                    result.self_training_run_id,
                    item.reason_codes_json,
                    result.run_id,
                    item.result_id,
                )
                for item in result.promotions
                if item.action == "promote_clean"
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO cleaned_labels (
                run_id, result_id, image_rel_path, crop_path, final_label, export_label,
                decision_type, reliability_score, reason_codes_json, x1, y1, x2, y2,
                decision_policy_run_id, self_training_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            result.cleaned_rows,
        )
        conn.executemany(
            "DELETE FROM review_queue WHERE run_id = ? AND result_id = ?",
            [(result.run_id, item.result_id) for item in result.promotions if item.action == "promote_clean"],
        )
        _update_counts(conn, run_id=result.run_id)
    conn.commit()


def _read_candidates(conn: sqlite3.Connection, *, run_id: str, classifier_run_id: str, include_low_priority: bool) -> list[sqlite3.Row]:
    decision_types = ["suspect", "relabel_candidate"]
    if include_low_priority:
        decision_types.append("auto_accept_low_priority")
    placeholders = ", ".join("?" for _ in decision_types)
    return conn.execute(
        f"""
        SELECT d.result_id, d.initial_label, d.suggested_label, d.final_label,
               d.decision_type, d.reliability_score, d.prototype_class, d.prototype_similarity,
               d.nearest_core_class, d.nearest_core_similarity, d.reason_codes_json,
               p.predicted_label, p.predicted_probability, p.second_label,
               p.second_probability, p.margin AS classifier_margin,
               cv.image_rel_path, cv.crop_path, cv.x1, cv.y1, cv.x2, cv.y2
        FROM semantic_decisions d
        JOIN classifier_prediction_summary p ON p.result_id = d.result_id AND p.classifier_run_id = ?
        LEFT JOIN crop_views cv ON cv.run_id = d.run_id AND cv.result_id = d.result_id AND cv.view_name = 'tight'
        WHERE d.run_id = ? AND d.decision_type IN ({placeholders})
          AND d.final_label NOT IN ('reject', 'unknown', 'background', 'shadow', 'edge', 'object')
        ORDER BY p.predicted_probability DESC, p.margin DESC, d.result_id
        """,
        [classifier_run_id, run_id, *decision_types],
    ).fetchall()


def _decide_candidate(row: sqlite3.Row, *, box: dict[str, object] | None, consistency: dict[str, object] | None, config: SelfTrainingConfig) -> PromotionDecision:
    reasons = set(_parse_json_list(row["reason_codes_json"]))
    predicted_label = str(row["predicted_label"])
    previous_label = str(row["final_label"])
    confidence = float(row["predicted_probability"])
    margin = float(row["classifier_margin"])
    prototype_class = row["prototype_class"] if row["prototype_class"] is not None else None
    prototype_similarity = float(row["prototype_similarity"]) if row["prototype_similarity"] is not None else None
    core_class = row["nearest_core_class"] if row["nearest_core_class"] is not None else None
    core_similarity = float(row["nearest_core_similarity"]) if row["nearest_core_similarity"] is not None else None
    geometry_decision = str(box.get("decision_type")) if box and box.get("decision_type") is not None else None
    consistency_score = float(consistency["label_consistency"]) if consistency and consistency.get("label_consistency") is not None else None
    evidence = {
        "classifier_second_label": row["second_label"],
        "classifier_second_probability": row["second_probability"],
        "source_decision_type": row["decision_type"],
        "box_decision": geometry_decision,
        "consistency_status": consistency.get("status") if consistency else None,
    }
    promotion_reasons: list[str] = []
    blockers: list[str] = []
    if confidence >= config.classifier_confidence_threshold:
        promotion_reasons.append("classifier_confidence_high")
    else:
        blockers.append("classifier_confidence_low")
    if margin >= config.classifier_margin_threshold:
        promotion_reasons.append("classifier_margin_high")
    else:
        blockers.append("classifier_margin_low")
    prototype_match = prototype_class == predicted_label and prototype_similarity is not None and prototype_similarity >= config.prototype_min_similarity
    core_match = core_class == predicted_label and core_similarity is not None and core_similarity >= config.core_min_similarity
    if prototype_match:
        promotion_reasons.append("prototype_agrees_with_classifier")
    if core_match:
        promotion_reasons.append("core_agrees_with_classifier")
    if not prototype_match and not core_match:
        blockers.append("missing_core_or_prototype_agreement")
    if geometry_decision in {"suspect_composite_box", "suspect_broad_box", "manual_box_review"} or (box and int(box.get("keep_for_cleaned", 1)) == 0):
        blockers.append("geometry_conflict")
    if "near_reject_prototype" in reasons:
        blockers.append("near_reject_prototype")
    if consistency_score is not None:
        if consistency_score >= config.consistency_min_score:
            promotion_reasons.append("multi_crop_consistent")
        else:
            blockers.append("multi_crop_inconsistent")
    else:
        promotion_reasons.append("multi_crop_consistency_missing_not_blocking")
    action = "promote_clean" if not blockers else "defer_review"
    all_reasons = [*promotion_reasons, *blockers]
    return PromotionDecision(
        result_id=int(row["result_id"]),
        previous_label=previous_label,
        predicted_label=predicted_label,
        action=action,
        classifier_confidence=confidence,
        classifier_margin=margin,
        prototype_class=str(prototype_class) if prototype_class is not None else None,
        prototype_similarity=prototype_similarity,
        nearest_core_class=str(core_class) if core_class is not None else None,
        nearest_core_similarity=core_similarity,
        geometry_decision=geometry_decision,
        consistency_score=consistency_score,
        reason_codes=all_reasons,
        evidence=evidence,
    )


def _read_box_decisions(conn: sqlite3.Connection, *, run_id: str) -> dict[int, dict[str, object]]:
    graph = conn.execute("SELECT box_graph_run_id FROM box_graph_runs WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1", (run_id,)).fetchone()
    if graph is None:
        return {}
    rows = conn.execute(
        "SELECT result_id, decision_type, keep_for_cleaned, box_quality_score FROM box_cleanup_decisions WHERE box_graph_run_id = ?",
        (str(graph["box_graph_run_id"]),),
    ).fetchall()
    return {int(row["result_id"]): dict(row) for row in rows}


def _read_consistency(conn: sqlite3.Connection, *, run_id: str) -> dict[int, dict[str, object]]:
    rows = conn.execute("SELECT result_id, label_consistency, status FROM crop_consistency_features WHERE run_id = ?", (run_id,)).fetchall()
    return {int(row["result_id"]): dict(row) for row in rows}


def _resolve_taxonomy(conn: sqlite3.Connection, *, run_id: str) -> LabelTaxonomy:
    row = conn.execute("SELECT taxonomy_version_id FROM resemi_runs WHERE run_id = ?", (run_id,)).fetchone()
    version_id = str(row["taxonomy_version_id"] or "label_taxonomy_v1") if row is not None else "label_taxonomy_v1"
    return build_label_taxonomy(version_id=version_id or "label_taxonomy_v1")


def _update_counts(conn: sqlite3.Connection, *, run_id: str) -> None:
    total = int(conn.execute("SELECT COUNT(*) FROM semantic_decisions WHERE run_id = ?", (run_id,)).fetchone()[0])
    cleaned = int(conn.execute("SELECT COUNT(*) FROM cleaned_labels WHERE run_id = ?", (run_id,)).fetchone()[0])
    suspect = int(conn.execute("SELECT COUNT(*) FROM review_queue WHERE run_id = ? AND queue_type != 'reject'", (run_id,)).fetchone()[0])
    reject = int(conn.execute("SELECT COUNT(*) FROM review_queue WHERE run_id = ? AND queue_type = 'reject'", (run_id,)).fetchone()[0])
    conn.execute("UPDATE resemi_runs SET total_detections = ?, cleaned_count = ?, suspect_count = ?, reject_count = ? WHERE run_id = ?", (total, cleaned, suspect, reject, run_id))


def _parse_json_list(raw: object) -> list[str]:
    try:
        value = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []
