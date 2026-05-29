from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bbox_quality import BBoxQualityResult, BoxCleanupDecision, run_bbox_quality_filter
from .crop_generation import CropView, CropViewSpec, DEFAULT_VIEW_SPECS, generate_crop_views
from .decision_policy import DecisionConfig, SemanticDecision, decide
from .label_taxonomy import LabelTaxonomy, build_label_taxonomy
from .schema import connect_output, utc_now
from .semantic_ensemble import SemanticEnsembleResult, build_semantic_ensemble
from .source_store import (
    SemanticRunMetadata,
    SourceDetection,
    connect_readonly,
    read_kept_result_ids,
    read_semantic_run_metadata,
    read_source_detections,
    resolve_dedup_run_id,
    resolve_semantic_run_id,
)


@dataclass(frozen=True)
class ResemiConfig:
    source_db: Path
    output_db: Path
    semantic_run_id: str = "latest"
    dedup_db: Path | None = None
    dedup_run_id: str = "latest"
    labels: tuple[str, ...] = ()
    limit: int = 0
    run_id: str = ""
    accept_threshold: float = 0.75
    suspect_threshold: float = 0.50
    low_margin_threshold: float = 0.03
    strong_margin_threshold: float = 0.10
    prototype_version_id: str = ""
    export_dir: Path | None = None
    image_root: Path | None = None
    crop_dir: Path | None = None
    crop_view_specs: tuple[CropViewSpec, ...] = DEFAULT_VIEW_SPECS
    generate_crops: bool = True
    taxonomy_version_id: str = "label_taxonomy_v1"
    stain_export_label: str = "stain"


@dataclass(frozen=True)
class ResemiSummary:
    run_id: str
    output_db: Path
    source_semantic_run_id: str
    total_detections: int
    cleaned_count: int
    suspect_count: int
    reject_count: int
    export_dir: Path
    review_csv: Path
    review_json: Path
    cleaned_csv: Path
    cleaned_json: Path
    box_cleanup_csv: Path
    box_cleanup_json: Path
    box_review_csv: Path
    box_review_json: Path
    semantic_ensemble_csv: Path
    semantic_ensemble_json: Path
    label_taxonomy_json: Path


class ResemiPipeline:
    def __init__(self, config: ResemiConfig) -> None:
        self.config = config

    def run(self) -> ResemiSummary:
        source_db = Path(self.config.source_db).expanduser().resolve()
        output_db = Path(self.config.output_db).expanduser().resolve()
        run_id = self.config.run_id.strip() or f"resemi_{uuid.uuid4().hex[:12]}"

        source_conn = connect_readonly(source_db)
        dedup_conn: sqlite3.Connection | None = None
        output_conn = connect_output(output_db)
        try:
            semantic_run_id = resolve_semantic_run_id(source_conn, self.config.semantic_run_id)
            semantic_metadata = read_semantic_run_metadata(source_conn, semantic_run_id)
            dedup_run_id: str | None = None
            kept_result_ids: set[int] | None = None
            if self.config.dedup_db is not None:
                dedup_conn = connect_readonly(Path(self.config.dedup_db))
                dedup_run_id = resolve_dedup_run_id(dedup_conn, self.config.dedup_run_id)
                kept_result_ids = read_kept_result_ids(dedup_conn, dedup_run_id=dedup_run_id)

            detections = read_source_detections(
                source_conn,
                semantic_run_id=semantic_run_id,
                labels=self.config.labels,
                kept_result_ids=kept_result_ids,
                limit=self.config.limit,
            )
            decision_config = DecisionConfig(
                accept_threshold=self.config.accept_threshold,
                suspect_threshold=self.config.suspect_threshold,
                low_margin_threshold=self.config.low_margin_threshold,
                strong_margin_threshold=self.config.strong_margin_threshold,
                labels=self.config.labels or DecisionConfig().labels,
            )
            taxonomy = build_label_taxonomy(
                version_id=self.config.taxonomy_version_id,
                stain_export_label=self.config.stain_export_label,
            )
            semantic_ensemble = build_semantic_ensemble(detections)
            agreements_by_id = semantic_ensemble.agreements_by_result_id
            decisions = [decide(item, decision_config, agreements_by_id.get(item.result_id)) for item in detections]
            bbox_result = run_bbox_quality_filter(run_id=run_id, detections=detections, semantic_decisions=decisions)
            crop_dir = self._resolve_crop_dir(run_id, output_db)
            if self.config.generate_crops:
                crop_views, crop_errors = generate_crop_views(
                    detections,
                    image_root=self.config.image_root,
                    crop_dir=crop_dir,
                    view_specs=self.config.crop_view_specs,
                )
            else:
                crop_views = self._step2_crop_views(detections)
                crop_errors = {}

            self._replace_run(output_conn, run_id, semantic_metadata, dedup_run_id)
            self._write_label_taxonomy(output_conn, run_id, taxonomy)
            self._write_crop_views(output_conn, run_id, crop_views)
            self._write_crop_consistency_features(output_conn, run_id, detections, crop_views, crop_errors)
            self._write_scores(output_conn, run_id, detections)
            self._write_semantic_ensemble(output_conn, run_id, semantic_ensemble)
            self._write_bbox_quality_result(output_conn, bbox_result)
            self._write_decisions(output_conn, run_id, detections, decisions, bbox_result.decisions_by_result_id, taxonomy)
            cleaned_count, suspect_count, reject_count = self._finalize_counts(output_conn, run_id)
            export_paths = self._export_artifacts(output_conn, run_id, output_db)
        finally:
            source_conn.close()
            if dedup_conn is not None:
                dedup_conn.close()
            output_conn.close()

        return ResemiSummary(
            run_id=run_id,
            output_db=output_db,
            source_semantic_run_id=semantic_run_id,
            total_detections=len(detections),
            cleaned_count=cleaned_count,
            suspect_count=suspect_count,
            reject_count=reject_count,
            export_dir=export_paths["export_dir"],
            review_csv=export_paths["review_csv"],
            review_json=export_paths["review_json"],
            cleaned_csv=export_paths["cleaned_csv"],
            cleaned_json=export_paths["cleaned_json"],
            box_cleanup_csv=export_paths["box_cleanup_csv"],
            box_cleanup_json=export_paths["box_cleanup_json"],
            box_review_csv=export_paths["box_review_csv"],
            box_review_json=export_paths["box_review_json"],
            semantic_ensemble_csv=export_paths["semantic_ensemble_csv"],
            semantic_ensemble_json=export_paths["semantic_ensemble_json"],
            label_taxonomy_json=export_paths["label_taxonomy_json"],
        )

    def _replace_run(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        semantic_metadata: SemanticRunMetadata,
        dedup_run_id: str | None,
    ) -> None:
        conn.execute(
            "DELETE FROM review_decisions WHERE review_session_id IN (SELECT review_session_id FROM review_sessions WHERE run_id = ?)",
            (run_id,),
        )
        conn.execute(
            "DELETE FROM classifier_predictions WHERE classifier_run_id IN (SELECT classifier_run_id FROM classifier_runs WHERE run_id = ?)",
            (run_id,),
        )
        conn.execute(
            "DELETE FROM crop_embeddings WHERE embedding_run_id IN (SELECT embedding_run_id FROM embedding_runs WHERE run_id = ?)",
            (run_id,),
        )
        conn.execute(
            "DELETE FROM prototype_items WHERE prototype_version_id IN (SELECT prototype_version_id FROM prototype_versions WHERE run_id = ?)",
            (run_id,),
        )
        conn.execute(
            "DELETE FROM box_review_queue WHERE box_graph_run_id IN (SELECT box_graph_run_id FROM box_graph_runs WHERE run_id = ?)",
            (run_id,),
        )
        conn.execute(
            "DELETE FROM box_cleanup_decisions WHERE box_graph_run_id IN (SELECT box_graph_run_id FROM box_graph_runs WHERE run_id = ?)",
            (run_id,),
        )
        conn.execute(
            "DELETE FROM box_quality_scores WHERE box_graph_run_id IN (SELECT box_graph_run_id FROM box_graph_runs WHERE run_id = ?)",
            (run_id,),
        )
        conn.execute(
            "DELETE FROM box_graph_edges WHERE box_graph_run_id IN (SELECT box_graph_run_id FROM box_graph_runs WHERE run_id = ?)",
            (run_id,),
        )
        for table in (
            "classifier_runs",
            "review_sessions",
            "embedding_runs",
            "box_graph_runs",
            "core_cluster_members",
            "core_clusters",
            "prototype_versions",
            "review_queue",
            "cleaned_labels",
            "label_taxonomy_versions",
            "semantic_decisions",
            "reliability_scores",
            "semantic_agreements",
            "semantic_model_outputs",
            "semantic_model_scores",
            "crop_consistency_features",
            "crop_views",
        ):
            conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM resemi_runs WHERE run_id = ?", (run_id,))
        options = {
            "labels": list(self.config.labels),
            "limit": int(self.config.limit),
            "export_dir": str(self._resolve_export_dir(run_id, Path(self.config.output_db).expanduser().resolve())),
            "image_root": str(Path(self.config.image_root).expanduser().resolve()) if self.config.image_root is not None else "",
            "crop_dir": str(self._resolve_crop_dir(run_id, Path(self.config.output_db).expanduser().resolve())),
            "crop_views": [item.name for item in self.config.crop_view_specs],
            "generate_crops": bool(self.config.generate_crops),
            "prototype_version_id": self.config.prototype_version_id,
            "taxonomy_version_id": self.config.taxonomy_version_id,
            "stain_export_label": self.config.stain_export_label,
        }
        thresholds = {
            "accept_threshold": float(self.config.accept_threshold),
            "suspect_threshold": float(self.config.suspect_threshold),
            "low_margin_threshold": float(self.config.low_margin_threshold),
            "strong_margin_threshold": float(self.config.strong_margin_threshold),
        }
        model_versions = {
            "semantic_models": [
                {
                    "name": "openclip",
                    "model_name": semantic_metadata.model_name,
                    "pretrained": semantic_metadata.pretrained,
                    "device": semantic_metadata.device,
                    "source_semantic_run_id": semantic_metadata.semantic_run_id,
                    "source_created_at_utc": semantic_metadata.created_at_utc,
                    "source_stage": semantic_metadata.source_stage,
                }
            ],
            "semantic_vote_sources": ["openclip_step2", "groundingdino_prompt"],
            "decision_policy": "resemi_baseline_v0",
            "embedding_models": [],
            "prototype_versions": [self.config.prototype_version_id] if self.config.prototype_version_id else [],
        }
        prompt_set = _json_or_raw(semantic_metadata.prompt_config_json)
        conn.execute(
            """
            INSERT INTO resemi_runs (
                run_id, created_at_utc, source_db_path, source_semantic_run_id,
                source_dedup_db_path, source_dedup_run_id, options_json,
                model_versions_json, thresholds_json, prompt_set_json, prototype_version_id, taxonomy_version_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                utc_now(),
                str(Path(self.config.source_db).expanduser().resolve()),
                semantic_metadata.semantic_run_id,
                str(Path(self.config.dedup_db).expanduser().resolve()) if self.config.dedup_db is not None else None,
                dedup_run_id,
                json.dumps(options, ensure_ascii=False, sort_keys=True),
                json.dumps(model_versions, ensure_ascii=False, sort_keys=True),
                json.dumps(thresholds, ensure_ascii=False, sort_keys=True),
                json.dumps(prompt_set, ensure_ascii=False, sort_keys=True),
                self.config.prototype_version_id or None,
                self.config.taxonomy_version_id,
            ),
        )
        conn.commit()

    @staticmethod
    def _write_label_taxonomy(conn: sqlite3.Connection, run_id: str, taxonomy: LabelTaxonomy) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO label_taxonomy_versions (
                taxonomy_version_id, run_id, created_at_utc, working_labels_json,
                damage_labels_json, reject_labels_json, export_mapping_json, guidelines_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                taxonomy.version_id,
                run_id,
                utc_now(),
                taxonomy.working_labels_json,
                taxonomy.damage_labels_json,
                taxonomy.reject_labels_json,
                taxonomy.export_mapping_json,
                taxonomy.guidelines_json,
            ),
        )
        conn.commit()

    @staticmethod
    def _write_crop_views(conn: sqlite3.Connection, run_id: str, crop_views: list[CropView]) -> None:
        rows = [
            (
                run_id,
                view.result_id,
                view.view_name,
                view.crop_path,
                view.image_rel_path,
                view.x1,
                view.y1,
                view.x2,
                view.y2,
                view.source,
                view.crop_hash,
                view.crop_width,
                view.crop_height,
                view.padding_ratio,
                view.status,
                view.error_message,
            )
            for view in crop_views
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO crop_views (
                run_id, result_id, view_name, crop_path, image_rel_path, x1, y1, x2, y2, source,
                crop_hash, crop_width, crop_height, padding_ratio, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    @staticmethod
    def _write_crop_consistency_features(
        conn: sqlite3.Connection,
        run_id: str,
        detections: list[SourceDetection],
        crop_views: list[CropView],
        crop_errors: dict[int, str],
    ) -> None:
        views_by_result: dict[int, list[CropView]] = {}
        for view in crop_views:
            views_by_result.setdefault(int(view.result_id), []).append(view)
        now = utc_now()
        rows = []
        for detection in detections:
            views = views_by_result.get(detection.result_id, [])
            error = crop_errors.get(detection.result_id)
            status = "crop_generation_error" if error else "semantic_scores_pending"
            features = {
                "available_views": [view.view_name for view in sorted(views, key=lambda item: item.padding_ratio)],
                "crop_hashes": {view.view_name: view.crop_hash for view in views},
                "error": error or "",
                "note": "Consistency labels require per-view semantic scoring in SPEC 04.",
            }
            rows.append(
                (
                    run_id,
                    detection.result_id,
                    None,
                    None,
                    None,
                    None,
                    status,
                    json.dumps(features, ensure_ascii=False, sort_keys=True),
                    now,
                )
            )
        conn.executemany(
            """
            INSERT OR REPLACE INTO crop_consistency_features (
                run_id, result_id, label_consistency, score_variance, view_disagreement_count,
                context_shift_label, status, features_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    @staticmethod
    def _step2_crop_views(detections: list[SourceDetection]) -> list[CropView]:
        return [
            CropView(
                result_id=item.result_id,
                view_name="openclip_crop",
                crop_path=item.crop_path,
                image_rel_path=item.image_rel_path,
                x1=item.x1,
                y1=item.y1,
                x2=item.x2,
                y2=item.y2,
                source="step2_sematic",
                crop_hash="",
                crop_width=0,
                crop_height=0,
                padding_ratio=0.0,
            )
            for item in detections
        ]

    @staticmethod
    def _write_scores(conn: sqlite3.Connection, run_id: str, detections: list[SourceDetection]) -> None:
        rows = []
        for item in detections:
            ranked = sorted(item.scores.items(), key=lambda pair: pair[1], reverse=True)
            for rank, (label, probability) in enumerate(ranked, start=1):
                rows.append((run_id, item.result_id, "openclip", str(label), float(probability), rank))
        conn.executemany(
            """
            INSERT OR REPLACE INTO semantic_model_scores (
                run_id, result_id, model_name, label, probability, rank
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    @staticmethod
    def _write_semantic_ensemble(conn: sqlite3.Connection, run_id: str, result: SemanticEnsembleResult) -> None:
        now = utc_now()
        conn.executemany(
            """
            INSERT OR REPLACE INTO semantic_model_outputs (
                run_id, result_id, model_name, source_type, top1_label, top1_score,
                top2_label, top2_score, margin, entropy, raw_scores_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    output.result_id,
                    output.model_name,
                    output.source_type,
                    output.top1_label,
                    output.top1_score,
                    output.top2_label,
                    output.top2_score,
                    output.margin,
                    output.entropy,
                    output.raw_scores_json,
                    now,
                )
                for output in result.outputs
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO semantic_agreements (
                run_id, result_id, majority_label, agreement_ratio, strong_agreement_count,
                conflict_labels_json, sources_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    agreement.result_id,
                    agreement.majority_label,
                    agreement.agreement_ratio,
                    agreement.strong_agreement_count,
                    agreement.conflict_labels_json,
                    agreement.sources_json,
                    now,
                )
                for agreement in result.agreements
            ],
        )
        conn.commit()

    @staticmethod
    def _write_bbox_quality_result(conn: sqlite3.Connection, result: BBoxQualityResult) -> None:
        now = utc_now()
        review_count = len(result.review_decisions)
        conn.execute(
            """
            INSERT OR REPLACE INTO box_graph_runs (
                box_graph_run_id, run_id, created_at_utc, options_json,
                total_boxes, edge_count, decision_count, review_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.box_graph_run_id,
                result.run_id,
                now,
                json.dumps(result.options, ensure_ascii=False, sort_keys=True),
                len(result.quality_scores),
                len(result.edges),
                len(result.decisions),
                review_count,
            ),
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO box_graph_edges (
                box_graph_run_id, parent_result_id, child_result_id, image_rel_path,
                iou, intersection_area, containment_small_in_large, child_coverage_of_parent,
                area_ratio, center_distance_norm, aspect_ratio_similarity, label_agreement,
                edge_type, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    result.box_graph_run_id,
                    edge.parent_result_id,
                    edge.child_result_id,
                    edge.image_rel_path,
                    edge.iou,
                    edge.intersection_area,
                    edge.containment_small_in_large,
                    edge.child_coverage_of_parent,
                    edge.area_ratio,
                    edge.center_distance_norm,
                    edge.aspect_ratio_similarity,
                    1 if edge.label_agreement else 0,
                    edge.edge_type,
                    edge.features_json,
                )
                for edge in result.edges
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO box_quality_scores (
                box_graph_run_id, result_id, image_rel_path, label, box_quality_score,
                detector_score, semantic_confidence, semantic_margin, crop_consistency,
                embedding_core_similarity, prototype_similarity, area_ratio_to_image,
                aspect_ratio, elongation, child_count, child_label_diversity,
                child_alignment_score, background_context_penalty, composite_penalty, components_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    result.box_graph_run_id,
                    score.result_id,
                    score.image_rel_path,
                    score.label,
                    score.box_quality_score,
                    score.detector_score,
                    score.semantic_confidence,
                    score.semantic_margin,
                    score.crop_consistency,
                    score.embedding_core_similarity,
                    score.prototype_similarity,
                    score.area_ratio_to_image,
                    score.aspect_ratio,
                    score.elongation,
                    score.child_count,
                    score.child_label_diversity,
                    score.child_alignment_score,
                    score.background_context_penalty,
                    score.composite_penalty,
                    score.components_json,
                )
                for score in result.quality_scores
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO box_cleanup_decisions (
                box_graph_run_id, result_id, image_rel_path, label, decision_type,
                keep_for_cleaned, box_quality_score, representative_id,
                reason_codes_json, features_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    result.box_graph_run_id,
                    decision.result_id,
                    decision.image_rel_path,
                    decision.label,
                    decision.decision_type,
                    1 if decision.keep_for_cleaned else 0,
                    decision.box_quality_score,
                    decision.representative_id,
                    decision.reason_codes_json,
                    decision.features_json,
                    now,
                )
                for decision in result.decisions
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO box_review_queue (
                box_graph_run_id, result_id, image_rel_path, label, queue_type,
                box_quality_score, reason_codes_json, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    result.box_graph_run_id,
                    decision.result_id,
                    decision.image_rel_path,
                    decision.label,
                    decision.decision_type,
                    decision.box_quality_score,
                    decision.reason_codes_json,
                    decision.features_json,
                )
                for decision in result.review_decisions
            ],
        )
        conn.commit()

    @staticmethod
    def _write_decisions(
        conn: sqlite3.Connection,
        run_id: str,
        detections: list[SourceDetection],
        decisions: list[SemanticDecision],
        box_decisions: dict[int, BoxCleanupDecision],
        taxonomy: LabelTaxonomy,
    ) -> None:
        by_id = {item.result_id: item for item in detections}
        now = utc_now()
        reliability_rows = []
        decision_rows = []
        cleaned_rows = []
        review_by_result_id: dict[int, tuple] = {}
        for decision in decisions:
            detection = by_id[decision.result_id]
            box_decision = box_decisions.get(decision.result_id)
            box_decision_type = box_decision.decision_type if box_decision is not None else "keep_representative"
            box_keep_for_cleaned = True if box_decision is None else box_decision.keep_for_cleaned
            box_drop = box_decision_type == "drop_nested_duplicate"
            box_review = box_decision_type in {"suspect_composite_box", "suspect_broad_box", "manual_box_review"}
            reliability_rows.append(
                (
                    run_id,
                    decision.result_id,
                    decision.reliability_score,
                    decision.model_agreement,
                    decision.top1_top2_margin,
                    None,
                    None,
                    None,
                    None,
                    decision.reason_codes_json,
                    decision.score_components_json,
                    now,
                )
            )
            decision_rows.append(
                (
                    run_id,
                    decision.result_id,
                    decision.initial_label,
                    decision.suggested_label,
                    decision.final_label,
                    decision.decision_type,
                    decision.reliability_score,
                    None,
                    None,
                    None,
                    None,
                    decision.model_agreement,
                    decision.reason_codes_json,
                    decision.score_components_json,
                    now,
                )
            )
            if decision.decision_type == "auto_accept" and box_keep_for_cleaned:
                cleaned_rows.append(
                    (
                        run_id,
                        decision.result_id,
                        detection.image_rel_path,
                        detection.crop_path,
                        decision.final_label,
                        taxonomy.export_label(decision.final_label),
                        decision.decision_type,
                        decision.reliability_score,
                        decision.reason_codes_json,
                        detection.x1,
                        detection.y1,
                        detection.x2,
                        detection.y2,
                    )
                )
            if not box_drop and decision.decision_type in {"suspect", "reject", "relabel_candidate"}:
                review_by_result_id[decision.result_id] = (
                    run_id,
                    decision.result_id,
                    detection.image_rel_path,
                    detection.crop_path,
                    decision.initial_label,
                    decision.suggested_label,
                    decision.decision_type,
                    decision.reliability_score,
                    decision.reason_codes_json,
                )
            if box_review and box_decision is not None:
                combined_reasons = sorted(set([*decision.reason_codes, *box_decision.reason_codes, "bbox_quality_filter"]))
                review_by_result_id[decision.result_id] = (
                    run_id,
                    decision.result_id,
                    detection.image_rel_path,
                    detection.crop_path,
                    decision.initial_label,
                    decision.suggested_label,
                    box_decision.decision_type,
                    min(decision.reliability_score, box_decision.box_quality_score),
                    json.dumps(combined_reasons, ensure_ascii=False, sort_keys=True),
                )

        conn.executemany(
            """
            INSERT OR REPLACE INTO reliability_scores (
                run_id, result_id, reliability_score, model_agreement, top1_top2_margin,
                nearest_core_class, nearest_core_similarity, prototype_class, prototype_similarity,
                reason_codes_json, score_components_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            reliability_rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO semantic_decisions (
                run_id, result_id, initial_label, suggested_label, final_label, decision_type,
                reliability_score, nearest_core_class, nearest_core_similarity, prototype_class,
                prototype_similarity, model_agreement, reason_codes_json, score_components_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            decision_rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO cleaned_labels (
                run_id, result_id, image_rel_path, crop_path, final_label, export_label, decision_type,
                reliability_score, reason_codes_json, x1, y1, x2, y2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            cleaned_rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO review_queue (
                run_id, result_id, image_rel_path, crop_path, initial_label, suggested_label,
                queue_type, reliability_score, reason_codes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            list(review_by_result_id.values()),
        )
        conn.commit()

    @staticmethod
    def _finalize_counts(conn: sqlite3.Connection, run_id: str) -> tuple[int, int, int]:
        total = int(conn.execute("SELECT COUNT(*) FROM semantic_decisions WHERE run_id = ?", (run_id,)).fetchone()[0])
        cleaned = int(conn.execute("SELECT COUNT(*) FROM cleaned_labels WHERE run_id = ?", (run_id,)).fetchone()[0])
        suspect = int(
            conn.execute("SELECT COUNT(*) FROM review_queue WHERE run_id = ? AND queue_type != 'reject'", (run_id,)).fetchone()[0]
        )
        reject = int(
            conn.execute("SELECT COUNT(*) FROM review_queue WHERE run_id = ? AND queue_type = 'reject'", (run_id,)).fetchone()[0]
        )
        conn.execute(
            """
            UPDATE resemi_runs
            SET total_detections = ?, cleaned_count = ?, suspect_count = ?, reject_count = ?
            WHERE run_id = ?
            """,
            (total, cleaned, suspect, reject, run_id),
        )
        conn.commit()
        return cleaned, suspect, reject

    def _export_artifacts(self, conn: sqlite3.Connection, run_id: str, output_db: Path) -> dict[str, Path]:
        export_dir = self._resolve_export_dir(run_id, output_db)
        export_dir.mkdir(parents=True, exist_ok=True)
        review_rows = self._fetch_export_rows(
            conn,
            """
            SELECT q.result_id, q.image_rel_path, q.crop_path, q.initial_label, q.suggested_label,
                   q.queue_type, q.reliability_score, q.reason_codes_json,
                   r.top1_top2_margin, d.score_components_json
            FROM review_queue q
            JOIN semantic_decisions d ON d.run_id = q.run_id AND d.result_id = q.result_id
            JOIN reliability_scores r ON r.run_id = q.run_id AND r.result_id = q.result_id
            WHERE q.run_id = ?
            ORDER BY q.queue_type, q.reliability_score ASC, q.result_id
            """,
            run_id,
        )
        cleaned_rows = self._fetch_export_rows(
            conn,
            """
            SELECT c.result_id, c.image_rel_path, c.crop_path, c.final_label, c.decision_type,
                   c.export_label, c.reliability_score, c.reason_codes_json, c.x1, c.y1, c.x2, c.y2,
                   d.initial_label, d.suggested_label, r.top1_top2_margin, d.score_components_json
            FROM cleaned_labels c
            JOIN semantic_decisions d ON d.run_id = c.run_id AND d.result_id = c.result_id
            JOIN reliability_scores r ON r.run_id = c.run_id AND r.result_id = c.result_id
            WHERE c.run_id = ?
            ORDER BY c.final_label, c.reliability_score DESC, c.result_id
            """,
            run_id,
        )
        summary_rows = self._fetch_export_rows(
            conn,
            """
            SELECT run_id, created_at_utc, source_db_path, source_semantic_run_id,
                   source_dedup_db_path, source_dedup_run_id, options_json,
                   model_versions_json, thresholds_json, prompt_set_json, prototype_version_id, taxonomy_version_id,
                   total_detections, cleaned_count, suspect_count, reject_count
            FROM resemi_runs
            WHERE run_id = ?
            """,
            run_id,
        )
        box_cleanup_rows = self._fetch_export_rows(
            conn,
            """
            SELECT d.result_id, d.image_rel_path, d.label, d.decision_type, d.keep_for_cleaned,
                   d.box_quality_score, d.representative_id, d.reason_codes_json, d.features_json
            FROM box_cleanup_decisions d
            JOIN box_graph_runs r ON r.box_graph_run_id = d.box_graph_run_id
            WHERE r.run_id = ?
            ORDER BY d.decision_type, d.box_quality_score ASC, d.result_id
            """,
            run_id,
        )
        box_review_rows = self._fetch_export_rows(
            conn,
            """
            SELECT q.result_id, q.image_rel_path, q.label, q.queue_type,
                   q.box_quality_score, q.reason_codes_json, q.features_json
            FROM box_review_queue q
            JOIN box_graph_runs r ON r.box_graph_run_id = q.box_graph_run_id
            WHERE r.run_id = ?
            ORDER BY q.queue_type, q.box_quality_score ASC, q.result_id
            """,
            run_id,
        )
        semantic_ensemble_rows = self._fetch_export_rows(
            conn,
            """
            SELECT a.result_id, a.majority_label, a.agreement_ratio, a.strong_agreement_count,
                   a.conflict_labels_json, a.sources_json,
                   d.final_label, d.decision_type, d.reliability_score
            FROM semantic_agreements a
            JOIN semantic_decisions d ON d.run_id = a.run_id AND d.result_id = a.result_id
            WHERE a.run_id = ?
            ORDER BY a.agreement_ratio ASC, d.reliability_score ASC, a.result_id
            """,
            run_id,
        )
        taxonomy_rows = self._fetch_export_rows(
            conn,
            """
            SELECT taxonomy_version_id, working_labels_json, damage_labels_json,
                   reject_labels_json, export_mapping_json, guidelines_json
            FROM label_taxonomy_versions
            WHERE run_id = ?
            """,
            run_id,
        )

        review_csv = export_dir / "review_queue.csv"
        review_json = export_dir / "review_queue.json"
        cleaned_csv = export_dir / "cleaned_labels.csv"
        cleaned_json = export_dir / "cleaned_labels.json"
        box_cleanup_csv = export_dir / "box_cleanup_decisions.csv"
        box_cleanup_json = export_dir / "box_cleanup_decisions.json"
        box_review_csv = export_dir / "box_review_queue.csv"
        box_review_json = export_dir / "box_review_queue.json"
        semantic_ensemble_csv = export_dir / "semantic_ensemble.csv"
        semantic_ensemble_json = export_dir / "semantic_ensemble.json"
        label_taxonomy_json = export_dir / "label_taxonomy.json"
        summary_json = export_dir / "run_summary.json"
        self._write_csv(review_csv, review_rows)
        self._write_json(review_json, review_rows)
        self._write_csv(cleaned_csv, cleaned_rows)
        self._write_json(cleaned_json, cleaned_rows)
        self._write_csv(box_cleanup_csv, box_cleanup_rows)
        self._write_json(box_cleanup_json, box_cleanup_rows)
        self._write_csv(box_review_csv, box_review_rows)
        self._write_json(box_review_json, box_review_rows)
        self._write_csv(semantic_ensemble_csv, semantic_ensemble_rows)
        self._write_json(semantic_ensemble_json, semantic_ensemble_rows)
        self._write_json(label_taxonomy_json, taxonomy_rows[0] if taxonomy_rows else {})
        self._write_json(summary_json, summary_rows[0] if summary_rows else {})
        return {
            "export_dir": export_dir,
            "review_csv": review_csv,
            "review_json": review_json,
            "cleaned_csv": cleaned_csv,
            "cleaned_json": cleaned_json,
            "box_cleanup_csv": box_cleanup_csv,
            "box_cleanup_json": box_cleanup_json,
            "box_review_csv": box_review_csv,
            "box_review_json": box_review_json,
            "semantic_ensemble_csv": semantic_ensemble_csv,
            "semantic_ensemble_json": semantic_ensemble_json,
            "label_taxonomy_json": label_taxonomy_json,
        }

    def _resolve_export_dir(self, run_id: str, output_db: Path) -> Path:
        if self.config.export_dir is not None:
            return Path(self.config.export_dir).expanduser().resolve()
        return Path(output_db).expanduser().resolve().parent / "exports" / run_id

    def _resolve_crop_dir(self, run_id: str, output_db: Path) -> Path:
        if self.config.crop_dir is not None:
            return Path(self.config.crop_dir).expanduser().resolve()
        return Path(output_db).expanduser().resolve().parent / "crops" / run_id

    @staticmethod
    def _fetch_export_rows(conn: sqlite3.Connection, sql: str, run_id: str) -> list[dict[str, Any]]:
        rows = conn.execute(sql, (run_id,)).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")


def _json_or_raw(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
