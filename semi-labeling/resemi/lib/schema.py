from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "resemi_schema_v1"
SCHEMA_DESCRIPTION = "Resemi auditable SQLite artifact for semantic label cleaning."


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect_output(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    _migrate_review_tables(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key             TEXT PRIMARY KEY,
            value           TEXT NOT NULL,
            updated_at_utc  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id    TEXT PRIMARY KEY,
            applied_at_utc  TEXT NOT NULL,
            description     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS resemi_runs (
            run_id                  TEXT PRIMARY KEY,
            created_at_utc          TEXT NOT NULL,
            source_db_path          TEXT NOT NULL,
            source_semantic_run_id  TEXT NOT NULL,
            source_dedup_db_path    TEXT,
            source_dedup_run_id     TEXT,
            options_json            TEXT NOT NULL,
            total_detections        INTEGER NOT NULL DEFAULT 0,
            cleaned_count           INTEGER NOT NULL DEFAULT 0,
            suspect_count           INTEGER NOT NULL DEFAULT 0,
            reject_count            INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS crop_views (
            run_id          TEXT NOT NULL,
            result_id       INTEGER NOT NULL,
            view_name       TEXT NOT NULL,
            crop_path       TEXT,
            image_rel_path  TEXT NOT NULL,
            x1              REAL NOT NULL,
            y1              REAL NOT NULL,
            x2              REAL NOT NULL,
            y2              REAL NOT NULL,
            source          TEXT NOT NULL,
            PRIMARY KEY (run_id, result_id, view_name),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS crop_consistency_features (
            run_id                   TEXT NOT NULL,
            result_id                INTEGER NOT NULL,
            label_consistency        REAL,
            score_variance           REAL,
            view_disagreement_count  INTEGER,
            context_shift_label      TEXT,
            status                   TEXT NOT NULL,
            features_json            TEXT NOT NULL,
            created_at_utc           TEXT NOT NULL,
            PRIMARY KEY (run_id, result_id),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS semantic_model_scores (
            run_id       TEXT NOT NULL,
            result_id    INTEGER NOT NULL,
            model_name   TEXT NOT NULL,
            label        TEXT NOT NULL,
            probability  REAL NOT NULL,
            rank         INTEGER NOT NULL,
            PRIMARY KEY (run_id, result_id, model_name, label),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS semantic_model_outputs (
            run_id           TEXT NOT NULL,
            result_id        INTEGER NOT NULL,
            model_name       TEXT NOT NULL,
            source_type      TEXT NOT NULL,
            top1_label       TEXT NOT NULL,
            top1_score       REAL NOT NULL,
            top2_label       TEXT,
            top2_score       REAL,
            margin           REAL NOT NULL,
            entropy          REAL,
            raw_scores_json  TEXT NOT NULL,
            created_at_utc   TEXT NOT NULL,
            PRIMARY KEY (run_id, result_id, model_name),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS semantic_agreements (
            run_id                  TEXT NOT NULL,
            result_id               INTEGER NOT NULL,
            majority_label          TEXT NOT NULL,
            agreement_ratio         REAL NOT NULL,
            strong_agreement_count  INTEGER NOT NULL,
            conflict_labels_json    TEXT NOT NULL,
            sources_json            TEXT NOT NULL,
            created_at_utc          TEXT NOT NULL,
            PRIMARY KEY (run_id, result_id),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_semantic_agreement_ratio
        ON semantic_agreements (run_id, agreement_ratio);

        CREATE TABLE IF NOT EXISTS label_taxonomy_versions (
            taxonomy_version_id   TEXT NOT NULL,
            run_id                TEXT NOT NULL,
            created_at_utc        TEXT NOT NULL,
            working_labels_json   TEXT NOT NULL,
            damage_labels_json    TEXT NOT NULL,
            reject_labels_json    TEXT NOT NULL,
            export_mapping_json   TEXT NOT NULL,
            guidelines_json       TEXT NOT NULL,
            PRIMARY KEY (taxonomy_version_id, run_id),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS box_graph_runs (
            box_graph_run_id    TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL,
            created_at_utc      TEXT NOT NULL,
            options_json        TEXT NOT NULL,
            total_boxes         INTEGER NOT NULL,
            edge_count          INTEGER NOT NULL,
            decision_count      INTEGER NOT NULL,
            review_count        INTEGER NOT NULL,
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS box_graph_edges (
            box_graph_run_id              TEXT NOT NULL,
            parent_result_id              INTEGER NOT NULL,
            child_result_id               INTEGER NOT NULL,
            image_rel_path                TEXT NOT NULL,
            iou                           REAL NOT NULL,
            intersection_area             REAL NOT NULL,
            containment_small_in_large    REAL NOT NULL,
            child_coverage_of_parent      REAL NOT NULL,
            area_ratio                    REAL NOT NULL,
            center_distance_norm          REAL NOT NULL,
            aspect_ratio_similarity       REAL NOT NULL,
            label_agreement               INTEGER NOT NULL,
            edge_type                     TEXT NOT NULL,
            features_json                 TEXT NOT NULL,
            PRIMARY KEY (box_graph_run_id, parent_result_id, child_result_id),
            FOREIGN KEY(box_graph_run_id) REFERENCES box_graph_runs(box_graph_run_id)
        );

        CREATE TABLE IF NOT EXISTS box_quality_scores (
            box_graph_run_id          TEXT NOT NULL,
            result_id                 INTEGER NOT NULL,
            image_rel_path            TEXT NOT NULL,
            label                     TEXT NOT NULL,
            box_quality_score         REAL NOT NULL,
            detector_score            REAL NOT NULL,
            semantic_confidence       REAL NOT NULL,
            semantic_margin           REAL NOT NULL,
            crop_consistency          REAL,
            embedding_core_similarity REAL,
            prototype_similarity      REAL,
            area_ratio_to_image       REAL NOT NULL,
            aspect_ratio              REAL NOT NULL,
            elongation                REAL NOT NULL,
            child_count               INTEGER NOT NULL,
            child_label_diversity     INTEGER NOT NULL,
            child_alignment_score     REAL NOT NULL,
            background_context_penalty REAL NOT NULL,
            composite_penalty         REAL NOT NULL,
            components_json           TEXT NOT NULL,
            PRIMARY KEY (box_graph_run_id, result_id),
            FOREIGN KEY(box_graph_run_id) REFERENCES box_graph_runs(box_graph_run_id)
        );

        CREATE TABLE IF NOT EXISTS box_cleanup_decisions (
            box_graph_run_id      TEXT NOT NULL,
            result_id             INTEGER NOT NULL,
            image_rel_path        TEXT NOT NULL,
            label                 TEXT NOT NULL,
            decision_type         TEXT NOT NULL,
            keep_for_cleaned      INTEGER NOT NULL,
            box_quality_score     REAL NOT NULL,
            representative_id     INTEGER,
            reason_codes_json     TEXT NOT NULL,
            features_json         TEXT NOT NULL,
            created_at_utc        TEXT NOT NULL,
            PRIMARY KEY (box_graph_run_id, result_id),
            FOREIGN KEY(box_graph_run_id) REFERENCES box_graph_runs(box_graph_run_id)
        );

        CREATE TABLE IF NOT EXISTS box_review_queue (
            box_graph_run_id      TEXT NOT NULL,
            result_id             INTEGER NOT NULL,
            image_rel_path        TEXT NOT NULL,
            label                 TEXT NOT NULL,
            queue_type            TEXT NOT NULL,
            box_quality_score     REAL NOT NULL,
            reason_codes_json     TEXT NOT NULL,
            features_json         TEXT NOT NULL,
            PRIMARY KEY (box_graph_run_id, result_id),
            FOREIGN KEY(box_graph_run_id) REFERENCES box_graph_runs(box_graph_run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_box_cleanup_decision_type
        ON box_cleanup_decisions (box_graph_run_id, decision_type);

        CREATE INDEX IF NOT EXISTS idx_box_review_queue_type
        ON box_review_queue (box_graph_run_id, queue_type);

        CREATE TABLE IF NOT EXISTS embedding_runs (
            embedding_run_id  TEXT PRIMARY KEY,
            run_id            TEXT NOT NULL,
            created_at_utc    TEXT NOT NULL,
            model_name        TEXT NOT NULL,
            dim               INTEGER NOT NULL,
            options_json      TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS crop_embeddings (
            embedding_run_id  TEXT NOT NULL,
            result_id         INTEGER NOT NULL,
            view_name         TEXT NOT NULL,
            embedding_blob    BLOB NOT NULL,
            PRIMARY KEY (embedding_run_id, result_id, view_name),
            FOREIGN KEY(embedding_run_id) REFERENCES embedding_runs(embedding_run_id)
        );

        CREATE TABLE IF NOT EXISTS skipped_crop_embeddings (
            embedding_run_id  TEXT NOT NULL,
            result_id         INTEGER NOT NULL,
            view_name         TEXT NOT NULL,
            crop_path         TEXT,
            reason            TEXT NOT NULL,
            error_message     TEXT NOT NULL,
            PRIMARY KEY (embedding_run_id, result_id, view_name),
            FOREIGN KEY(embedding_run_id) REFERENCES embedding_runs(embedding_run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_crop_embeddings_lookup
        ON crop_embeddings (embedding_run_id, view_name, result_id);

        CREATE INDEX IF NOT EXISTS idx_embedding_runs_latest
        ON embedding_runs (model_name, created_at_utc);

        CREATE TABLE IF NOT EXISTS prototype_versions (
            prototype_version_id  TEXT PRIMARY KEY,
            run_id                TEXT NOT NULL,
            created_at_utc        TEXT NOT NULL,
            notes                 TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS prototype_items (
            prototype_version_id  TEXT NOT NULL,
            result_id             INTEGER NOT NULL,
            label                 TEXT NOT NULL,
            is_reject             INTEGER NOT NULL DEFAULT 0,
            note                  TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (prototype_version_id, result_id),
            FOREIGN KEY(prototype_version_id) REFERENCES prototype_versions(prototype_version_id)
        );

        CREATE TABLE IF NOT EXISTS prototype_scoring_runs (
            prototype_score_run_id TEXT PRIMARY KEY,
            prototype_version_id   TEXT NOT NULL,
            run_id                 TEXT NOT NULL,
            embedding_run_id       TEXT NOT NULL,
            model_name             TEXT NOT NULL,
            view_name              TEXT NOT NULL,
            created_at_utc         TEXT NOT NULL,
            options_json           TEXT NOT NULL,
            prototype_count        INTEGER NOT NULL,
            scored_count           INTEGER NOT NULL,
            FOREIGN KEY(prototype_version_id) REFERENCES prototype_versions(prototype_version_id),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS prototype_scores (
            prototype_score_run_id TEXT NOT NULL,
            result_id              INTEGER NOT NULL,
            top_prototype_label    TEXT NOT NULL,
            top_prototype_sim      REAL NOT NULL,
            second_prototype_label TEXT,
            second_prototype_sim   REAL,
            prototype_margin       REAL NOT NULL,
            top_prototype_result_id INTEGER,
            is_reject_match        INTEGER NOT NULL,
            similarities_json      TEXT NOT NULL,
            reason_codes_json      TEXT NOT NULL,
            PRIMARY KEY (prototype_score_run_id, result_id),
            FOREIGN KEY(prototype_score_run_id) REFERENCES prototype_scoring_runs(prototype_score_run_id)
        );

        CREATE TABLE IF NOT EXISTS core_clusters (
            run_id              TEXT NOT NULL,
            core_cluster_id     TEXT NOT NULL,
            label               TEXT NOT NULL,
            centroid_json       TEXT,
            size                INTEGER NOT NULL DEFAULT 0,
            created_at_utc      TEXT NOT NULL,
            PRIMARY KEY (run_id, core_cluster_id),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS core_mining_runs (
            core_mining_run_id  TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL,
            embedding_run_id    TEXT NOT NULL,
            model_name          TEXT NOT NULL,
            view_name           TEXT NOT NULL,
            created_at_utc      TEXT NOT NULL,
            options_json        TEXT NOT NULL,
            total_embeddings    INTEGER NOT NULL,
            clustered_count     INTEGER NOT NULL,
            core_cluster_count  INTEGER NOT NULL,
            rare_count          INTEGER NOT NULL,
            noise_count         INTEGER NOT NULL,
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS core_outliers (
            core_mining_run_id  TEXT NOT NULL,
            result_id           INTEGER NOT NULL,
            label               TEXT NOT NULL,
            outlier_type        TEXT NOT NULL,
            nearest_cluster_id  TEXT,
            similarity          REAL,
            reason_codes_json   TEXT NOT NULL,
            PRIMARY KEY (core_mining_run_id, result_id),
            FOREIGN KEY(core_mining_run_id) REFERENCES core_mining_runs(core_mining_run_id)
        );

        CREATE TABLE IF NOT EXISTS core_cluster_members (
            run_id            TEXT NOT NULL,
            core_cluster_id   TEXT NOT NULL,
            result_id         INTEGER NOT NULL,
            similarity        REAL NOT NULL,
            PRIMARY KEY (run_id, core_cluster_id, result_id),
            FOREIGN KEY(run_id, core_cluster_id) REFERENCES core_clusters(run_id, core_cluster_id)
        );

        CREATE TABLE IF NOT EXISTS reliability_scores (
            run_id                 TEXT NOT NULL,
            result_id              INTEGER NOT NULL,
            reliability_score      REAL NOT NULL,
            model_agreement        REAL NOT NULL,
            top1_top2_margin       REAL NOT NULL,
            nearest_core_class     TEXT,
            nearest_core_similarity REAL,
            prototype_class        TEXT,
            prototype_similarity   REAL,
            reason_codes_json      TEXT NOT NULL,
            score_components_json  TEXT NOT NULL,
            created_at_utc         TEXT NOT NULL,
            PRIMARY KEY (run_id, result_id),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS reliability_scoring_runs (
            reliability_run_id      TEXT PRIMARY KEY,
            run_id                  TEXT NOT NULL,
            created_at_utc          TEXT NOT NULL,
            core_mining_run_id      TEXT,
            prototype_score_run_id  TEXT,
            options_json            TEXT NOT NULL,
            scored_count            INTEGER NOT NULL,
            auto_accept_count       INTEGER NOT NULL,
            suspect_count           INTEGER NOT NULL,
            reject_count            INTEGER NOT NULL,
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS semantic_decisions (
            run_id                 TEXT NOT NULL,
            result_id              INTEGER NOT NULL,
            initial_label          TEXT NOT NULL,
            suggested_label        TEXT NOT NULL,
            final_label            TEXT NOT NULL,
            decision_type          TEXT NOT NULL,
            reliability_score      REAL NOT NULL,
            nearest_core_class     TEXT,
            nearest_core_similarity REAL,
            prototype_class        TEXT,
            prototype_similarity   REAL,
            model_agreement        REAL NOT NULL,
            reason_codes_json      TEXT NOT NULL,
            score_components_json  TEXT NOT NULL,
            created_at_utc         TEXT NOT NULL,
            PRIMARY KEY (run_id, result_id),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS decision_policy_runs (
            decision_policy_run_id TEXT PRIMARY KEY,
            run_id                 TEXT NOT NULL,
            created_at_utc         TEXT NOT NULL,
            reliability_run_id     TEXT,
            taxonomy_version_id    TEXT,
            options_json           TEXT NOT NULL,
            total_count            INTEGER NOT NULL,
            cleaned_count          INTEGER NOT NULL,
            review_count           INTEGER NOT NULL,
            reject_count           INTEGER NOT NULL,
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS decision_policy_audit (
            decision_policy_run_id TEXT NOT NULL,
            result_id              INTEGER NOT NULL,
            matched_rule           TEXT NOT NULL,
            previous_decision_type TEXT NOT NULL,
            final_decision_type    TEXT NOT NULL,
            final_label            TEXT NOT NULL,
            reliability_score      REAL NOT NULL,
            thresholds_json        TEXT NOT NULL,
            score_components_json  TEXT NOT NULL,
            reason_codes_json      TEXT NOT NULL,
            PRIMARY KEY (decision_policy_run_id, result_id),
            FOREIGN KEY(decision_policy_run_id) REFERENCES decision_policy_runs(decision_policy_run_id)
        );

        CREATE TABLE IF NOT EXISTS review_sessions (
            review_session_id              TEXT PRIMARY KEY,
            run_id                         TEXT NOT NULL,
            reviewer                       TEXT NOT NULL DEFAULT '',
            status                         TEXT NOT NULL DEFAULT 'draft',
            created_at_utc                 TEXT NOT NULL,
            committed_at_utc               TEXT,
            source_reliability_run_id      TEXT,
            source_decision_policy_run_id  TEXT,
            source_prototype_version_id    TEXT,
            notes                          TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS review_decisions (
            review_session_id       TEXT NOT NULL,
            target_type             TEXT NOT NULL,
            target_id               TEXT NOT NULL,
            result_id               INTEGER,
            core_cluster_id         TEXT,
            batch_id                TEXT,
            action                  TEXT NOT NULL,
            previous_label          TEXT,
            new_label               TEXT,
            previous_decision_type  TEXT,
            new_decision_type       TEXT,
            confidence_override     REAL,
            reason_codes_json       TEXT NOT NULL DEFAULT '[]',
            affected_result_ids_json TEXT NOT NULL DEFAULT '[]',
            note                    TEXT NOT NULL DEFAULT '',
            created_at_utc          TEXT NOT NULL,
            PRIMARY KEY (review_session_id, target_type, target_id),
            FOREIGN KEY(review_session_id) REFERENCES review_sessions(review_session_id)
        );

        CREATE TABLE IF NOT EXISTS classifier_runs (
            classifier_run_id  TEXT PRIMARY KEY,
            run_id             TEXT NOT NULL,
            created_at_utc     TEXT NOT NULL,
            model_json         TEXT NOT NULL,
            options_json       TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS classifier_predictions (
            classifier_run_id  TEXT NOT NULL,
            result_id          INTEGER NOT NULL,
            label              TEXT NOT NULL,
            probability        REAL NOT NULL,
            PRIMARY KEY (classifier_run_id, result_id, label),
            FOREIGN KEY(classifier_run_id) REFERENCES classifier_runs(classifier_run_id)
        );

        CREATE TABLE IF NOT EXISTS classifier_training_items (
            classifier_run_id  TEXT NOT NULL,
            result_id          INTEGER NOT NULL,
            label              TEXT NOT NULL,
            source_type        TEXT NOT NULL,
            source_ref         TEXT NOT NULL,
            reliability_score  REAL,
            reason_codes_json  TEXT NOT NULL,
            PRIMARY KEY (classifier_run_id, result_id),
            FOREIGN KEY(classifier_run_id) REFERENCES classifier_runs(classifier_run_id)
        );

        CREATE TABLE IF NOT EXISTS classifier_prediction_summary (
            classifier_run_id    TEXT NOT NULL,
            result_id            INTEGER NOT NULL,
            predicted_label      TEXT NOT NULL,
            predicted_probability REAL NOT NULL,
            second_label         TEXT,
            second_probability   REAL,
            margin               REAL NOT NULL,
            disagrees_with_policy INTEGER NOT NULL,
            policy_label         TEXT,
            reason_codes_json    TEXT NOT NULL,
            PRIMARY KEY (classifier_run_id, result_id),
            FOREIGN KEY(classifier_run_id) REFERENCES classifier_runs(classifier_run_id)
        );

        CREATE TABLE IF NOT EXISTS classifier_oof_predictions (
            classifier_run_id  TEXT NOT NULL,
            result_id          INTEGER NOT NULL,
            true_label         TEXT NOT NULL,
            predicted_label    TEXT NOT NULL,
            probability        REAL NOT NULL,
            is_disagreement    INTEGER NOT NULL,
            PRIMARY KEY (classifier_run_id, result_id),
            FOREIGN KEY(classifier_run_id) REFERENCES classifier_runs(classifier_run_id)
        );

        CREATE TABLE IF NOT EXISTS self_training_runs (
            self_training_run_id TEXT PRIMARY KEY,
            run_id               TEXT NOT NULL,
            classifier_run_id    TEXT NOT NULL,
            created_at_utc       TEXT NOT NULL,
            round_index          INTEGER NOT NULL,
            options_json         TEXT NOT NULL,
            candidate_count      INTEGER NOT NULL,
            promoted_count       INTEGER NOT NULL,
            rejected_count       INTEGER NOT NULL,
            deferred_count       INTEGER NOT NULL,
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id),
            FOREIGN KEY(classifier_run_id) REFERENCES classifier_runs(classifier_run_id)
        );

        CREATE TABLE IF NOT EXISTS self_training_promotions (
            self_training_run_id TEXT NOT NULL,
            result_id            INTEGER NOT NULL,
            previous_label       TEXT NOT NULL,
            predicted_label      TEXT NOT NULL,
            action               TEXT NOT NULL,
            classifier_confidence REAL NOT NULL,
            classifier_margin    REAL NOT NULL,
            prototype_class      TEXT,
            prototype_similarity REAL,
            nearest_core_class   TEXT,
            nearest_core_similarity REAL,
            geometry_decision    TEXT,
            consistency_score    REAL,
            reason_codes_json    TEXT NOT NULL,
            evidence_json        TEXT NOT NULL,
            PRIMARY KEY (self_training_run_id, result_id),
            FOREIGN KEY(self_training_run_id) REFERENCES self_training_runs(self_training_run_id)
        );

        CREATE TABLE IF NOT EXISTS cleaned_labels (
            run_id             TEXT NOT NULL,
            result_id          INTEGER NOT NULL,
            image_rel_path     TEXT NOT NULL,
            crop_path          TEXT,
            final_label        TEXT NOT NULL,
            decision_type      TEXT NOT NULL,
            reliability_score  REAL NOT NULL,
            reason_codes_json  TEXT NOT NULL,
            x1                 REAL NOT NULL,
            y1                 REAL NOT NULL,
            x2                 REAL NOT NULL,
            y2                 REAL NOT NULL,
            PRIMARY KEY (run_id, result_id),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS review_queue (
            run_id             TEXT NOT NULL,
            result_id          INTEGER NOT NULL,
            image_rel_path     TEXT NOT NULL,
            crop_path          TEXT,
            initial_label      TEXT NOT NULL,
            suggested_label    TEXT NOT NULL,
            queue_type         TEXT NOT NULL,
            reliability_score  REAL NOT NULL,
            reason_codes_json  TEXT NOT NULL,
            PRIMARY KEY (run_id, result_id),
            FOREIGN KEY(run_id) REFERENCES resemi_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_resemi_decision_type
        ON semantic_decisions (run_id, decision_type);

        CREATE INDEX IF NOT EXISTS idx_resemi_cleaned_label
        ON cleaned_labels (run_id, final_label);

        CREATE INDEX IF NOT EXISTS idx_resemi_review_queue
        ON review_queue (run_id, queue_type, reliability_score);

        CREATE INDEX IF NOT EXISTS idx_review_decisions_session
        ON review_decisions (review_session_id);
        """
    )
    _ensure_column(conn, "resemi_runs", "model_versions_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "resemi_runs", "thresholds_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "resemi_runs", "prompt_set_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "resemi_runs", "prototype_version_id", "TEXT")
    _ensure_column(conn, "resemi_runs", "taxonomy_version_id", "TEXT")
    _ensure_column(conn, "crop_views", "crop_hash", "TEXT")
    _ensure_column(conn, "crop_views", "crop_width", "INTEGER")
    _ensure_column(conn, "crop_views", "crop_height", "INTEGER")
    _ensure_column(conn, "crop_views", "padding_ratio", "REAL")
    _ensure_column(conn, "crop_views", "status", "TEXT NOT NULL DEFAULT 'ok'")
    _ensure_column(conn, "crop_views", "error_message", "TEXT")
    _ensure_column(conn, "cleaned_labels", "export_label", "TEXT")
    _ensure_column(conn, "embedding_runs", "model_version", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "embedding_runs", "embedding_type", "TEXT NOT NULL DEFAULT 'dinov2_crop'")
    _ensure_column(conn, "embedding_runs", "device", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "embedding_runs", "view_name", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "embedding_runs", "total_crops", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "embedding_runs", "embedded_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "embedding_runs", "skipped_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "crop_embeddings", "crop_view_id", "TEXT")
    _ensure_column(conn, "crop_embeddings", "crop_path", "TEXT")
    _ensure_column(conn, "crop_embeddings", "dim", "INTEGER")
    _ensure_column(conn, "crop_embeddings", "created_at_utc", "TEXT")
    _ensure_column(conn, "prototype_versions", "source_session", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "prototype_versions", "label_map_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "prototype_versions", "selected_result_ids_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "prototype_versions", "selected_cluster_ids_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "prototype_versions", "excluded_ids_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "prototype_versions", "embedding_run_id", "TEXT")
    _ensure_column(conn, "prototype_versions", "model_name", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "prototype_versions", "view_name", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "prototype_versions", "options_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "prototype_items", "source_type", "TEXT NOT NULL DEFAULT 'manual_result'")
    _ensure_column(conn, "prototype_items", "source_ref", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "prototype_items", "embedding_run_id", "TEXT")
    _ensure_column(conn, "prototype_items", "view_name", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "prototype_items", "embedding_blob", "BLOB")
    _ensure_column(conn, "prototype_items", "created_at_utc", "TEXT")
    _ensure_column(conn, "core_clusters", "core_mining_run_id", "TEXT")
    _ensure_column(conn, "core_clusters", "embedding_run_id", "TEXT")
    _ensure_column(conn, "core_clusters", "view_name", "TEXT")
    _ensure_column(conn, "core_clusters", "centroid_blob", "BLOB")
    _ensure_column(conn, "core_clusters", "member_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "core_clusters", "density_score", "REAL")
    _ensure_column(conn, "core_clusters", "agreement_score", "REAL")
    _ensure_column(conn, "core_clusters", "status", "TEXT NOT NULL DEFAULT 'core'")
    _ensure_column(conn, "core_cluster_members", "core_mining_run_id", "TEXT")
    _ensure_column(conn, "core_cluster_members", "distance_to_centroid", "REAL")
    _ensure_column(conn, "core_cluster_members", "is_core_member", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "reliability_scores", "reliability_run_id", "TEXT")
    _ensure_column(conn, "semantic_decisions", "matched_rule", "TEXT")
    _ensure_column(conn, "cleaned_labels", "decision_policy_run_id", "TEXT")
    _ensure_column(conn, "review_queue", "decision_policy_run_id", "TEXT")
    _ensure_column(conn, "classifier_runs", "embedding_run_id", "TEXT")
    _ensure_column(conn, "classifier_runs", "model_type", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "classifier_runs", "feature_set_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "classifier_runs", "label_set_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "classifier_runs", "train_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "classifier_runs", "prediction_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "classifier_runs", "evaluation_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "classifier_runs", "model_blob", "BLOB")
    _ensure_column(conn, "semantic_decisions", "self_training_run_id", "TEXT")
    _ensure_column(conn, "cleaned_labels", "self_training_run_id", "TEXT")
    _write_schema_metadata(conn)
    conn.commit()


def _write_schema_metadata(conn: sqlite3.Connection) -> None:
    now = utc_now()
    rows = [
        ("schema_version", SCHEMA_VERSION, now),
        ("schema_description", SCHEMA_DESCRIPTION, now),
        ("schema_owner", "resemi", now),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO schema_metadata (key, value, updated_at_utc) VALUES (?, ?, ?)",
        rows,
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_migrations (migration_id, applied_at_utc, description)
        VALUES (?, ?, ?)
        """,
        (SCHEMA_VERSION, now, SCHEMA_DESCRIPTION),
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def _migrate_review_tables(conn: sqlite3.Connection) -> None:
    """Drop the legacy v0 review tables so the SPEC-11 schema below can recreate them.

    The v0 review_decisions PK is (review_session_id, result_id) with NOT NULL columns
    that SPEC-11 needs to relax; SQLite cannot ALTER a primary key, so we drop and let
    ensure_schema recreate. Safe because we only drop when the tables are still on the
    old shape AND hold no rows.
    """
    cols = _table_columns(conn, "review_decisions")
    if not cols:
        return  # table does not exist yet; CREATE will use the new schema
    if "target_type" in cols:
        return  # already migrated
    count = conn.execute("SELECT COUNT(*) FROM review_decisions").fetchone()[0]
    session_count = conn.execute("SELECT COUNT(*) FROM review_sessions").fetchone()[0]
    if count or session_count:
        raise RuntimeError(
            "Refusing to migrate review tables: legacy review_sessions/review_decisions "
            "contain data. Export and clear them before upgrading to the SPEC-11 schema."
        )
    conn.execute("DROP TABLE IF EXISTS review_decisions")
    conn.execute("DROP TABLE IF EXISTS review_sessions")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row["name"]) for row in rows}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
