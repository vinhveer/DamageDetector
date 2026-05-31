#!/usr/bin/env python3
"""Tool — audit the resemi SQLite schema and basic table inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.runtime import bootstrap

bootstrap.ensure_on_path()

from shared.runtime.paths import default_resemi_db  # noqa: E402
from shared.db.schema import SCHEMA_VERSION, connect_output  # noqa: E402


REQUIRED_TABLES: tuple[str, ...] = (
    "schema_metadata",
    "schema_migrations",
    "resemi_runs",
    "crop_views",
    "crop_consistency_features",
    "semantic_model_scores",
    "semantic_model_outputs",
    "semantic_agreements",
    "label_taxonomy_versions",
    "box_graph_runs",
    "box_graph_edges",
    "box_quality_scores",
    "box_cleanup_decisions",
    "box_review_queue",
    "embedding_runs",
    "crop_embeddings",
    "skipped_crop_embeddings",
    "prototype_versions",
    "prototype_items",
    "prototype_scoring_runs",
    "prototype_scores",
    "core_mining_runs",
    "core_clusters",
    "core_cluster_members",
    "core_outliers",
    "reliability_scoring_runs",
    "reliability_scores",
    "semantic_decisions",
    "decision_policy_runs",
    "decision_policy_audit",
    "review_sessions",
    "review_decisions",
    "classifier_runs",
    "classifier_training_items",
    "classifier_predictions",
    "classifier_prediction_summary",
    "classifier_oof_predictions",
    "self_training_runs",
    "self_training_promotions",
    "cleaned_labels",
    "review_queue",
)


REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "schema_metadata": ("key", "value", "updated_at_utc"),
    "resemi_runs": ("run_id", "created_at_utc", "source_db_path", "source_semantic_run_id", "options_json", "model_versions_json", "thresholds_json"),
    "crop_views": ("run_id", "result_id", "view_name", "crop_path", "image_rel_path", "x1", "y1", "x2", "y2", "crop_hash", "status"),
    "semantic_decisions": ("run_id", "result_id", "initial_label", "suggested_label", "final_label", "decision_type", "reliability_score", "reason_codes_json", "score_components_json", "created_at_utc"),
    "reliability_scores": ("run_id", "result_id", "reliability_score", "model_agreement", "top1_top2_margin", "reason_codes_json", "score_components_json"),
    "prototype_items": ("prototype_version_id", "result_id", "label", "is_reject", "source_type", "embedding_blob"),
    "core_clusters": ("run_id", "core_cluster_id", "label", "centroid_blob", "member_count", "density_score"),
    "decision_policy_audit": ("decision_policy_run_id", "result_id", "matched_rule", "thresholds_json", "score_components_json", "reason_codes_json"),
    "review_sessions": ("review_session_id", "run_id", "reviewer", "status", "created_at_utc", "committed_at_utc"),
    "review_decisions": ("review_session_id", "target_type", "target_id", "action", "affected_result_ids_json", "created_at_utc"),
    "classifier_runs": ("classifier_run_id", "run_id", "embedding_run_id", "model_type", "evaluation_json", "model_blob"),
    "self_training_runs": ("self_training_run_id", "run_id", "classifier_run_id", "round_index", "options_json"),
    "cleaned_labels": ("run_id", "result_id", "image_rel_path", "final_label", "export_label", "decision_type", "reliability_score"),
    "review_queue": ("run_id", "result_id", "image_rel_path", "initial_label", "suggested_label", "queue_type", "reason_codes_json"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the resemi SQLite schema and basic table inventory.")
    parser.add_argument("--db", default=str(default_resemi_db()), help="Resemi SQLite DB.")
    parser.add_argument("--json", action="store_true", help="Print full JSON audit result.")
    return parser


def _columns(conn, table: str) -> list[str]:
    return [str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def audit_schema(db_path: Path) -> dict[str, object]:
    conn = connect_output(db_path)
    try:
        table_rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        tables = [str(row["name"]) for row in table_rows]
        table_set = set(tables)
        columns = {table: _columns(conn, table) for table in tables}
        missing_tables = [table for table in REQUIRED_TABLES if table not in table_set]
        missing_columns = {
            table: [column for column in required if column not in columns.get(table, [])]
            for table, required in REQUIRED_COLUMNS.items()
            if table in table_set and any(column not in columns.get(table, []) for column in required)
        }
        row_counts = {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}
        metadata = {str(row["key"]): str(row["value"]) for row in conn.execute("SELECT key, value FROM schema_metadata").fetchall()} if "schema_metadata" in table_set else {}
        status = "ok" if not missing_tables and not missing_columns and metadata.get("schema_version") == SCHEMA_VERSION else "needs_attention"
        return {
            "status": status,
            "db_path": str(db_path),
            "expected_schema_version": SCHEMA_VERSION,
            "metadata": metadata,
            "table_count": len(tables),
            "tables": tables,
            "missing_tables": missing_tables,
            "missing_columns": missing_columns,
            "row_counts": row_counts,
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Resemi DB not found: {db_path}")
    result = audit_schema(db_path)
    if bool(args.json):
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"status={result['status']}")
        print(f"db={result['db_path']}")
        print(f"expected_schema_version={result['expected_schema_version']}")
        print(f"actual_schema_version={result['metadata'].get('schema_version') if isinstance(result['metadata'], dict) else None}")
        print(f"table_count={result['table_count']}")
        print(f"missing_tables={len(result['missing_tables'])}")
        print(f"missing_columns={sum(len(v) for v in result['missing_columns'].values()) if isinstance(result['missing_columns'], dict) else 0}")
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
