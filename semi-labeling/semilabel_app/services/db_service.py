from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import pathname2url

from ..domain.models import Candidate, ClassDist, CleanedItem, QueueItem
from .sampling import decode_vec, select_diverse_sample

try:  # Tests for this service intentionally run without Qt.
    from PySide6 import QtCore
except Exception:  # pragma: no cover - exercised only on machines without PySide6
    QtCore = None  # type: ignore[assignment]


PROTOTYPE_CANDIDATE_SQL = """
WITH cand AS (
  SELECT sd.result_id, sd.final_label, sd.reliability_score, sd.model_agreement,
         cv.crop_path, cv.image_rel_path, cv.x1, cv.y1, cv.x2, cv.y2,
         CASE WHEN sd.reliability_score < ? THEN 'reject' ELSE sd.final_label END AS eff_label,
         CASE WHEN sd.reliability_score >= 0.9 THEN 90
              WHEN sd.reliability_score >= 0.5 THEN CAST(sd.reliability_score * 10 AS INT) * 10
              ELSE 40 END AS band
  FROM semantic_decisions sd
  JOIN crop_views cv ON cv.run_id = sd.run_id AND cv.result_id = sd.result_id AND cv.view_name = 'tight'
  JOIN crop_embeddings ce ON ce.embedding_run_id = ? AND ce.result_id = sd.result_id AND ce.view_name = 'tight'
  WHERE sd.run_id = ? AND sd.final_label IN ('crack','mold','spall')
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY eff_label, band
    ORDER BY reliability_score DESC, result_id ASC
  ) AS rn
  FROM cand
)
SELECT * FROM ranked WHERE rn <= ? ORDER BY eff_label ASC, band DESC, rn ASC
"""


def clean_path(value: Any, field_name: str) -> str:
    raw = str(value or "").strip()
    if not raw or "\0" in raw:
        raise ValueError(f"Invalid {field_name}")
    return raw


def expand_home(value: str) -> str:
    raw = str(value or "").strip()
    if raw.startswith("~"):
        return str(Path.home() / raw[1:].lstrip("/\\"))
    return raw


def resolve_db_path(db_path: str | Path) -> Path:
    return Path(expand_home(str(db_path))).resolve()


def _sqlite_file_uri(path: Path, *, immutable: bool = False) -> str:
    # sqlite3 URI handling on Windows requires spaces to be percent-encoded.
    if os.name == "nt":
        uri = "file:///" + quote(str(path).replace("\\", "/"), safe="/:")
    else:
        uri = "file://" + pathname2url(str(path))
    suffix = "mode=ro"
    if immutable:
        suffix += "&immutable=1"
    return f"{uri}?{suffix}"


def connect_ro(db_path: str | Path) -> sqlite3.Connection:
    resolved = resolve_db_path(db_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")
    if Path(f"{resolved}-shm").exists():
        try:
            writer = sqlite3.connect(str(resolved), timeout=60)
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            writer.close()
        except Exception:
            pass
    conn = sqlite3.connect(_sqlite_file_uri(resolved, immutable=True), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def connect_rw(db_path: str | Path) -> sqlite3.Connection:
    resolved = resolve_db_path(db_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")
    conn = sqlite3.connect(str(resolved), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _file_uri(path: str | Path) -> str:
    if not path:
        return ""
    return Path(path).resolve().as_uri()


def _resolve_image_path(image_root: str, rel_path: str) -> str:
    root = str(image_root or "").strip()
    rel = str(rel_path or "").strip()
    if not root or not rel:
        return ""
    full = Path(expand_home(root)).resolve() / rel
    return str(full) if full.is_file() else ""


def _resolve_crop_path(crop_path: str) -> str:
    raw = str(crop_path or "").strip()
    if not raw:
        return ""
    path = Path(expand_home(raw)).resolve()
    return str(path) if path.is_file() else ""


def _json_array(value: Any) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value or "[]"))
    except Exception:
        parsed = []
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed)


def _box(row: sqlite3.Row) -> tuple[float, float, float, float] | None:
    if row["x1"] is None:
        return None
    return (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))


def list_runs(db_path: str | Path) -> dict[str, Any]:
    conn = connect_ro(db_path)
    try:
        rows = conn.execute(
            """
            SELECT r.run_id, r.created_at_utc,
                   (SELECT COUNT(*) FROM review_queue q WHERE q.run_id = r.run_id) AS queue_count,
                   (SELECT COUNT(*) FROM cleaned_labels c WHERE c.run_id = r.run_id) AS cleaned_count
            FROM resemi_runs r
            ORDER BY r.created_at_utc DESC, r.run_id DESC
            """
        ).fetchall()
        return {"db_path": str(resolve_db_path(db_path)), "runs": [dict(row) for row in rows]}
    finally:
        conn.close()


def list_queue(
    db_path: str | Path,
    run_id: str,
    image_root: str,
    queue_type: str = "",
    sample_ratio: float = 0.0,
) -> dict[str, Any]:
    conn = connect_ro(db_path)
    try:
        latest_clf = conn.execute(
            """
            SELECT classifier_run_id FROM classifier_runs
            WHERE run_id = ? ORDER BY created_at_utc DESC, classifier_run_id DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        latest_st = conn.execute(
            """
            SELECT self_training_run_id FROM self_training_runs
            WHERE run_id = ? ORDER BY created_at_utc DESC, self_training_run_id DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        latest_clf_id = str(latest_clf["classifier_run_id"]) if latest_clf else ""
        latest_st_id = str(latest_st["self_training_run_id"]) if latest_st else ""
        type_clause = "AND q.queue_type = ?" if queue_type and queue_type != "all" else ""
        params: list[Any] = [run_id, latest_clf_id, latest_st_id, run_id]
        if type_clause:
            params.append(queue_type)
        rows = conn.execute(
            f"""
            SELECT q.result_id, q.image_rel_path, q.crop_path, q.initial_label, q.suggested_label,
                   q.queue_type, q.reliability_score, q.reason_codes_json,
                   cv.x1, cv.y1, cv.x2, cv.y2,
                   rd.action AS decided_action, rd.new_label AS decided_label,
                   cps.predicted_label, cps.predicted_probability, cps.margin,
                   cps.second_label, cps.second_probability,
                   cps.disagrees_with_policy, cps.policy_label,
                   stp.reason_codes_json AS defer_reason_codes_json
            FROM review_queue q
            LEFT JOIN crop_views cv ON cv.run_id = q.run_id AND cv.result_id = q.result_id AND cv.view_name = 'tight'
            LEFT JOIN review_decisions rd ON rd.result_id = q.result_id
              AND rd.review_session_id IN (SELECT review_session_id FROM review_sessions WHERE run_id = ?)
            LEFT JOIN classifier_prediction_summary cps
              ON cps.classifier_run_id = ? AND cps.result_id = q.result_id
            LEFT JOIN self_training_promotions stp
              ON stp.self_training_run_id = ? AND stp.result_id = q.result_id AND stp.action = 'defer_review'
            WHERE q.run_id = ? {type_clause}
            ORDER BY q.reliability_score ASC, q.result_id ASC
            """,
            params,
        ).fetchall()

        selected_ids: set[int] | None = None
        ratio = float(sample_ratio or 0)
        if 0 < ratio < 1 and rows:
            emb_row = conn.execute(
                """
                SELECT embedding_run_id FROM embedding_runs
                WHERE run_id = ? AND view_name = 'tight'
                ORDER BY created_at_utc DESC, rowid DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            emb_run_id = str(emb_row["embedding_run_id"]) if emb_row else ""
            vec_by_id: dict[int, Any] = {}
            if emb_run_id:
                for erow in conn.execute(
                    """
                    SELECT result_id, embedding_blob FROM crop_embeddings
                    WHERE embedding_run_id = ? AND view_name = 'tight'
                    """,
                    (emb_run_id,),
                ):
                    vec_by_id[int(erow["result_id"])] = decode_vec(erow["embedding_blob"])
            sample_rows = [
                {
                    "result_id": int(row["result_id"]),
                    "label": str(row["suggested_label"] or row["initial_label"] or "unknown"),
                    "reliability": float(row["reliability_score"] or 0),
                    "vec": vec_by_id.get(int(row["result_id"])),
                }
                for row in rows
            ]
            selected_ids = select_diverse_sample(sample_rows, ratio)

        all_items = []
        for row in rows:
            crop_path = _resolve_crop_path(row["crop_path"])
            pred_label = str(row["predicted_label"]) if row["predicted_label"] is not None else None
            all_items.append(
                QueueItem(
                    result_id=int(row["result_id"]),
                    image_rel_path=str(row["image_rel_path"] or ""),
                    crop_path=str(row["crop_path"] or ""),
                    image_uri=_file_uri(_resolve_image_path(image_root, row["image_rel_path"])),
                    crop_uri=_file_uri(crop_path) if crop_path else "",
                    initial_label=str(row["initial_label"] or ""),
                    suggested_label=str(row["suggested_label"] or ""),
                    queue_type=str(row["queue_type"] or ""),
                    reliability_score=float(row["reliability_score"] or 0),
                    reasons=_json_array(row["reason_codes_json"]),
                    box=_box(row),
                    decided_action=str(row["decided_action"] or ""),
                    decided_label=str(row["decided_label"] or ""),
                    pred_label=pred_label,
                    pred_prob=float(row["predicted_probability"] or 0),
                    pred_margin=float(row["margin"] or 0),
                    second_label=str(row["second_label"] or ""),
                    second_prob=(
                        None if row["second_probability"] is None else float(row["second_probability"])
                    ),
                    disagrees_with_policy=bool(int(row["disagrees_with_policy"] or 0)),
                    policy_label=str(row["policy_label"] or ""),
                    defer_reasons=_json_array(row["defer_reason_codes_json"]),
                )
            )
        items = (
            [item for item in all_items if item.result_id in selected_ids]
            if selected_ids is not None
            else all_items
        )
        counts: dict[str, int] = {}
        for item in items:
            counts[item.queue_type] = counts.get(item.queue_type, 0) + 1
        return {
            "items": items,
            "counts": counts,
            "total": len(items),
            "queue_total": len(all_items),
            "sampled": selected_ids is not None,
        }
    finally:
        conn.close()


def list_cleaned(
    db_path: str | Path,
    run_id: str,
    image_root: str,
    final_label: str = "",
    decision_type: str = "",
    limit: int = 0,
) -> dict[str, Any]:
    conn = connect_ro(db_path)
    try:
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if final_label and final_label != "all":
            clauses.append("final_label = ?")
            params.append(final_label)
        if decision_type and decision_type != "all":
            clauses.append("decision_type = ?")
            params.append(decision_type)
        safe_limit = max(0, int(limit or 0))
        limit_sql = f" LIMIT {safe_limit}" if safe_limit > 0 else ""
        rows = conn.execute(
            f"""
            SELECT result_id, image_rel_path, crop_path, final_label, export_label,
                   decision_type, reliability_score, reason_codes_json,
                   x1, y1, x2, y2, self_training_run_id, decision_policy_run_id
            FROM cleaned_labels
            WHERE {' AND '.join(clauses)}
            ORDER BY result_id ASC{limit_sql}
            """,
            params,
        ).fetchall()
        items = []
        for row in rows:
            crop_path = _resolve_crop_path(row["crop_path"])
            items.append(
                CleanedItem(
                    result_id=int(row["result_id"]),
                    image_rel_path=str(row["image_rel_path"] or ""),
                    crop_path=str(row["crop_path"] or ""),
                    final_label=str(row["final_label"] or ""),
                    export_label=str(row["export_label"] or ""),
                    decision_type=str(row["decision_type"] or ""),
                    reliability_score=float(row["reliability_score"] or 0),
                    box=_box(row),
                    crop_uri=_file_uri(crop_path) if crop_path else "",
                    image_uri=_file_uri(_resolve_image_path(image_root, row["image_rel_path"])),
                    reasons=_json_array(row["reason_codes_json"]),
                    self_training_run_id=str(row["self_training_run_id"] or ""),
                    decision_policy_run_id=str(row["decision_policy_run_id"] or ""),
                )
            )
        total = conn.execute(
            "SELECT COUNT(*) n FROM cleaned_labels WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
        return {"items": items, "total": int(total), "filtered": len(items)}
    finally:
        conn.close()


def list_image_boxes(db_path: str | Path, run_id: str, image_rel_path: str) -> list[dict[str, Any]]:
    """All tight boxes on one source image, with their label (for context overlay)."""
    rel = str(image_rel_path or "").strip()
    if not rel:
        return []
    conn = connect_ro(db_path)
    try:
        rows = conn.execute(
            """
            SELECT cv.result_id, cv.x1, cv.y1, cv.x2, cv.y2, sd.final_label
            FROM crop_views cv
            LEFT JOIN semantic_decisions sd
              ON sd.run_id = cv.run_id AND sd.result_id = cv.result_id
            WHERE cv.run_id = ? AND cv.view_name = 'tight' AND cv.image_rel_path = ?
              AND cv.x1 IS NOT NULL
            """,
            (run_id, rel),
        ).fetchall()
        return [
            {
                "result_id": int(row["result_id"]),
                "label": str(row["final_label"] or ""),
                "box": (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
            }
            for row in rows
        ]
    finally:
        conn.close()


def cleaned_distribution(db_path: str | Path, run_id: str) -> ClassDist:
    conn = connect_ro(db_path)
    try:
        label_rows = conn.execute(
            """
            SELECT final_label AS k, COUNT(*) n
            FROM cleaned_labels WHERE run_id = ?
            GROUP BY final_label ORDER BY n DESC
            """,
            (run_id,),
        ).fetchall()
        type_rows = conn.execute(
            """
            SELECT decision_type AS k, COUNT(*) n
            FROM cleaned_labels WHERE run_id = ?
            GROUP BY decision_type ORDER BY n DESC
            """,
            (run_id,),
        ).fetchall()
        total = sum(int(row["n"] or 0) for row in label_rows)

        def shape(rows: list[sqlite3.Row]) -> list[tuple[str, int, float]]:
            return [
                (str(row["k"] or ""), int(row["n"] or 0), (int(row["n"] or 0) / total) if total else 0.0)
                for row in rows
            ]

        return ClassDist(total=total, by_label=shape(label_rows), by_decision_type=shape(type_rows))
    finally:
        conn.close()


def get_run_resources(db_path: str | Path, run_id: str) -> dict[str, Any]:
    conn = connect_ro(db_path)
    try:
        def one(sql: str) -> sqlite3.Row | None:
            try:
                return conn.execute(sql, (run_id,)).fetchone()
            except sqlite3.Error:
                return None

        emb = one(
            "SELECT embedding_run_id FROM embedding_runs WHERE run_id = ? AND view_name='tight' ORDER BY created_at_utc DESC LIMIT 1"
        )
        proto = one(
            "SELECT prototype_version_id FROM prototype_versions WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1"
        )
        core = one(
            "SELECT core_mining_run_id FROM core_mining_runs WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1"
        )
        clf = one(
            "SELECT classifier_run_id, created_at_utc FROM classifier_runs WHERE run_id = ? ORDER BY created_at_utc DESC LIMIT 1"
        )
        counts = {
            "reviewQueue": int(
                conn.execute("SELECT COUNT(*) n FROM review_queue WHERE run_id = ?", (run_id,)).fetchone()[
                    "n"
                ]
            ),
            "cleaned": int(
                conn.execute("SELECT COUNT(*) n FROM cleaned_labels WHERE run_id = ?", (run_id,)).fetchone()[
                    "n"
                ]
            ),
            "reviewDecisions": int(
                conn.execute(
                    """
                    SELECT COUNT(*) n FROM review_decisions d
                    JOIN review_sessions s ON s.review_session_id = d.review_session_id
                    WHERE s.run_id = ?
                    """,
                    (run_id,),
                ).fetchone()["n"]
            ),
        }
        return {
            "embeddingRunId": str(emb["embedding_run_id"]) if emb else "",
            "prototypeVersionId": str(proto["prototype_version_id"]) if proto else "",
            "coreMiningRunId": str(core["core_mining_run_id"]) if core else "",
            "classifierRunId": str(clf["classifier_run_id"]) if clf else "",
            "counts": counts,
        }
    finally:
        conn.close()


def list_prototype_candidates(
    db_path: str | Path,
    run_id: str,
    image_root: str,
    reject_below: float = 0.5,
    per_band: int = 200,
) -> dict[str, Any]:
    conn = connect_ro(db_path)
    try:
        safe_per_band = max(10, min(2000, int(per_band or 200)))
        safe_reject = float(reject_below) if reject_below is not None else 0.5
        emb_row = conn.execute(
            """
            SELECT embedding_run_id FROM embedding_runs
            WHERE run_id = ? AND view_name = 'tight'
            ORDER BY created_at_utc DESC, rowid DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        emb_run_id = str(emb_row["embedding_run_id"]) if emb_row else ""
        core_row = conn.execute(
            """
            SELECT core_mining_run_id FROM core_mining_runs
            WHERE run_id = ? ORDER BY created_at_utc DESC, rowid DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        core_run_id = str(core_row["core_mining_run_id"]) if core_row else ""

        size_by_cluster: dict[Any, int] = {}
        cluster_by_result: dict[int, dict[str, Any]] = {}
        if core_run_id:
            for row in conn.execute(
                "SELECT core_cluster_id, member_count FROM core_clusters WHERE core_mining_run_id = ?",
                (core_run_id,),
            ):
                size_by_cluster[row["core_cluster_id"]] = int(row["member_count"] or 0)
            for row in conn.execute(
                """
                SELECT result_id, core_cluster_id, similarity
                FROM core_cluster_members WHERE core_mining_run_id = ?
                """,
                (core_run_id,),
            ):
                cluster_by_result[int(row["result_id"])] = {
                    "cluster_id": row["core_cluster_id"],
                    "similarity": float(row["similarity"] or 0),
                }

        rows = conn.execute(
            PROTOTYPE_CANDIDATE_SQL, (safe_reject, emb_run_id, run_id, safe_per_band)
        ).fetchall()

        by_label_clusters: dict[str, set[Any]] = {}
        for row in rows:
            info = cluster_by_result.get(int(row["result_id"]))
            if info is None:
                continue
            by_label_clusters.setdefault(str(row["eff_label"] or ""), set()).add(info["cluster_id"])

        domain_index_by_cluster: dict[Any, int] = {}
        domains: dict[str, list[dict[str, Any]]] = {}
        for label, cluster_ids in by_label_clusters.items():
            ordered = sorted(cluster_ids, key=lambda cid: size_by_cluster.get(cid, 0), reverse=True)
            domains[label] = []
            for idx, cid in enumerate(ordered):
                domain_index_by_cluster[cid] = idx
                domains[label].append(
                    {"clusterId": str(cid), "domainIndex": idx, "size": size_by_cluster.get(cid, 0)}
                )

        items: list[Candidate] = []
        for row in rows:
            crop_path = _resolve_crop_path(row["crop_path"])
            info = cluster_by_result.get(int(row["result_id"]))
            cid = info["cluster_id"] if info else None
            items.append(
                Candidate(
                    result_id=int(row["result_id"]),
                    label=str(row["eff_label"] or ""),
                    predicted_label=str(row["final_label"] or ""),
                    reliability_score=float(row["reliability_score"] or 0),
                    crop_uri=_file_uri(crop_path) if crop_path else "",
                    image_uri=_file_uri(_resolve_image_path(image_root, row["image_rel_path"])),
                    box=_box(row),
                    cluster_id="" if cid is None else str(cid),
                    domain_index=None if cid is None else domain_index_by_cluster.get(cid),
                    cluster_size=0 if cid is None else size_by_cluster.get(cid, 0),
                    centroid_similarity=None if info is None else float(info["similarity"]),
                    model_agreement=float(row["model_agreement"] or 0),
                )
            )
        return {
            "items": items,
            "labels": ["crack", "mold", "spall", "reject"],
            "embeddingRunId": emb_run_id,
            "coreMiningRunId": core_run_id,
            "domains": domains,
            "rejectBelow": safe_reject,
            "perBand": safe_per_band,
        }
    finally:
        conn.close()


def latest_prototype(db_path: str | Path, run_id: str) -> dict[str, Any]:
    conn = connect_ro(db_path)
    try:
        row = conn.execute(
            """
            SELECT pv.prototype_version_id, pv.created_at_utc, pv.notes,
                   (SELECT COUNT(*) FROM prototype_items pi WHERE pi.prototype_version_id = pv.prototype_version_id) AS item_count
            FROM prototype_versions pv WHERE pv.run_id = ?
            ORDER BY pv.created_at_utc DESC LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        return {"prototype": dict(row) if row else None}
    finally:
        conn.close()


def get_selftrain_promotions(
    db_path: str | Path,
    run_id: str,
    image_root: str,
    self_training_run_id: str = "",
    action: str = "",
) -> dict[str, Any]:
    conn = connect_ro(db_path)
    try:
        st_id = str(self_training_run_id or "").strip()
        if not st_id:
            row = conn.execute(
                """
                SELECT self_training_run_id FROM self_training_runs
                WHERE run_id = ? ORDER BY created_at_utc DESC, self_training_run_id DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            st_id = str(row["self_training_run_id"]) if row else ""
        if not st_id:
            return {"promotions": []}
        action_clause = "AND p.action = ?" if action and action != "all" else ""
        params: list[Any] = [run_id, st_id]
        if action_clause:
            params.append(action)
        rows = conn.execute(
            f"""
            SELECT p.result_id, p.action, p.predicted_label, p.classifier_confidence, p.classifier_margin,
                   cv.image_rel_path, cv.crop_path, cv.x1, cv.y1, cv.x2, cv.y2
            FROM self_training_promotions p
            LEFT JOIN crop_views cv ON cv.run_id = ? AND cv.result_id = p.result_id AND cv.view_name = 'tight'
            WHERE p.self_training_run_id = ? {action_clause}
            ORDER BY p.result_id ASC
            """,
            params,
        ).fetchall()
        promotions = []
        for row in rows:
            crop_path = _resolve_crop_path(row["crop_path"])
            promotions.append(
                {
                    "resultId": int(row["result_id"]),
                    "action": str(row["action"] or ""),
                    "predictedLabel": str(row["predicted_label"] or ""),
                    "classifierConfidence": float(row["classifier_confidence"] or 0),
                    "classifierMargin": float(row["classifier_margin"] or 0),
                    "imageUri": _file_uri(_resolve_image_path(image_root, row["image_rel_path"])),
                    "cropUri": _file_uri(crop_path) if crop_path else "",
                    "box": _box(row),
                }
            )
        return {"promotions": promotions}
    finally:
        conn.close()


if QtCore is not None:

    class DbWorkerSignals(QtCore.QObject):
        finished = QtCore.Signal(object)
        error = QtCore.Signal(str)


    class DbWorker(QtCore.QRunnable):
        def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.fn = fn
            self.args = args
            self.kwargs = kwargs
            self.signals = DbWorkerSignals()

        def run(self) -> None:
            try:
                self.signals.finished.emit(self.fn(*self.args, **self.kwargs))
            except Exception as exc:
                self.signals.error.emit(str(exc))

else:

    class DbWorker:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required for DbWorker")
