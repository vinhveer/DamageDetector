from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass

import numpy as np

from shared.db.embedding_cache import load_embeddings


@dataclass(frozen=True)
class PrototypeSpec:
    result_id: int
    label: str
    is_reject: bool = False
    source_type: str = "manual_result"
    source_ref: str = ""
    note: str = ""


@dataclass(frozen=True)
class PrototypeBankConfig:
    run_id: str
    model_name: str
    view_name: str
    embedding_run_id: str = ""
    prototype_version_id: str = ""
    source_session: str = ""
    notes: str = ""
    selected_clusters: tuple[tuple[str, str | None], ...] = ()
    excluded_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PrototypeBankResult:
    prototype_version_id: str
    run_id: str
    embedding_run_id: str
    model_name: str
    view_name: str
    prototypes: list[PrototypeSpec]
    selected_cluster_ids: list[str]
    excluded_ids: list[int]


@dataclass(frozen=True)
class PrototypeScore:
    result_id: int
    top_label: str
    top_similarity: float
    second_label: str | None
    second_similarity: float | None
    margin: float
    top_prototype_result_id: int | None
    is_reject_match: bool
    similarities: list[dict[str, object]]
    reason_codes: list[str]

    @property
    def similarities_json(self) -> str:
        return json.dumps(self.similarities, ensure_ascii=False, sort_keys=True)

    @property
    def reason_codes_json(self) -> str:
        return json.dumps(sorted(set(self.reason_codes)), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class PrototypeScoringResult:
    prototype_score_run_id: str
    prototype_version_id: str
    run_id: str
    embedding_run_id: str
    model_name: str
    view_name: str
    prototype_count: int
    scores: list[PrototypeScore]
    options: dict[str, object]


def build_prototype_bank(
    conn: sqlite3.Connection,
    *,
    config: PrototypeBankConfig,
    manual_prototypes: list[PrototypeSpec],
) -> PrototypeBankResult:
    cluster_prototypes = _read_cluster_prototypes(conn, config=config)
    excluded = {int(item) for item in config.excluded_ids}
    prototypes_by_id: dict[int, PrototypeSpec] = {}
    for proto in [*manual_prototypes, *cluster_prototypes]:
        if int(proto.result_id) in excluded:
            continue
        prototypes_by_id[int(proto.result_id)] = proto
    prototypes = sorted(prototypes_by_id.values(), key=lambda item: (item.label, item.result_id))
    if not prototypes:
        raise RuntimeError("Prototype bank is empty. Provide --prototype/--reject or --cluster picks.")
    _, _, embedding_run = load_embeddings(
        conn,
        model_name=config.model_name,
        view_name=config.view_name,
        run_id=config.run_id,
        embedding_run_id=config.embedding_run_id or None,
        result_ids=[item.result_id for item in prototypes],
    )
    embedding_run_id = str(embedding_run["embedding_run_id"])
    available_ids = _embedding_ids(conn, embedding_run_id=embedding_run_id, view_name=config.view_name)
    missing = [item.result_id for item in prototypes if item.result_id not in available_ids]
    if missing:
        raise RuntimeError(f"Missing embeddings for prototype result_id(s): {missing[:20]}")
    version_id = config.prototype_version_id or f"proto_{config.run_id}_{embedding_run_id[:8]}_{uuid.uuid4().hex[:8]}"
    return PrototypeBankResult(
        prototype_version_id=version_id,
        run_id=config.run_id,
        embedding_run_id=embedding_run_id,
        model_name=config.model_name,
        view_name=config.view_name,
        prototypes=prototypes,
        selected_cluster_ids=[cluster_id for cluster_id, _ in config.selected_clusters],
        excluded_ids=sorted(excluded),
    )


def persist_prototype_bank(conn: sqlite3.Connection, result: PrototypeBankResult, *, created_at_utc: str, source_session: str, notes: str, options: dict[str, object]) -> None:
    _delete_prototype_version(conn, result.prototype_version_id)
    label_map: dict[str, list[int]] = {}
    for proto in result.prototypes:
        label_map.setdefault(proto.label, []).append(proto.result_id)
    embeddings, ids, _ = load_embeddings(
        conn,
        model_name=result.model_name,
        view_name=result.view_name,
        embedding_run_id=result.embedding_run_id,
        result_ids=[item.result_id for item in result.prototypes],
    )
    vector_by_id = {int(result_id): embeddings[idx] for idx, result_id in enumerate(ids)}
    conn.execute(
        """
        INSERT INTO prototype_versions (
            prototype_version_id, run_id, created_at_utc, notes, source_session,
            label_map_json, selected_result_ids_json, selected_cluster_ids_json,
            excluded_ids_json, embedding_run_id, model_name, view_name, options_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.prototype_version_id,
            result.run_id,
            created_at_utc,
            notes,
            source_session,
            json.dumps(label_map, ensure_ascii=False, sort_keys=True),
            json.dumps([item.result_id for item in result.prototypes], ensure_ascii=False),
            json.dumps(result.selected_cluster_ids, ensure_ascii=False),
            json.dumps(result.excluded_ids, ensure_ascii=False),
            result.embedding_run_id,
            result.model_name,
            result.view_name,
            json.dumps(options, ensure_ascii=False, sort_keys=True),
        ),
    )
    conn.executemany(
        """
        INSERT INTO prototype_items (
            prototype_version_id, result_id, label, is_reject, note,
            source_type, source_ref, embedding_run_id, view_name, embedding_blob, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                result.prototype_version_id,
                proto.result_id,
                proto.label,
                1 if proto.is_reject else 0,
                proto.note,
                proto.source_type,
                proto.source_ref,
                result.embedding_run_id,
                result.view_name,
                np.asarray(vector_by_id[proto.result_id], dtype="<f4").tobytes(),
                created_at_utc,
            )
            for proto in result.prototypes
        ],
    )
    conn.commit()


def score_prototypes(
    conn: sqlite3.Connection,
    *,
    prototype_version_id: str,
    run_id: str,
    model_name: str,
    view_name: str,
    embedding_run_id: str = "",
    top_k: int = 8,
) -> PrototypeScoringResult:
    version = conn.execute("SELECT * FROM prototype_versions WHERE prototype_version_id = ?", (prototype_version_id,)).fetchone()
    if version is None:
        raise RuntimeError(f"Prototype version not found: {prototype_version_id}")
    effective_embedding_run_id = embedding_run_id or str(version["embedding_run_id"] or "")
    embeddings, result_ids, embedding_run = load_embeddings(
        conn,
        model_name=model_name,
        view_name=view_name,
        run_id=run_id,
        embedding_run_id=effective_embedding_run_id or None,
    )
    effective_embedding_run_id = str(embedding_run["embedding_run_id"])
    proto_rows = conn.execute(
        """
        SELECT result_id, label, is_reject, embedding_blob
        FROM prototype_items
        WHERE prototype_version_id = ? AND embedding_run_id = ? AND view_name = ?
        ORDER BY label, result_id
        """,
        (prototype_version_id, effective_embedding_run_id, view_name),
    ).fetchall()
    if not proto_rows:
        raise RuntimeError("No prototype items matched the requested embedding run/view.")
    dim = int(embedding_run["dim"])
    proto_ids = [int(row["result_id"]) for row in proto_rows]
    proto_labels = [str(row["label"]) for row in proto_rows]
    proto_reject = [bool(int(row["is_reject"])) for row in proto_rows]
    proto_matrix = np.vstack([_decode_vector(bytes(row["embedding_blob"]), dim=dim) for row in proto_rows]).astype(np.float32, copy=False)
    similarities = embeddings @ proto_matrix.T
    scores = [_score_row(int(result_id), similarities[row_idx], proto_ids, proto_labels, proto_reject, top_k=top_k) for row_idx, result_id in enumerate(result_ids)]
    score_run_id = f"pscore_{prototype_version_id}_{effective_embedding_run_id[:8]}_{view_name}"
    return PrototypeScoringResult(
        prototype_score_run_id=score_run_id,
        prototype_version_id=prototype_version_id,
        run_id=run_id,
        embedding_run_id=effective_embedding_run_id,
        model_name=model_name,
        view_name=view_name,
        prototype_count=len(proto_rows),
        scores=scores,
        options={"top_k": int(top_k)},
    )


def score_prototype_bank_preview(conn: sqlite3.Connection, *, bank: PrototypeBankResult, top_k: int = 8) -> PrototypeScoringResult:
    embeddings, result_ids, embedding_run = load_embeddings(
        conn,
        model_name=bank.model_name,
        view_name=bank.view_name,
        run_id=bank.run_id,
        embedding_run_id=bank.embedding_run_id,
    )
    proto_embeddings, proto_ids, _ = load_embeddings(
        conn,
        model_name=bank.model_name,
        view_name=bank.view_name,
        embedding_run_id=bank.embedding_run_id,
        result_ids=[item.result_id for item in bank.prototypes],
    )
    proto_by_id = {int(result_id): proto_embeddings[idx] for idx, result_id in enumerate(proto_ids)}
    proto_specs = [item for item in bank.prototypes if item.result_id in proto_by_id]
    if not proto_specs:
        raise RuntimeError("No prototype embeddings available for preview scoring.")
    proto_matrix = np.vstack([proto_by_id[item.result_id] for item in proto_specs]).astype(np.float32, copy=False)
    proto_result_ids = [item.result_id for item in proto_specs]
    proto_labels = [item.label for item in proto_specs]
    proto_reject = [item.is_reject for item in proto_specs]
    similarities = embeddings @ proto_matrix.T
    scores = [_score_row(int(result_id), similarities[row_idx], proto_result_ids, proto_labels, proto_reject, top_k=top_k) for row_idx, result_id in enumerate(result_ids)]
    embedding_run_id = str(embedding_run["embedding_run_id"])
    return PrototypeScoringResult(
        prototype_score_run_id=f"pscore_{bank.prototype_version_id}_{embedding_run_id[:8]}_{bank.view_name}",
        prototype_version_id=bank.prototype_version_id,
        run_id=bank.run_id,
        embedding_run_id=embedding_run_id,
        model_name=bank.model_name,
        view_name=bank.view_name,
        prototype_count=len(proto_specs),
        scores=scores,
        options={"top_k": int(top_k), "preview": True},
    )


def persist_prototype_scores(conn: sqlite3.Connection, result: PrototypeScoringResult, *, created_at_utc: str, update_decisions: bool = True) -> None:
    conn.execute("DELETE FROM prototype_scores WHERE prototype_score_run_id = ?", (result.prototype_score_run_id,))
    conn.execute("DELETE FROM prototype_scoring_runs WHERE prototype_score_run_id = ?", (result.prototype_score_run_id,))
    conn.execute(
        """
        INSERT INTO prototype_scoring_runs (
            prototype_score_run_id, prototype_version_id, run_id, embedding_run_id,
            model_name, view_name, created_at_utc, options_json, prototype_count, scored_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.prototype_score_run_id,
            result.prototype_version_id,
            result.run_id,
            result.embedding_run_id,
            result.model_name,
            result.view_name,
            created_at_utc,
            json.dumps(result.options, ensure_ascii=False, sort_keys=True),
            result.prototype_count,
            len(result.scores),
        ),
    )
    conn.executemany(
        """
        INSERT INTO prototype_scores (
            prototype_score_run_id, result_id, top_prototype_label, top_prototype_sim,
            second_prototype_label, second_prototype_sim, prototype_margin,
            top_prototype_result_id, is_reject_match, similarities_json, reason_codes_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                result.prototype_score_run_id,
                score.result_id,
                score.top_label,
                score.top_similarity,
                score.second_label,
                score.second_similarity,
                score.margin,
                score.top_prototype_result_id,
                1 if score.is_reject_match else 0,
                score.similarities_json,
                score.reason_codes_json,
            )
            for score in result.scores
        ],
    )
    if update_decisions:
        conn.executemany(
            """
            UPDATE semantic_decisions
            SET prototype_class = ?, prototype_similarity = ?
            WHERE run_id = ? AND result_id = ?
            """,
            [(score.top_label, score.top_similarity, result.run_id, score.result_id) for score in result.scores],
        )
        conn.executemany(
            """
            UPDATE reliability_scores
            SET prototype_class = ?, prototype_similarity = ?
            WHERE run_id = ? AND result_id = ?
            """,
            [(score.top_label, score.top_similarity, result.run_id, score.result_id) for score in result.scores],
        )
    conn.commit()


def _read_cluster_prototypes(conn: sqlite3.Connection, *, config: PrototypeBankConfig) -> list[PrototypeSpec]:
    prototypes: list[PrototypeSpec] = []
    for cluster_id, override_label in config.selected_clusters:
        row = conn.execute(
            "SELECT label FROM core_clusters WHERE run_id = ? AND core_cluster_id = ?",
            (config.run_id, cluster_id),
        ).fetchone()
        label = str(override_label or (row["label"] if row is not None else "")).strip()
        if not label:
            raise RuntimeError(f"Cluster label is missing for {cluster_id}; pass cluster_id:label.")
        member_rows = conn.execute(
            """
            SELECT result_id
            FROM core_cluster_members
            WHERE run_id = ? AND core_cluster_id = ? AND COALESCE(is_core_member, 1) = 1
            ORDER BY similarity DESC, result_id
            """,
            (config.run_id, cluster_id),
        ).fetchall()
        prototypes.extend(
            PrototypeSpec(result_id=int(member["result_id"]), label=label, source_type="core_cluster", source_ref=cluster_id)
            for member in member_rows
        )
    return prototypes


def _score_row(result_id: int, similarities: np.ndarray, proto_ids: list[int], proto_labels: list[str], proto_reject: list[bool], *, top_k: int) -> PrototypeScore:
    order = np.argsort(-similarities)
    label_best: dict[str, tuple[float, int, bool]] = {}
    for proto_idx in order:
        label = proto_labels[int(proto_idx)]
        if label not in label_best:
            label_best[label] = (float(similarities[int(proto_idx)]), int(proto_ids[int(proto_idx)]), bool(proto_reject[int(proto_idx)]))
    ranked_labels = sorted(label_best.items(), key=lambda item: item[1][0], reverse=True)
    top_label, (top_sim, top_proto_id, top_is_reject) = ranked_labels[0]
    if len(ranked_labels) > 1:
        second_label, (second_sim, _, _) = ranked_labels[1]
    else:
        second_label, second_sim = None, None
    margin = top_sim - float(second_sim if second_sim is not None else 0.0)
    top_items = [
        {
            "prototype_result_id": int(proto_ids[int(idx)]),
            "label": proto_labels[int(idx)],
            "similarity": float(similarities[int(idx)]),
            "is_reject": bool(proto_reject[int(idx)]),
        }
        for idx in order[: max(1, int(top_k))]
    ]
    reasons = ["prototype_top_reject"] if top_is_reject else []
    if second_sim is not None and margin < 0.03:
        reasons.append("prototype_low_margin")
    return PrototypeScore(
        result_id=result_id,
        top_label=top_label,
        top_similarity=top_sim,
        second_label=second_label,
        second_similarity=second_sim,
        margin=margin,
        top_prototype_result_id=top_proto_id,
        is_reject_match=top_is_reject,
        similarities=top_items,
        reason_codes=reasons,
    )


def _embedding_ids(conn: sqlite3.Connection, *, embedding_run_id: str, view_name: str) -> set[int]:
    rows = conn.execute(
        "SELECT result_id FROM crop_embeddings WHERE embedding_run_id = ? AND view_name = ?",
        (embedding_run_id, view_name),
    ).fetchall()
    return {int(row["result_id"]) for row in rows}


def _delete_prototype_version(conn: sqlite3.Connection, prototype_version_id: str) -> None:
    score_rows = conn.execute("SELECT prototype_score_run_id FROM prototype_scoring_runs WHERE prototype_version_id = ?", (prototype_version_id,)).fetchall()
    for row in score_rows:
        conn.execute("DELETE FROM prototype_scores WHERE prototype_score_run_id = ?", (str(row["prototype_score_run_id"]),))
    conn.execute("DELETE FROM prototype_scoring_runs WHERE prototype_version_id = ?", (prototype_version_id,))
    conn.execute("DELETE FROM prototype_items WHERE prototype_version_id = ?", (prototype_version_id,))
    conn.execute("DELETE FROM prototype_versions WHERE prototype_version_id = ?", (prototype_version_id,))
    conn.commit()


def _decode_vector(blob: bytes, *, dim: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype="<f4")
    if vector.size != int(dim):
        raise ValueError(f"Invalid prototype embedding size: got {vector.size}, expected {dim}")
    return vector.astype(np.float32, copy=True)
