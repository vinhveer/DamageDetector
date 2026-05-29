from __future__ import annotations

import json
import pickle
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .embedding_cache import load_embeddings


@dataclass(frozen=True)
class ClassifierConfig:
    run_id: str
    model_name: str
    view_name: str
    embedding_run_id: str = ""
    model_type: str = "logistic_regression"
    classifier_run_id: str = ""
    min_train_reliability: float = 0.75
    include_low_priority: bool = False
    prototype_version_id: str = ""
    core_mining_run_id: str = ""
    cv_folds: int = 5
    random_state: int = 17


@dataclass(frozen=True)
class TrainingItem:
    result_id: int
    label: str
    source_type: str
    source_ref: str
    reliability_score: float | None
    reason_codes: list[str]

    @property
    def reason_codes_json(self) -> str:
        return json.dumps(sorted(set(self.reason_codes)), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class PredictionSummary:
    result_id: int
    predicted_label: str
    predicted_probability: float
    second_label: str | None
    second_probability: float | None
    margin: float
    disagrees_with_policy: bool
    policy_label: str | None
    reason_codes: list[str]

    @property
    def reason_codes_json(self) -> str:
        return json.dumps(sorted(set(self.reason_codes)), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class OofPrediction:
    result_id: int
    true_label: str
    predicted_label: str
    probability: float
    is_disagreement: bool


@dataclass(frozen=True)
class ClassifierResult:
    classifier_run_id: str
    run_id: str
    embedding_run_id: str
    model_type: str
    model_json: dict[str, object]
    options: dict[str, object]
    feature_set: dict[str, object]
    labels: list[str]
    training_items: list[TrainingItem]
    predictions: dict[int, dict[str, float]]
    summaries: list[PredictionSummary]
    oof_predictions: list[OofPrediction]
    evaluation: dict[str, object]
    model_blob: bytes


def train_lightweight_classifier(conn: sqlite3.Connection, *, config: ClassifierConfig) -> ClassifierResult:
    embeddings, result_ids, embedding_run = load_embeddings(
        conn,
        model_name=config.model_name,
        view_name=config.view_name,
        run_id=config.run_id,
        embedding_run_id=config.embedding_run_id or None,
    )
    if embeddings.size == 0:
        raise RuntimeError("No embeddings available for classifier training.")
    embedding_run_id = str(embedding_run["embedding_run_id"])
    training_items = _collect_training_items(conn, config=config, available_ids=set(result_ids))
    if len(training_items) < 2:
        raise RuntimeError("Not enough training items. Need at least 2 labeled embeddings.")
    labels = sorted({item.label for item in training_items})
    if len(labels) < 2:
        raise RuntimeError("Not enough classes. Need at least 2 classes to train classifier.")
    id_to_index = {int(result_id): idx for idx, result_id in enumerate(result_ids)}
    train_indices = [id_to_index[item.result_id] for item in training_items]
    x_train = embeddings[train_indices]
    y_train = np.asarray([item.label for item in training_items], dtype=object)
    estimator = _build_estimator(config)
    evaluation, oof_predictions = _evaluate(estimator, x_train, y_train, training_items, config=config, labels=labels)
    estimator.fit(x_train, y_train)
    probabilities = _predict_probabilities(estimator, embeddings, labels)
    policy_labels = _read_policy_labels(conn, run_id=config.run_id)
    predictions: dict[int, dict[str, float]] = {}
    summaries: list[PredictionSummary] = []
    for row_idx, result_id in enumerate(result_ids):
        probs = {label: float(probabilities[row_idx, label_idx]) for label_idx, label in enumerate(labels)}
        predictions[int(result_id)] = probs
        summaries.append(_summarize_prediction(int(result_id), probs, policy_labels.get(int(result_id))))
    classifier_run_id = config.classifier_run_id or f"clf_{config.run_id}_{embedding_run_id[:8]}_{uuid.uuid4().hex[:8]}"
    model_json = {
        "model_type": config.model_type,
        "sklearn_estimator": estimator.__class__.__name__,
        "classes": labels,
    }
    return ClassifierResult(
        classifier_run_id=classifier_run_id,
        run_id=config.run_id,
        embedding_run_id=embedding_run_id,
        model_type=config.model_type,
        model_json=model_json,
        options={
            "model_name": config.model_name,
            "view_name": config.view_name,
            "embedding_run_id": config.embedding_run_id,
            "min_train_reliability": config.min_train_reliability,
            "include_low_priority": config.include_low_priority,
            "prototype_version_id": config.prototype_version_id,
            "core_mining_run_id": config.core_mining_run_id,
            "cv_folds": config.cv_folds,
            "random_state": config.random_state,
        },
        feature_set={"embedding": config.model_name, "view_name": config.view_name, "dim": int(embedding_run["dim"])},
        labels=labels,
        training_items=training_items,
        predictions=predictions,
        summaries=summaries,
        oof_predictions=oof_predictions,
        evaluation=evaluation,
        model_blob=pickle.dumps(estimator),
    )


def persist_classifier_result(conn: sqlite3.Connection, result: ClassifierResult, *, created_at_utc: str) -> None:
    _delete_existing(conn, result.classifier_run_id)
    conn.execute(
        """
        INSERT INTO classifier_runs (
            classifier_run_id, run_id, created_at_utc, model_json, options_json,
            embedding_run_id, model_type, feature_set_json, label_set_json,
            train_count, prediction_count, evaluation_json, model_blob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.classifier_run_id,
            result.run_id,
            created_at_utc,
            json.dumps(result.model_json, ensure_ascii=False, sort_keys=True),
            json.dumps(result.options, ensure_ascii=False, sort_keys=True),
            result.embedding_run_id,
            result.model_type,
            json.dumps(result.feature_set, ensure_ascii=False, sort_keys=True),
            json.dumps(result.labels, ensure_ascii=False),
            len(result.training_items),
            len(result.summaries),
            json.dumps(result.evaluation, ensure_ascii=False, sort_keys=True),
            result.model_blob,
        ),
    )
    conn.executemany(
        """
        INSERT INTO classifier_training_items (
            classifier_run_id, result_id, label, source_type, source_ref,
            reliability_score, reason_codes_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                result.classifier_run_id,
                item.result_id,
                item.label,
                item.source_type,
                item.source_ref,
                item.reliability_score,
                item.reason_codes_json,
            )
            for item in result.training_items
        ],
    )
    conn.executemany(
        """
        INSERT INTO classifier_predictions (classifier_run_id, result_id, label, probability)
        VALUES (?, ?, ?, ?)
        """,
        [
            (result.classifier_run_id, result_id, label, probability)
            for result_id, probs in result.predictions.items()
            for label, probability in probs.items()
        ],
    )
    conn.executemany(
        """
        INSERT INTO classifier_prediction_summary (
            classifier_run_id, result_id, predicted_label, predicted_probability,
            second_label, second_probability, margin, disagrees_with_policy,
            policy_label, reason_codes_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                result.classifier_run_id,
                item.result_id,
                item.predicted_label,
                item.predicted_probability,
                item.second_label,
                item.second_probability,
                item.margin,
                1 if item.disagrees_with_policy else 0,
                item.policy_label,
                item.reason_codes_json,
            )
            for item in result.summaries
        ],
    )
    conn.executemany(
        """
        INSERT INTO classifier_oof_predictions (
            classifier_run_id, result_id, true_label, predicted_label,
            probability, is_disagreement
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                result.classifier_run_id,
                item.result_id,
                item.true_label,
                item.predicted_label,
                item.probability,
                1 if item.is_disagreement else 0,
            )
            for item in result.oof_predictions
        ],
    )
    conn.commit()


def _collect_training_items(conn: sqlite3.Connection, *, config: ClassifierConfig, available_ids: set[int]) -> list[TrainingItem]:
    candidates: dict[int, TrainingItem] = {}
    priority = {"manual_review": 4, "prototype": 3, "core": 2, "cleaned_auto_accept": 1}

    def add(item: TrainingItem) -> None:
        if item.result_id not in available_ids:
            return
        existing = candidates.get(item.result_id)
        if existing is None or priority.get(item.source_type, 0) >= priority.get(existing.source_type, 0):
            candidates[item.result_id] = item

    cleaned_types = ["auto_accept"]
    if config.include_low_priority:
        cleaned_types.append("auto_accept_low_priority")
    placeholders = ", ".join("?" for _ in cleaned_types)
    rows = conn.execute(
        f"""
        SELECT result_id, final_label, reliability_score, decision_type, reason_codes_json
        FROM cleaned_labels
        WHERE run_id = ?
          AND decision_type IN ({placeholders})
          AND reliability_score >= ?
          AND final_label NOT IN ('reject', 'unknown', 'background', 'shadow', 'edge', 'object')
        """,
        [config.run_id, *cleaned_types, float(config.min_train_reliability)],
    ).fetchall()
    for row in rows:
        add(
            TrainingItem(
                result_id=int(row["result_id"]),
                label=str(row["final_label"]),
                source_type="cleaned_auto_accept",
                source_ref=str(row["decision_type"]),
                reliability_score=float(row["reliability_score"]),
                reason_codes=_parse_json_list(row["reason_codes_json"]),
            )
        )

    manual_rows = conn.execute(
        """
        SELECT d.result_id, d.final_label, d.decision_type, d.review_session_id
        FROM review_decisions d
        JOIN review_sessions s ON s.review_session_id = d.review_session_id
        WHERE s.run_id = ? AND d.decision_type IN ('manual_accept', 'manual_relabel')
          AND d.final_label NOT IN ('reject', 'unknown', 'background', 'shadow', 'edge', 'object')
        """,
        (config.run_id,),
    ).fetchall()
    for row in manual_rows:
        add(
            TrainingItem(
                result_id=int(row["result_id"]),
                label=str(row["final_label"]),
                source_type="manual_review",
                source_ref=str(row["review_session_id"]),
                reliability_score=None,
                reason_codes=[str(row["decision_type"])],
            )
        )

    if config.prototype_version_id:
        proto_rows = conn.execute(
            """
            SELECT result_id, label, is_reject, source_type
            FROM prototype_items
            WHERE prototype_version_id = ? AND is_reject = 0
              AND label NOT IN ('reject', 'unknown', 'background', 'shadow', 'edge', 'object')
            """,
            (config.prototype_version_id,),
        ).fetchall()
        for row in proto_rows:
            add(
                TrainingItem(
                    result_id=int(row["result_id"]),
                    label=str(row["label"]),
                    source_type="prototype",
                    source_ref=config.prototype_version_id,
                    reliability_score=None,
                    reason_codes=[str(row["source_type"]), "prototype_item"],
                )
            )

    if config.core_mining_run_id:
        core_rows = conn.execute(
            """
            SELECT m.result_id, c.label, m.core_cluster_id, m.similarity
            FROM core_cluster_members m
            JOIN core_clusters c ON c.run_id = m.run_id AND c.core_cluster_id = m.core_cluster_id
            WHERE c.core_mining_run_id = ? AND COALESCE(m.is_core_member, 1) = 1
              AND c.label NOT IN ('reject', 'unknown', 'background', 'shadow', 'edge', 'object')
            """,
            (config.core_mining_run_id,),
        ).fetchall()
        for row in core_rows:
            add(
                TrainingItem(
                    result_id=int(row["result_id"]),
                    label=str(row["label"]),
                    source_type="core",
                    source_ref=str(row["core_cluster_id"]),
                    reliability_score=float(row["similarity"]),
                    reason_codes=["core_member"],
                )
            )
    return sorted(candidates.values(), key=lambda item: item.result_id)


def _build_estimator(config: ClassifierConfig):
    if config.model_type == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)),
            ]
        )
    if config.model_type == "linear_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", random_state=config.random_state)),
            ]
        )
    if config.model_type == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=config.random_state)),
            ]
        )
    raise ValueError("model_type must be one of: logistic_regression, linear_svm, mlp")


def _evaluate(estimator, x_train: np.ndarray, y_train: np.ndarray, training_items: list[TrainingItem], *, config: ClassifierConfig, labels: list[str]) -> tuple[dict[str, object], list[OofPrediction]]:
    counts = Counter(str(label) for label in y_train)
    min_class_count = min(counts.values())
    effective_folds = min(int(config.cv_folds), int(min_class_count), len(y_train))
    evaluation: dict[str, object] = {
        "train_count": int(len(y_train)),
        "class_counts": dict(sorted(counts.items())),
        "requested_cv_folds": int(config.cv_folds),
        "effective_cv_folds": int(max(0, effective_folds)),
    }
    if effective_folds < 2 or len(labels) < 2:
        evaluation["cv_status"] = "skipped_insufficient_class_count"
        evaluation["cv_skip_reason"] = "Need at least 2 samples per class for StratifiedKFold."
        return evaluation, []
    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=config.random_state)
    if config.model_type == "linear_svm":
        raw = cross_val_predict(estimator, x_train, y_train, cv=cv, method="decision_function")
        probabilities = _softmax(raw if raw.ndim == 2 else np.vstack([-raw, raw]).T)
    else:
        probabilities = cross_val_predict(estimator, x_train, y_train, cv=cv, method="predict_proba")
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted_labels = [labels[int(idx)] for idx in predicted_indices]
    true_labels = [str(label) for label in y_train]
    evaluation["cv_status"] = "ok"
    evaluation["classification_report"] = classification_report(true_labels, predicted_labels, labels=labels, output_dict=True, zero_division=0)
    evaluation["confusion_matrix"] = confusion_matrix(true_labels, predicted_labels, labels=labels).tolist()
    evaluation["labels"] = labels
    oof = [
        OofPrediction(
            result_id=training_items[idx].result_id,
            true_label=true_labels[idx],
            predicted_label=predicted_labels[idx],
            probability=float(probabilities[idx, int(predicted_indices[idx])]),
            is_disagreement=predicted_labels[idx] != true_labels[idx],
        )
        for idx in range(len(training_items))
    ]
    evaluation["oof_disagreement_count"] = sum(1 for item in oof if item.is_disagreement)
    return evaluation, oof


def _predict_probabilities(estimator, embeddings: np.ndarray, labels: list[str]) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(embeddings)
    else:
        raw = estimator.decision_function(embeddings)
        probs = _softmax(raw if raw.ndim == 2 else np.vstack([-raw, raw]).T)
    estimator_classes = [str(item) for item in estimator.classes_] if hasattr(estimator, "classes_") else labels
    if estimator_classes == labels:
        return np.asarray(probs, dtype=np.float32)
    ordered = np.zeros((probs.shape[0], len(labels)), dtype=np.float32)
    class_to_idx = {label: idx for idx, label in enumerate(estimator_classes)}
    for idx, label in enumerate(labels):
        ordered[:, idx] = probs[:, class_to_idx[label]]
    return ordered


def _summarize_prediction(result_id: int, probs: dict[str, float], policy_label: str | None) -> PredictionSummary:
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_label, top_prob = ranked[0]
    second_label, second_prob = ranked[1] if len(ranked) > 1 else (None, None)
    margin = float(top_prob - (second_prob or 0.0))
    reasons: list[str] = []
    disagrees = bool(policy_label and policy_label != top_label)
    if disagrees:
        reasons.append("classifier_policy_disagreement")
    if margin < 0.10:
        reasons.append("classifier_low_margin")
    return PredictionSummary(
        result_id=result_id,
        predicted_label=top_label,
        predicted_probability=float(top_prob),
        second_label=second_label,
        second_probability=float(second_prob) if second_prob is not None else None,
        margin=margin,
        disagrees_with_policy=disagrees,
        policy_label=policy_label,
        reason_codes=reasons,
    )


def _read_policy_labels(conn: sqlite3.Connection, *, run_id: str) -> dict[int, str]:
    rows = conn.execute("SELECT result_id, final_label FROM semantic_decisions WHERE run_id = ?", (run_id,)).fetchall()
    return {int(row["result_id"]): str(row["final_label"]) for row in rows}


def _softmax(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    values = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def _delete_existing(conn: sqlite3.Connection, classifier_run_id: str) -> None:
    conn.execute("DELETE FROM classifier_predictions WHERE classifier_run_id = ?", (classifier_run_id,))
    conn.execute("DELETE FROM classifier_prediction_summary WHERE classifier_run_id = ?", (classifier_run_id,))
    conn.execute("DELETE FROM classifier_training_items WHERE classifier_run_id = ?", (classifier_run_id,))
    conn.execute("DELETE FROM classifier_oof_predictions WHERE classifier_run_id = ?", (classifier_run_id,))
    conn.execute("DELETE FROM classifier_runs WHERE classifier_run_id = ?", (classifier_run_id,))
    conn.commit()


def _parse_json_list(raw: object) -> list[str]:
    try:
        value = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []
