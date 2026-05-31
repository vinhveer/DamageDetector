from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceDetection:
    detection_id: int
    source_run_id: str
    source_input_dir: Path
    image_id: int
    image_rel_path: str
    stored_image_path: str
    image_name: str
    image_width: int
    image_height: int
    prompt_key: str
    detector_label: str
    detector_score: float
    x1: float
    y1: float
    x2: float
    y2: float


class Step2SemanticStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.ensure_schema()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS openclip_semantic_runs (
                semantic_run_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                source_db_path TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                source_stage TEXT NOT NULL,
                model_name TEXT NOT NULL,
                pretrained TEXT NOT NULL,
                device TEXT NOT NULL,
                prompt_config_json TEXT NOT NULL,
                options_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS openclip_semantic_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                semantic_run_id TEXT NOT NULL,
                source_detection_id INTEGER NOT NULL,
                source_run_id TEXT NOT NULL,
                image_id INTEGER NOT NULL,
                image_rel_path TEXT NOT NULL,
                image_path TEXT NOT NULL,
                prompt_key TEXT NOT NULL,
                detector_label TEXT NOT NULL,
                detector_score REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                crop_path TEXT NOT NULL,
                status TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                predicted_probability REAL NOT NULL,
                predicted_probability_pct REAL NOT NULL,
                top_prompt TEXT NOT NULL,
                error_type TEXT,
                error_message TEXT,
                raw_json TEXT NOT NULL,
                FOREIGN KEY(semantic_run_id) REFERENCES openclip_semantic_runs(semantic_run_id),
                FOREIGN KEY(source_detection_id) REFERENCES detections(detection_id),
                FOREIGN KEY(image_id) REFERENCES images(image_id),
                UNIQUE(semantic_run_id, source_detection_id)
            );

            CREATE TABLE IF NOT EXISTS openclip_semantic_scores (
                result_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                probability REAL NOT NULL,
                probability_pct REAL NOT NULL,
                PRIMARY KEY (result_id, label),
                FOREIGN KEY(result_id) REFERENCES openclip_semantic_results(result_id)
            );

            CREATE INDEX IF NOT EXISTS idx_openclip_semantic_results_detection
            ON openclip_semantic_results(source_detection_id);

            CREATE INDEX IF NOT EXISTS idx_openclip_semantic_results_run
            ON openclip_semantic_results(semantic_run_id, status);
            """
        )
        # C5: additive negative-scoring columns (preserve raw_json for before/after).
        existing = {str(row[1]) for row in self.conn.execute("PRAGMA table_info(openclip_semantic_results)")}
        for name in ("neg_penalty_json", "adjusted_scores_json"):
            if name not in existing:
                self.conn.execute(f"ALTER TABLE openclip_semantic_results ADD COLUMN {name} TEXT")
        self.conn.commit()

    def resolve_source_run_id(self, requested: str) -> str:
        raw = str(requested or "latest").strip()
        if raw and raw.lower() != "latest":
            return raw
        row = self.conn.execute(
            "SELECT run_id FROM runs ORDER BY created_at_utc DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("No damage_scan run found in SQLite.")
        return str(row["run_id"])

    def create_run(
        self,
        *,
        semantic_run_id: str,
        source_run_id: str,
        source_stage: str,
        model_name: str,
        pretrained: str,
        device: str,
        prompt_config: dict[str, Any],
        options: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO openclip_semantic_runs (
                semantic_run_id, created_at_utc, source_db_path, source_run_id, source_stage,
                model_name, pretrained, device, prompt_config_json, options_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                semantic_run_id,
                datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                str(self.db_path),
                source_run_id,
                source_stage,
                model_name,
                pretrained,
                device,
                json.dumps(prompt_config, ensure_ascii=False, sort_keys=True),
                json.dumps(options, ensure_ascii=False, sort_keys=True),
            ),
        )
        self.conn.commit()

    def list_source_detections(
        self,
        *,
        source_run_id: str,
        stage: str,
        limit: int,
        detection_ids: list[int],
    ) -> list[SourceDetection]:
        clauses = ["d.stage = ?"]
        params: list[Any] = [stage]
        if str(source_run_id).lower() != "all":
            clauses.append("d.run_id = ?")
            params.append(source_run_id)
        if detection_ids:
            placeholders = ", ".join("?" for _ in detection_ids)
            clauses.append(f"d.detection_id IN ({placeholders})")
            params.extend(int(item) for item in detection_ids)

        sql = (
            "SELECT d.detection_id, d.run_id AS source_run_id, r.input_dir AS source_input_dir, d.image_id, i.rel_path, i.path, i.name, i.width, i.height, "
            "d.prompt_key, d.label AS detector_label, d.score AS detector_score, d.x1, d.y1, d.x2, d.y2 "
            "FROM detections d JOIN images i ON i.image_id = d.image_id JOIN runs r ON r.run_id = d.run_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY d.detection_id"
        )
        if int(limit) > 0:
            sql = f"{sql} LIMIT {int(limit)}"

        rows = self.conn.execute(sql, params).fetchall()
        return [
            SourceDetection(
                detection_id=int(row["detection_id"]),
                source_run_id=str(row["source_run_id"]),
                source_input_dir=Path(str(row["source_input_dir"])).expanduser(),
                image_id=int(row["image_id"]),
                image_rel_path=str(row["rel_path"]),
                stored_image_path=str(row["path"]),
                image_name=str(row["name"]),
                image_width=int(row["width"]),
                image_height=int(row["height"]),
                prompt_key=str(row["prompt_key"]),
                detector_label=str(row["detector_label"]),
                detector_score=float(row["detector_score"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
            )
            for row in rows
        ]

    def insert_success_result(
        self,
        *,
        semantic_run_id: str,
        detection: SourceDetection,
        crop_path: str,
        classification: dict[str, Any],
    ) -> int:
        raw_json = json.dumps(classification, ensure_ascii=False, sort_keys=True)
        top_prompt = str(((classification.get("top_prompts") or [{}])[0]).get("prompt") or "")
        neg_penalty = classification.get("neg_penalty")
        adjusted_scores = classification.get("adjusted_scores")
        neg_penalty_json = json.dumps(neg_penalty, ensure_ascii=False, sort_keys=True) if neg_penalty is not None else None
        adjusted_scores_json = (
            json.dumps(adjusted_scores, ensure_ascii=False, sort_keys=True) if adjusted_scores is not None else None
        )
        row = self.conn.execute(
            """
            INSERT INTO openclip_semantic_results (
                semantic_run_id, source_detection_id, source_run_id, image_id, image_rel_path, image_path,
                prompt_key, detector_label, detector_score, x1, y1, x2, y2, crop_path, status,
                predicted_label, predicted_probability, predicted_probability_pct, top_prompt,
                error_type, error_message, raw_json, neg_penalty_json, adjusted_scores_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                semantic_run_id,
                detection.detection_id,
                detection.source_run_id,
                detection.image_id,
                detection.image_rel_path,
                detection.stored_image_path,
                detection.prompt_key,
                detection.detector_label,
                detection.detector_score,
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                crop_path,
                "ok",
                str(classification.get("predicted_label") or ""),
                float(classification.get("predicted_probability") or 0.0),
                float(classification.get("predicted_probability_pct") or 0.0),
                top_prompt,
                None,
                None,
                raw_json,
                neg_penalty_json,
                adjusted_scores_json,
            ),
        )
        result_id = int(row.lastrowid)
        score_rows = [
            (
                result_id,
                str(item.get("label") or ""),
                float(item.get("probability") or 0.0),
                float(item.get("probability_pct") or 0.0),
            )
            for item in list(classification.get("class_scores") or [])
        ]
        self.conn.executemany(
            "INSERT INTO openclip_semantic_scores (result_id, label, probability, probability_pct) VALUES (?, ?, ?, ?)",
            score_rows,
        )
        self.conn.commit()
        return result_id

    def insert_error_result(
        self,
        *,
        semantic_run_id: str,
        detection: SourceDetection,
        crop_path: str,
        exc: Exception,
    ) -> int:
        row = self.conn.execute(
            """
            INSERT INTO openclip_semantic_results (
                semantic_run_id, source_detection_id, source_run_id, image_id, image_rel_path, image_path,
                prompt_key, detector_label, detector_score, x1, y1, x2, y2, crop_path, status,
                predicted_label, predicted_probability, predicted_probability_pct, top_prompt,
                error_type, error_message, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                semantic_run_id,
                detection.detection_id,
                detection.source_run_id,
                detection.image_id,
                detection.image_rel_path,
                detection.stored_image_path,
                detection.prompt_key,
                detection.detector_label,
                detection.detector_score,
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                crop_path,
                "error",
                "",
                0.0,
                0.0,
                "",
                exc.__class__.__name__,
                str(exc),
                "{}",
            ),
        )
        self.conn.commit()
        return int(row.lastrowid)
