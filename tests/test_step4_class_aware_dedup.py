import math
import pathlib
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


STEP4_DIR = pathlib.Path(__file__).resolve().parents[1] / "semi-labeling" / "step4_class_aware_dedup"
sys.path.insert(0, str(STEP4_DIR))

from crack_topology import angle_difference, orientation
from dedup_detections import main as dedup_main
from nms_cluster import weighted_box_fusion
from pair_features import pair_features
from source_store import Detection


class Step4ClassAwareDedupTests(unittest.TestCase):
    def _detection(
        self,
        *,
        result_id: int,
        label: str = "crack",
        box: tuple[float, float, float, float] = (10.0, 10.0, 60.0, 60.0),
        detector_score: float = 0.9,
        semantic_pct: float = 90.0,
        image_rel_path: str = "sample.png",
        image_root: pathlib.Path | None = None,
    ) -> Detection:
        x1, y1, x2, y2 = box
        image_path = "sample.png" if image_root is None else str(image_root / image_rel_path)
        return Detection(
            result_id=result_id,
            source_detection_id=result_id,
            image_id=1,
            image_rel_path=image_rel_path,
            image_path=image_path,
            source_input_dir="" if image_root is None else str(image_root),
            predicted_label=label,
            detector_label="damage",
            detector_score=detector_score,
            predicted_probability=semantic_pct / 100.0,
            predicted_probability_pct=semantic_pct,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            image_width=200,
            image_height=200,
            crop_path="",
        )

    def test_pair_features_identical_boxes_are_maximal(self) -> None:
        box = self._detection(result_id=1)
        features = pair_features(box, box, np.array([1.0, 0.0], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32))

        self.assertAlmostEqual(features["iou"], 1.0)
        self.assertAlmostEqual(features["cos_sim"], 1.0)
        self.assertAlmostEqual(features["center_sim"], 1.0)
        self.assertAlmostEqual(features["size_sim"], 1.0)
        self.assertAlmostEqual(features["aspect_sim"], 1.0)

    def test_crack_topology_perpendicular_boxes_have_right_angle_diff(self) -> None:
        horizontal = self._detection(result_id=1, box=(0.0, 0.0, 100.0, 20.0))
        vertical = self._detection(result_id=2, box=(0.0, 0.0, 20.0, 100.0))

        self.assertAlmostEqual(angle_difference(orientation(horizontal), orientation(vertical)), math.pi / 2.0)

    def test_weighted_box_fusion_stays_inside_convex_hull(self) -> None:
        boxes = [
            self._detection(result_id=1, box=(10.0, 10.0, 50.0, 50.0)),
            self._detection(result_id=2, box=(12.0, 8.0, 52.0, 48.0)),
            self._detection(result_id=3, box=(8.0, 12.0, 48.0, 52.0)),
        ]

        fused = weighted_box_fusion(boxes, [0.8, 0.6, 0.4])

        self.assertGreaterEqual(fused[0], 8.0)
        self.assertGreaterEqual(fused[1], 8.0)
        self.assertLessEqual(fused[2], 52.0)
        self.assertLessEqual(fused[3], 52.0)

    def _create_fixture(
        self,
        tmpdir: pathlib.Path,
        rows: list[dict[str, object]],
        *,
        image_size: tuple[int, int] = (200, 200),
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        image_root = tmpdir / "images"
        image_root.mkdir()
        image_path = image_root / "sample.png"
        Image.new("RGB", image_size, color=(180, 180, 180)).save(image_path)
        source_db = tmpdir / "source.sqlite3"
        embedding_db = tmpdir / "embeddings.sqlite3"

        conn = sqlite3.connect(str(source_db))
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                input_dir TEXT NOT NULL,
                db_path TEXT NOT NULL,
                detector_name TEXT NOT NULL,
                checkpoint TEXT NOT NULL,
                device TEXT NOT NULL,
                config_json TEXT NOT NULL
            );
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'ok'
            );
            CREATE TABLE openclip_semantic_runs (
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
            CREATE TABLE openclip_semantic_results (
                result_id INTEGER PRIMARY KEY,
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
                raw_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("run1", "2026-01-01T00:00:00+00:00", str(image_root), str(source_db), "dino", "", "cpu", "{}"),
        )
        conn.execute(
            "INSERT INTO images (image_id, run_id, rel_path, path, name, width, height, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "run1", "sample.png", str(image_path), "sample.png", image_size[0], image_size[1], "ok"),
        )
        conn.execute(
            "INSERT INTO openclip_semantic_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("sem1", "2026-01-01T00:00:00+00:00", str(source_db), "run1", "final", "openclip", "", "cpu", "{}", "{}"),
        )
        for row in rows:
            x1, y1, x2, y2 = row["box"]
            semantic_pct = float(row.get("semantic_pct", 90.0))
            conn.execute(
                """
                INSERT INTO openclip_semantic_results (
                    result_id, semantic_run_id, source_detection_id, source_run_id, image_id,
                    image_rel_path, image_path, prompt_key, detector_label, detector_score,
                    x1, y1, x2, y2, crop_path, status, predicted_label,
                    predicted_probability, predicted_probability_pct, top_prompt,
                    error_type, error_message, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row["result_id"]),
                    "sem1",
                    int(row["result_id"]),
                    "run1",
                    1,
                    "sample.png",
                    str(image_path),
                    "damage",
                    "damage",
                    float(row.get("detector_score", 0.9)),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    "",
                    "ok",
                    str(row.get("label", "crack")),
                    semantic_pct / 100.0,
                    semantic_pct,
                    "prompt",
                    None,
                    None,
                    "{}",
                ),
            )
        conn.commit()
        conn.close()

        emb_conn = sqlite3.connect(str(embedding_db))
        emb_conn.executescript(
            """
            CREATE TABLE embedding_runs (
                embedding_run_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                source_db_path TEXT NOT NULL,
                source_semantic_run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                device TEXT NOT NULL,
                padding_ratio REAL NOT NULL,
                total_detections INTEGER NOT NULL,
                embedded_count INTEGER NOT NULL,
                skipped_count INTEGER NOT NULL,
                options_json TEXT NOT NULL
            );
            CREATE TABLE detection_embeddings (
                embedding_run_id TEXT NOT NULL,
                result_id INTEGER NOT NULL,
                image_rel_path TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                embedding_blob BLOB NOT NULL,
                PRIMARY KEY (embedding_run_id, result_id)
            );
            """
        )
        emb_conn.execute(
            "INSERT INTO embedding_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("emb1", "2026-01-01T00:00:00+00:00", str(source_db), "sem1", "toy", 4, "cpu", 0.0, len(rows), len(rows), 0, "{}"),
        )
        for row in rows:
            vector = np.asarray(row.get("embedding", [1.0, 0.0, 0.0, 0.0]), dtype="<f4")
            emb_conn.execute(
                "INSERT INTO detection_embeddings VALUES (?, ?, ?, ?, ?)",
                ("emb1", int(row["result_id"]), "sample.png", str(row.get("label", "crack")), vector.tobytes()),
            )
        emb_conn.commit()
        emb_conn.close()
        return source_db, embedding_db, image_root

    def test_cli_drops_exact_duplicate_and_low_quality_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            source_db, embedding_db, image_root = self._create_fixture(
                tmpdir,
                [
                    {"result_id": 1, "label": "spall", "box": (10.0, 10.0, 60.0, 60.0), "embedding": [1.0, 0.0, 0.0, 0.0]},
                    {"result_id": 2, "label": "spall", "box": (10.0, 10.0, 60.0, 60.0), "embedding": [1.0, 0.0, 0.0, 0.0]},
                    {
                        "result_id": 3,
                        "label": "crack",
                        "box": (80.0, 80.0, 190.0, 190.0),
                        "semantic_pct": 10.0,
                        "detector_score": 0.0,
                        "embedding": [0.0, 1.0, 0.0, 0.0],
                    },
                ],
            )
            output_db = tmpdir / "dedup.sqlite3"

            exit_code = dedup_main(
                [
                    "--source-db",
                    str(source_db),
                    "--embedding-db",
                    str(embedding_db),
                    "--output-db",
                    str(output_db),
                    "--semantic-run-id",
                    "sem1",
                    "--embedding-run-id",
                    "emb1",
                    "--image-root",
                    str(image_root),
                    "--quality-min",
                    "0.8",
                    "--keeper-mode-table",
                    '{"crack":"keep_one","spall":"keep_one","mold":"wbf"}',
                    "--log-every",
                    "0",
                ]
            )

            self.assertEqual(exit_code, 0)
            conn = sqlite3.connect(str(output_db))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM dedup_results ORDER BY result_id").fetchall()
            counts = conn.execute("SELECT kept_count, fused_count, dropped_count, total_detections FROM dedup_runs").fetchone()
            pair_count = conn.execute("SELECT COUNT(*) FROM dedup_pair_scores").fetchone()[0]
            conn.close()

            self.assertEqual(len(rows), 3)
            self.assertEqual(tuple(counts), (1, 0, 2, 3))
            self.assertGreaterEqual(pair_count, 1)
            duplicate_reasons = {int(row["result_id"]): str(row["drop_reason"]) for row in rows if int(row["keep"]) == 0}
            self.assertIn("duplicate", duplicate_reasons.values())
            self.assertEqual(duplicate_reasons[3], "low_quality")

    def test_cli_preserves_perpendicular_crack_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            source_db, embedding_db, image_root = self._create_fixture(
                tmpdir,
                [
                    {
                        "result_id": 1,
                        "label": "crack",
                        "box": (0.0, 0.0, 100.0, 70.0),
                        "detector_score": 1.0,
                        "embedding": [1.0, 0.0, 0.0, 0.0],
                    },
                    {
                        "result_id": 2,
                        "label": "crack",
                        "box": (15.0, -15.0, 85.0, 85.0),
                        "detector_score": 0.0,
                        "embedding": [0.0, 1.0, 0.0, 0.0],
                    },
                ],
                image_size=(120, 120),
            )
            output_db = tmpdir / "dedup.sqlite3"

            dedup_main(
                [
                    "--source-db",
                    str(source_db),
                    "--embedding-db",
                    str(embedding_db),
                    "--output-db",
                    str(output_db),
                    "--semantic-run-id",
                    "sem1",
                    "--embedding-run-id",
                    "emb1",
                    "--image-root",
                    str(image_root),
                    "--quality-min",
                    "0.0",
                    "--keeper-mode-table",
                    '{"crack":"keep_one","spall":"wbf","mold":"wbf"}',
                    "--log-every",
                    "0",
                ]
            )

            conn = sqlite3.connect(str(output_db))
            keeps = [row[0] for row in conn.execute("SELECT keep FROM dedup_results ORDER BY result_id").fetchall()]
            p_dup = conn.execute("SELECT p_dup FROM dedup_pair_scores").fetchone()[0]
            conn.close()

            self.assertEqual(keeps, [1, 1])
            self.assertLess(p_dup, 0.5)


if __name__ == "__main__":
    unittest.main()
