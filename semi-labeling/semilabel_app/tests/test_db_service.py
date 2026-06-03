from __future__ import annotations

import unittest
from pathlib import Path

from semilabel_app.paths import default_image_root, default_resemi_db
from semilabel_app.services import db_service
from semilabel_app.services.sampling import select_diverse_sample


RUN_ID = "myrun"


class DbServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = default_resemi_db()
        cls.image_root = default_image_root()
        if not Path(cls.db_path).is_file():
            raise unittest.SkipTest(f"Missing resemi DB: {cls.db_path}")

    def test_list_runs_contains_myrun(self) -> None:
        payload = db_service.list_runs(self.db_path)
        run_ids = {row["run_id"] for row in payload["runs"]}
        self.assertIn(RUN_ID, run_ids)

    def test_list_cleaned_limit_and_distribution(self) -> None:
        cleaned = db_service.list_cleaned(
            self.db_path,
            RUN_ID,
            str(self.image_root),
            final_label="crack",
            limit=5,
        )
        self.assertLessEqual(len(cleaned["items"]), 5)
        self.assertGreater(cleaned["total"], 0)
        if cleaned["items"]:
            self.assertEqual(cleaned["items"][0].final_label, "crack")

        dist = db_service.cleaned_distribution(self.db_path, RUN_ID)
        self.assertEqual(dist.total, cleaned["total"])
        self.assertGreater(len(dist.by_label), 0)

    def test_queue_query_keeps_reliability_order(self) -> None:
        queue = db_service.list_queue(
            self.db_path,
            RUN_ID,
            str(self.image_root),
            queue_type="suspect_broad_box",
            sample_ratio=0,
        )
        self.assertEqual(queue["queue_total"], queue["total"])
        scores = [item.reliability_score for item in queue["items"]]
        self.assertEqual(scores, sorted(scores))

    def test_prototype_query_is_stratified_without_core_join(self) -> None:
        sql = db_service.PROTOTYPE_CANDIDATE_SQL.lower()
        self.assertNotIn("core_cluster_members", sql)
        self.assertNotIn("core_clusters", sql)
        self.assertIn("partition by eff_label, band", sql)

        payload = db_service.list_prototype_candidates(
            self.db_path,
            RUN_ID,
            str(self.image_root),
            reject_below=0.5,
            per_band=10,
        )
        self.assertEqual(payload["labels"], ["crack", "mold", "spall", "reject"])
        self.assertGreater(len(payload["items"]), 0)
        self.assertTrue(payload["embeddingRunId"])

    def test_sampling_farthest_point_per_label(self) -> None:
        rows = [
            {"result_id": 1, "label": "crack", "reliability": 0.2, "vec": [1.0, 0.0]},
            {"result_id": 2, "label": "crack", "reliability": 0.9, "vec": [0.9, 0.1]},
            {"result_id": 3, "label": "crack", "reliability": 0.8, "vec": [0.0, 1.0]},
            {"result_id": 4, "label": "mold", "reliability": 0.4, "vec": [1.0, 0.0]},
            {"result_id": 5, "label": "mold", "reliability": 0.7, "vec": [0.0, 1.0]},
        ]
        picked = select_diverse_sample(rows, 0.5)
        self.assertIn(1, picked)
        self.assertIn(4, picked)
        self.assertEqual(len(picked), 3)


if __name__ == "__main__":
    unittest.main()
