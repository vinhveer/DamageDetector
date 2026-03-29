import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from PySide6 import QtGui

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dino import engine
from editor_app.controllers.history_controller import HistoryController
from editor_app.config.prediction_settings import DEFAULT_EDITOR_SETTINGS, migrate_editor_settings
from editor_app.services.run_storage import RunStorageService
from editor_app.stores.history_store import HistoryStore
from editor_app.stores.workspace_store import WorkspaceStore
from inference_api.prediction_models import DETECTION_DINO, SEGMENTATION_SAM, TASK_GROUP_MORE_DAMAGE, PredictionConfig
from inference_api.request_builder import build_prediction_request


class DinoValidTilingTests(unittest.TestCase):
    def test_opencv_valid_mask_preserves_internal_black_object(self) -> None:
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        image[70:130, 70:130] = 0
        valid_mask = engine._build_opencv_valid_mask(image)
        self.assertTrue(bool(valid_mask[100, 100]))
        self.assertTrue(bool(valid_mask[10, 10]))

    def test_generate_valid_tiles_applies_coverage_per_patch_only(self) -> None:
        valid_mask = np.ones((1024, 1024), dtype=bool)
        valid_mask[:, :220] = False
        roi_box = (0, 0, 1024, 1024)

        tiles, total_tiles, skipped_tiles, refined_tiles = engine._generate_valid_tiles(
            valid_mask,
            roi_box,
            tile_size=512,
            overlap=64,
            min_valid_coverage=0.85,
        )

        self.assertGreater(total_tiles, 0)
        self.assertGreater(len(tiles), 0)
        self.assertGreaterEqual(refined_tiles, 1)
        self.assertTrue(any(tile[5] == "refined" for tile in tiles))
        self.assertTrue(any((tile[2] - tile[0]) != (tile[3] - tile[1]) for tile in tiles))

    def test_compute_oriented_roi_rotates_diagonal_valid_region(self) -> None:
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        valid_mask = np.zeros((800, 800), dtype=np.uint8)
        rect = ((400, 400), (260, 520), 27.0)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(image, box, (255, 255, 255))
        cv2.fillConvexPoly(valid_mask, box, 1)

        original_roi = engine._mask_bbox(valid_mask > 0)
        original_occupancy = engine._compute_roi_occupancy(valid_mask > 0, original_roi)
        oriented = engine._compute_oriented_roi(image, valid_mask > 0)
        rotated_occupancy = engine._compute_roi_occupancy(oriented["valid_mask"], oriented["roi_box"])

        self.assertTrue(oriented["rotated"])
        self.assertGreater(abs(float(oriented["rotation_angle"])), 1.0)
        self.assertGreater(rotated_occupancy, original_occupancy)

    def test_map_box_from_rotated_to_original_preserves_location(self) -> None:
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        valid_mask = np.ones((300, 300), dtype=bool)
        _, _, matrix, inverse_matrix = engine._rotate_image_and_mask(image, valid_mask, 18.0)
        original_box = np.array([80.0, 90.0, 180.0, 210.0], dtype=np.float32)
        corners = np.array(
            [[[original_box[0], original_box[1]], [original_box[2], original_box[1]], [original_box[2], original_box[3]], [original_box[0], original_box[3]]]],
            dtype=np.float32,
        )
        rotated_corners = cv2.transform(corners, matrix)[0]
        rotated_bbox = np.array(
            [
                rotated_corners[:, 0].min(),
                rotated_corners[:, 1].min(),
                rotated_corners[:, 0].max(),
                rotated_corners[:, 1].max(),
            ],
            dtype=np.float32,
        )

        mapped_box = engine._map_box_from_rotated_to_original(
            rotated_bbox,
            inverse_matrix,
            original_width=image.shape[1],
            original_height=image.shape[0],
        )

        center_x = float((original_box[0] + original_box[2]) / 2.0)
        center_y = float((original_box[1] + original_box[3]) / 2.0)
        self.assertLessEqual(float(mapped_box[0]), center_x)
        self.assertGreaterEqual(float(mapped_box[2]), center_x)
        self.assertLessEqual(float(mapped_box[1]), center_y)
        self.assertGreaterEqual(float(mapped_box[3]), center_y)
        self.assertGreater(engine._box_iou(mapped_box, original_box), 0.3)

    def test_build_valid_mask_prefers_rasterio_for_tiff(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        raster_rgb = np.full((32, 32, 3), 120, dtype=np.uint8)
        raster_mask = np.ones((32, 32), dtype=bool)
        with mock.patch("dino.engine._load_tiff_with_rasterio", return_value=(raster_rgb, raster_mask)):
            rgb, valid_mask, strategy = engine._build_valid_mask("sample.tif", image)

        self.assertEqual(strategy, "rasterio_dataset_mask")
        self.assertTrue(np.array_equal(rgb, raster_rgb))
        self.assertTrue(np.array_equal(valid_mask, raster_mask))

    def test_normalize_band_to_uint8_handles_16bit_input(self) -> None:
        band = np.array(
            [
                [0, 1024, 2048],
                [4096, 8192, 16384],
                [32768, 49152, 65535],
            ],
            dtype=np.uint16,
        )
        valid_mask = np.ones_like(band, dtype=bool)

        normalized = engine._normalize_band_to_uint8(band, valid_mask)

        self.assertEqual(normalized.dtype, np.uint8)
        self.assertGreater(int(normalized.max()), int(normalized.min()))

    def test_box_is_fully_valid_rejects_any_invalid_overlap(self) -> None:
        valid_mask = np.ones((32, 32), dtype=bool)
        valid_mask[10, 10] = False
        integral = engine._mask_integral(valid_mask)

        self.assertTrue(engine._box_is_fully_valid(np.array([0, 0, 8, 8], dtype=np.float32), integral, width=32, height=32))
        self.assertFalse(engine._box_is_fully_valid(np.array([8, 8, 12, 12], dtype=np.float32), integral, width=32, height=32))

    def test_box_has_any_pure_black_pixels_detects_near_black_overlap(self) -> None:
        image_rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
        image_rgb[10, 10] = (8, 8, 8)
        black_integral = engine._pure_black_integral(image_rgb)

        self.assertFalse(engine._box_has_any_pure_black_pixels(np.array([0, 0, 8, 8], dtype=np.float32), black_integral, width=32, height=32))
        self.assertTrue(engine._box_has_any_pure_black_pixels(np.array([8, 8, 12, 12], dtype=np.float32), black_integral, width=32, height=32))

    def test_box_black_ratio_only_rejects_when_black_is_at_least_forty_percent(self) -> None:
        image_rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
        image_rgb[:, :8] = 0
        black_integral = engine._pure_black_integral(image_rgb)

        self.assertLess(engine._box_black_ratio(np.array([0, 0, 28, 20], dtype=np.float32), black_integral, width=32, height=32), 0.40)
        self.assertGreaterEqual(engine._box_black_ratio(np.array([0, 0, 20, 20], dtype=np.float32), black_integral, width=32, height=32), 0.40)
        self.assertGreaterEqual(engine._box_black_ratio(np.array([0, 0, 8, 20], dtype=np.float32), black_integral, width=32, height=32), 0.40)

    def test_history_refresh_invalidates_cached_items(self) -> None:
        storage = RunStorageService()
        workspace_store = WorkspaceStore()
        history_store = HistoryStore()
        controller = HistoryController(workspace_store, history_store, storage)

        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = pathlib.Path(tmpdir)
            workspace_store.results_root = results_root
            workspace_store.workspace_root = results_root
            run_dir = results_root / "2026-03-29_00-00-00_sam_dino_abcd12"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "workflow": "sam_dino",
                        "status": "running",
                        "created_at": "2026-03-29T00:00:00",
                        "run_dir": str(run_dir),
                        "output_dir": str(run_dir / "outputs"),
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "request.json").write_text(json.dumps({"image_path": "/tmp/current.png"}), encoding="utf-8")
            (run_dir / "result.json").write_text(json.dumps({"result": {}}), encoding="utf-8")

            controller.refresh()
            _bundle, items_before = controller.load_run_details(str(run_dir))
            self.assertEqual(len(items_before), 1)
            self.assertEqual(items_before[0].get("image_path"), "/tmp/current.png")

            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "result": {
                            "image_path": "/tmp/current.png",
                            "detections": [{"label": "house", "box": [1, 2, 3, 4], "score": 0.9}],
                        }
                    }
                ),
                encoding="utf-8",
            )

            controller.refresh()
            _bundle, items_after = controller.load_run_details(str(run_dir))
            self.assertEqual(len(items_after), 1)
            self.assertEqual(items_after[0].get("image_path"), "/tmp/current.png")
            self.assertEqual(len(items_after[0].get("detections") or []), 1)

    def test_more_damage_request_can_attach_unet_for_crack_masks(self) -> None:
        settings = migrate_editor_settings(DEFAULT_EDITOR_SETTINGS)
        settings["sam_checkpoint"] = "/tmp/sam.pth"
        settings["dino_checkpoint"] = "IDEA-Research/grounding-dino-base"
        settings["dino_config_id"] = "IDEA-Research/grounding-dino-base"
        settings["unet_model"] = "/tmp/unet.ckpt"
        settings["more_damage_crack_mask_model"] = "unet"
        config = PredictionConfig(
            task_group=TASK_GROUP_MORE_DAMAGE,
            segmentation_model=SEGMENTATION_SAM,
            detection_model=DETECTION_DINO,
            scope="current",
        )

        request = build_prediction_request(config, settings, image_path="/tmp/image.png", output_dir="/tmp/out")

        self.assertEqual(request.params.get("crack_mask_model"), "unet")
        self.assertIn("crack_unet", request.params)
        self.assertEqual(request.params["crack_unet"].get("task_group"), "crack_only")


if __name__ == "__main__":
    unittest.main()
