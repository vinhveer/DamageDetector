from __future__ import annotations

import threading
from dataclasses import asdict

from PySide6 import QtCore

from predict.dino import get_dino_service
from predict.unet import get_unet_service
from predict_sam_dino import SamDinoParams
from predict_unet import UnetParams


class WorkerBase(QtCore.QObject):
    log = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    def _stop_checker(self) -> bool:
        return self._stop_event.is_set()

    @QtCore.Slot()
    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        # Hard stop: kill service process so hangs stop instantly.
        try:
            get_unet_service().close()
        except Exception:
            pass
        try:
            get_dino_service().close()
        except Exception:
            pass
        self.log.emit("Stop requested (killed process)...")


class UnetWorker(WorkerBase):
    def __init__(self, image_path: str, params: UnetParams) -> None:
        super().__init__()
        self._image_path = image_path
        self._params = params

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.log.emit(f"UNet: image={self._image_path}")
            self.log.emit(f"UNet: model={self._params.model_path}")
            get_unet_service().call(
                "warmup",
                {"unet": asdict(self._params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            details = get_unet_service().call(
                "run",
                {"image_path": self._image_path, "params": asdict(self._params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            self.finished.emit(details)
        except Exception as e:
            if self._stop_checker():
                self.finished.emit({"stopped": True})
            else:
                self.failed.emit(str(e))


class SamDinoWorker(WorkerBase):
    def __init__(self, image_path: str, params: SamDinoParams) -> None:
        super().__init__()
        self._image_path = image_path
        self._params = params

    @QtCore.Slot()
    def run(self) -> None:
        try:
            get_dino_service().call(
                "warmup",
                {"params": asdict(self._params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            details = get_dino_service().call(
                "run",
                {"image_path": self._image_path, "params": asdict(self._params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            self.finished.emit(details)
        except Exception as e:
            if self._stop_checker():
                self.finished.emit({"stopped": True})
            else:
                self.failed.emit(str(e))


class BatchUnetWorker(WorkerBase):
    def __init__(self, image_paths: list[str], params: UnetParams) -> None:
        super().__init__()
        self._image_paths = image_paths
        self._params = params

    @QtCore.Slot()
    def run(self) -> None:
        try:
            get_unet_service().call(
                "warmup",
                {"unet": asdict(self._params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            details = get_unet_service().call(
                "batch_run",
                {"image_paths": list(self._image_paths), "params": asdict(self._params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            self.finished.emit(details)
        except Exception as e:
            if self._stop_checker():
                self.finished.emit({"stopped": True})
            else:
                self.failed.emit(str(e))


class BatchSamDinoWorker(WorkerBase):
    def __init__(self, image_paths: list[str], params: SamDinoParams) -> None:
        super().__init__()
        self._image_paths = image_paths
        self._params = params

    @QtCore.Slot()
    def run(self) -> None:
        try:
            get_dino_service().call(
                "warmup",
                {"params": asdict(self._params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            details = get_dino_service().call(
                "batch_run",
                {"image_paths": list(self._image_paths), "params": asdict(self._params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            self.finished.emit(details)
        except Exception as e:
            if self._stop_checker():
                self.finished.emit({"stopped": True})
            else:
                self.failed.emit(str(e))


class SamDinoIsolateWorker(WorkerBase):
    def __init__(
        self,
        image_path: str,
        params: SamDinoParams,
        *,
        target_labels: list[str],
        outside_value: int,
        crop_to_bbox: bool,
    ) -> None:
        super().__init__()
        self._image_path = image_path
        self._params = params
        self._target_labels = target_labels
        self._outside_value = outside_value
        self._crop_to_bbox = crop_to_bbox

    @QtCore.Slot()
    def run(self) -> None:
        try:
            get_dino_service().call(
                "warmup",
                {"params": asdict(self._params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            details = get_dino_service().call(
                "isolate",
                {
                    "image_path": self._image_path,
                    "params": asdict(self._params),
                    "target_labels": list(self._target_labels),
                    "outside_value": int(self._outside_value),
                    "crop_to_bbox": bool(self._crop_to_bbox),
                },
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            self.finished.emit(details)
        except Exception as e:
            if self._stop_checker():
                self.finished.emit({"stopped": True})
            else:
                self.failed.emit(str(e))


class UnetDinoWorker(WorkerBase):
    def __init__(self, image_path: str, unet_params: UnetParams, dino_params: SamDinoParams) -> None:
        super().__init__()
        self._image_path = image_path
        self._unet_params = unet_params
        self._dino_params = dino_params

    @QtCore.Slot()
    def run(self) -> None:
        try:
            # 1. Warmup
            self.log.emit("Warming up DINO...")
            get_dino_service().call(
                "warmup",
                {"params": asdict(self._dino_params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            self.log.emit("Warming up UNet...")
            get_unet_service().call(
                "warmup",
                {"unet": asdict(self._unet_params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )

            # 2. Run DINO to get boxes
            self.log.emit("Running DINO (Object detection)...")
            # We use "run" but we will ignore masks. Or we can add a specific method.
            # Using run is fine, it will generate masks but we can check detections.
            # Actually, SamDinoParams has SAM settings. We should ensure SAM runs fast or is skipped?
            # Creating a "detect_only" in DINO service would be cleaner.
            # For now, let's use "run" and just take the boxes from detections.
            
            dino_res = get_dino_service().call(
                "run",
                {"image_path": self._image_path, "params": asdict(self._dino_params)},
                log_fn=self.log.emit,
                stop_checker=self._stop_checker,
            )
            
            detections = dino_res.get("detections") or []
            if not detections:
                self.log.emit("No objects detected. Skipping UNet.")
                self.finished.emit(dino_res) # Return what we have
                return
                
            self.log.emit(f"DINO found {len(detections)} objects. Running UNet on these regions...")
            
            # 3. Call UNet with single unified ROI
            # We merge all detections into one bounding box (min_x, min_y, max_x, max_y)
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")
            
            has_roi = False
            for det in detections:
                box = det.get("box") # x1, y1, x2, y2
                if box:
                    x1, y1, x2, y2 = box
                    if x1 < min_x: min_x = x1
                    if y1 < min_y: min_y = y1
                    if x2 > max_x: max_x = x2
                    if y2 > max_y: max_y = y2
                    has_roi = True
            
            rois = []
            if has_roi:
                 # Add padding
                 pad = 50 
                 
                 # x1, y1, x2, y2
                 ux1 = int(min_x - pad)
                 uy1 = int(min_y - pad)
                 ux2 = int(max_x + pad)
                 uy2 = int(max_y + pad)
                 
                 rois.append((ux1, uy1, ux2, uy2))
                 self.log.emit(f"Unified ROI: {rois[0]}")
            else:
                 self.log.emit("No valid boxes found from DINO.")
                 self.finished.emit(dino_res)
                 return
            
            # We need to extend UNet service to accept batch ROIs or just call run multiple times.
            # Calling multiple times is easier to implement right now without changing service protocol much.
            # Or better: "batch_rois_run"
            
            # Let's iterate here.
            final_mask_path = dino_res.get("mask_path") # Wait, DINO generated a mask? We might want to overwrite or merge.
            # Actually UNet generally produces a new mask.
            
            # If we run UNet on ROIs, we need to stitch them back. 
            # UNet service doesn't have "run_on_rois".
            # Let's add "run_rois" to UNet service/worker.
            # Or just hack it here: call "run" with roi_box param multiple times.
            # But the result is a mask file. We need to merge them?
            # Or we can ask UNet to return the mask array?
            
            # To do this properly involves changing UNet service.
            # Alternative: Assume UNet tile logic can handle this? No.
            
            # Let's use a specialized method in UNet "run_rois"
            # Since I cannot easily change UNet service code right this moment without viewing it...
            # I will call "run_rois" and ensure UnetService handles it.
            
            unet_res = get_unet_service().call(
               "run_rois",
               {
                   "image_path": self._image_path, 
                   "params": asdict(self._unet_params),
                   "rois": rois
               },
               log_fn=self.log.emit,
               stop_checker=self._stop_checker,
            )
            
            # Pack results. unet_res has mask_path.
            # We want to show bounding boxes from DINO in the list?
            # Or just UNet result?
            # User wants "Mask" list.
            # We should pass DINO detections so they appear in list (maybe with UNet mask logic if we merged it?)
            # Actually, Unet run_rois returns one merged mask.
            # It doesn't return list of "objects".
            # If we want the list item to be "UnetDino_...", we need to form a detection dict for it.
            # But we have one merged mask.
            # So we create one "detection" representing the whole unet output?
            
            # OR we keep DINO detections but replace their masks with crops from UNet? Too complex.
            # "Mask" tab lists masks.
            # If UNet returns one full image mask. Then we have 1 mask.
            
            # We can return a single "detection" for the whole UNet result.
            
            final_det = {
                "label": "Merged",
                "score": 1.0, 
                "model_name": "UnetDino", # Tag for UI
                "mask_path": unet_res.get("mask_path"),
                "box": rois[0], # Return the unified bbox
                 # We can read mask and encode it to base64 if we want efficient transport or just let UI load from path?
                 # UI `_populate_mask_list` uses `mask_b64`. If missing, it uses `mask_path`?
                 # No, `_update_composite_mask` logic uses `mask_b64`.
                 # We must encode it here if we want it to be toggleable in the list.
            }
            
            # Let's read the generated mask and encode it.
            import cv2
            import base64
            import os
            import numpy as np
            
            mpath = unet_res.get("mask_path")
            if mpath and os.path.isfile(mpath):
                mask_img = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                     success, png_bytes = cv2.imencode(".png", mask_img)
                     if success:
                         final_det["mask_b64"] = base64.b64encode(png_bytes.tobytes()).decode('ascii')

            # We return this single detection
            result = {
                "image_path": self._image_path,
                "mask_path": unet_res.get("mask_path"),
                "overlay_path": unet_res.get("overlay_path"),
                "detections": [final_det]
            }
            
            self.finished.emit(result)
        except Exception as e:
            if self._stop_checker():
                self.finished.emit({"stopped": True})
            else:
                self.failed.emit(str(e))
