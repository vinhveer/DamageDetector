import os
import traceback

import torch

from ui.qt import QtCore

from unet.unet_model import UNet
from predict_lib import _find_gt_mask, _safe_basename, predict_image


class PredictWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, int, str)
    result = QtCore.Signal(int, dict)
    log = QtCore.Signal(str)
    failed = QtCore.Signal(int, str)
    finished = QtCore.Signal()

    def __init__(self, *, image_paths, input_dir, gt_dir, model_path, output_dir, threshold,
                 apply_postprocessing, recursive, mode, input_size, tile_overlap, tile_batch_size):
        super().__init__()
        self._stop = False
        self.image_paths = list(image_paths)
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.model_path = model_path
        self.output_dir = output_dir
        self.threshold = float(threshold)
        self.apply_postprocessing = bool(apply_postprocessing)
        self.recursive = bool(recursive)
        self.mode = str(mode)
        self.input_size = int(input_size)
        self.tile_overlap = int(tile_overlap)
        self.tile_batch_size = int(tile_batch_size)

    @QtCore.Slot()
    def stop(self):
        self._stop = True

    @QtCore.Slot()
    def run(self):
        total = len(self.image_paths)
        self.started.emit(total)

        device = torch.device("cpu")
        self.log.emit(f"Using device: {device}")

        model = UNet(in_channels=3, out_channels=1)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model = model.to(device)
        self.log.emit(f"Loaded model: {self.model_path}")

        overlap = (self.input_size // 2) if self.tile_overlap == 0 else self.tile_overlap

        for idx0, image_path in enumerate(self.image_paths):
            if self._stop:
                self.log.emit("Stopped by user.")
                break

            idx = idx0 + 1
            self.progress.emit(idx, total, image_path)

            try:
                output_basename = _safe_basename(self.input_dir, image_path)
                gt_mask_path = _find_gt_mask(self.gt_dir, self.input_dir, image_path) if self.gt_dir else None
                details = predict_image(
                    model,
                    image_path,
                    device,
                    threshold=self.threshold,
                    output_dir=self.output_dir,
                    apply_postprocessing=self.apply_postprocessing,
                    output_basename=output_basename,
                    gt_mask_path=gt_mask_path,
                    gt_expected=bool(self.gt_dir),
                    return_details=True,
                    mode=self.mode,
                    input_size=self.input_size,
                    tile_overlap=overlap,
                    tile_batch_size=self.tile_batch_size,
                )
                self.result.emit(idx0, details)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.failed.emit(idx0, msg)
                self.log.emit(msg)
                self.log.emit(traceback.format_exc())

        self.finished.emit()


class SinglePredictWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, int, str)
    result = QtCore.Signal(dict)
    log = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        *,
        image_path,
        gt_mask_path,
        model_path,
        output_dir,
        threshold,
        apply_postprocessing,
        roi_box,
        mode,
        input_size,
        tile_overlap,
        tile_batch_size,
    ):
        super().__init__()
        self._stop = False
        self.image_path = image_path
        self.gt_mask_path = gt_mask_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.threshold = float(threshold)
        self.apply_postprocessing = bool(apply_postprocessing)
        self.roi_box = roi_box
        self.mode = str(mode)
        self.input_size = int(input_size)
        self.tile_overlap = int(tile_overlap)
        self.tile_batch_size = int(tile_batch_size)

    @QtCore.Slot()
    def stop(self):
        self._stop = True

    @QtCore.Slot()
    def run(self):
        self.started.emit(1)
        self.progress.emit(1, 1, self.image_path)
        try:
            device = torch.device("cpu")
            self.log.emit(f"Using device: {device}")

            model = UNet(in_channels=3, out_channels=1)
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            model = model.to(device)
            self.log.emit(f"Loaded model: {self.model_path}")

            overlap = (self.input_size // 2) if self.tile_overlap == 0 else self.tile_overlap
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            details = predict_image(
                model,
                self.image_path,
                device,
                threshold=self.threshold,
                output_dir=self.output_dir,
                apply_postprocessing=self.apply_postprocessing,
                output_basename=base,
                gt_mask_path=self.gt_mask_path,
                gt_expected=bool(self.gt_mask_path),
                return_details=True,
                roi_box=self.roi_box,
                mode=self.mode,
                input_size=self.input_size,
                tile_overlap=overlap,
                tile_batch_size=self.tile_batch_size,
            )
            self.result.emit(details)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self.failed.emit(msg)
            self.log.emit(msg)
            self.log.emit(traceback.format_exc())
        self.finished.emit()
