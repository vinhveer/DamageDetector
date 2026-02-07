from __future__ import annotations

import os
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from predict.dino import get_dino_service
from predict.unet import get_unet_service

from .dialogs import PredictDialog


class MainWindowSettingsMixin:
    def _settings_env_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / ".settings"

    def _format_env_value(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value)
        if text == "":
            return "\"\""
        if any(ch in text for ch in (" ", "\t", "\n", "\r", "#", "=", "\"", "'")):
            escaped = text.replace("\\", "\\\\").replace("\"", "\\\"")
            return f"\"{escaped}\""
        return text

    def _write_env_settings(self, data: dict) -> None:
        lines = []
        for key, value in data.items():
            lines.append(f"{key}={self._format_env_value(value)}")
        content = "\n".join(lines) + "\n"
        self._settings_env_path().write_text(content, encoding="utf-8")

    def _parse_env_text(self, text: str) -> dict:
        parsed: dict[str, str] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
                quote = value[0]
                inner = value[1:-1]
                if quote == "\"":
                    inner = inner.replace("\\\"", "\"").replace("\\\\", "\\")
                value = inner
            parsed[key] = value
        return parsed

    def _read_env_values(self) -> dict[str, str]:
        path = self._settings_env_path()
        if not path.is_file():
            return {}
        return self._parse_env_text(path.read_text(encoding="utf-8"))

    def _update_env_values(self, updates: dict[str, object]) -> None:
        data = self._read_env_values()
        for key, value in updates.items():
            data[str(key)] = "" if value is None else str(value)
        self._write_env_settings(data)

    def _load_env_settings(self, key_types: dict[str, type]) -> dict:
        path = self._settings_env_path()
        if not path.is_file():
            return {}
        data = self._parse_env_text(path.read_text(encoding="utf-8"))
        coerced: dict[str, object] = {}
        for key, val in data.items():
            if key not in key_types:
                continue
            target = key_types[key]
            if target is bool:
                low = str(val).strip().lower()
                coerced[key] = low in {"1", "true", "yes", "on"}
            elif target is int:
                try:
                    coerced[key] = int(val)
                except Exception:
                    continue
            elif target is float:
                try:
                    coerced[key] = float(val)
                except Exception:
                    continue
            else:
                coerced[key] = str(val)
        return coerced
    def _init_settings_persistence(self) -> None:
        self._isolate_last_labels: str = ""
        self._isolate_last_crop: bool = False
        self._isolate_last_white: bool = False
        self._ensure_hidden_model_widgets()
        self._loading_settings = False
        self._settings_save_timer = QtCore.QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(400)
        self._settings_save_timer.timeout.connect(self._save_settings)
        self._load_settings()
        self._bind_settings_autosave()

    def _ensure_hidden_model_widgets(self) -> None:
        if hasattr(self, "_sd_sam_ckpt") and hasattr(self, "_sd_gdino_ckpt") and hasattr(self, "_unet_model_edit"):
            return
        container = QtWidgets.QWidget(self)
        container.setVisible(False)
        container.setObjectName("_hidden_model_settings")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sam = self._build_tab_sam()
        dino = self._build_tab_dino()
        unet = self._build_tab_unet()
        sam.setParent(container)
        dino.setParent(container)
        unet.setParent(container)
        layout.addWidget(sam)
        layout.addWidget(dino)
        layout.addWidget(unet)
        layout.addStretch(1)
        self._hidden_model_settings = container

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._save_settings()
        except Exception:
            pass
        try:
            get_unet_service().close()
        except Exception:
            pass
        try:
            get_dino_service().close()
        except Exception:
            pass
        QtWidgets.QMainWindow.closeEvent(self, event)

    def _schedule_save_settings(self) -> None:
        if getattr(self, "_loading_settings", False):
            return
        if hasattr(self, "_settings_save_timer"):
            self._settings_save_timer.start()

    def _collect_settings(self) -> dict:
        cfg_data = self._sd_gdino_cfg.currentData()
        data = {
            # SAM
            "sam_checkpoint": self._sd_sam_ckpt.text().strip(),
            "sam_model_type": str(self._sd_sam_type.currentText()),
            "delta_type": str(self._sd_delta_type.currentText()),
            "delta_checkpoint": self._sd_delta_ckpt.text().strip(),
            "middle_dim": int(self._sd_middle_dim.value()),
            "scaling_factor": float(self._sd_scaling_factor.value()),
            "rank": int(self._sd_rank.value()),
            "invert_mask": bool(self._sd_invert.isChecked()),
            "min_area": int(self._sd_min_area.value()),
            "dilate": int(self._sd_dilate.value()),
            # DINO
            "dino_checkpoint": self._sd_gdino_ckpt.text().strip(),
            "dino_config_id": str(cfg_data) if cfg_data is not None else str(self._sd_gdino_cfg.currentText()),
            "device": str(self._sd_device.currentText()),
            "text_queries": self._sd_queries.text().strip(),
            "box_threshold": float(self._sd_box_thr.value()),
            "text_threshold": float(self._sd_text_thr.value()),
            "max_dets": int(self._sd_max_dets.value()),
            "isolate_labels": str(getattr(self, "_isolate_last_labels", "")),
            "isolate_crop": bool(getattr(self, "_isolate_last_crop", False)),
            "isolate_outside_white": bool(getattr(self, "_isolate_last_white", False)),
            # UNet
            "unet_model": self._unet_model_edit.text().strip(),
            "unet_threshold": float(self._unet_threshold.value()),
            "unet_post": bool(self._unet_post.isChecked()),
            "unet_mode": str(self._unet_mode.currentText()),
            "unet_input_size": int(self._unet_input_size.value()),
            "unet_overlap": int(self._unet_overlap.value()),
            "unet_tile_batch": int(self._unet_tile_batch.value()),
            # Editor
            "brush_radius": int(self._brush_slider.value()),
            "overlay_opacity": int(self._overlay_slider.value()),
        }
        if hasattr(self, "_workspace_root") and self._workspace_root is not None:
            data["workspace_path"] = str(self._workspace_root)
        else:
            data["workspace_path"] = ""
        return data

    def _apply_settings_dict(self, settings: dict) -> None:
        if not settings:
            return
        self._loading_settings = True
        try:
            def _set_text(attr: str, key: str) -> None:
                if key not in settings or not hasattr(self, attr):
                    return
                getattr(self, attr).setText(str(settings.get(key) or ""))

            def _set_checked(attr: str, key: str) -> None:
                if key not in settings or not hasattr(self, attr):
                    return
                getattr(self, attr).setChecked(bool(settings.get(key)))

            def _set_int(attr: str, key: str) -> None:
                if key not in settings or not hasattr(self, attr):
                    return
                getattr(self, attr).setValue(int(settings.get(key)))

            def _set_float(attr: str, key: str) -> None:
                if key not in settings or not hasattr(self, attr):
                    return
                getattr(self, attr).setValue(float(settings.get(key)))

            def _set_combo_text(attr: str, key: str) -> None:
                if key not in settings or not hasattr(self, attr):
                    return
                getattr(self, attr).setCurrentText(str(settings.get(key)))

            def _set_combo_data(attr: str, key: str) -> None:
                if key not in settings or not hasattr(self, attr):
                    return
                combo: QtWidgets.QComboBox = getattr(self, attr)
                target = str(settings.get(key) or "")
                for i in range(combo.count()):
                    if str(combo.itemData(i)) == target:
                        combo.setCurrentIndex(i)
                        return
                combo.setCurrentText(target)

            _set_text("_sd_sam_ckpt", "sam_checkpoint")
            _set_combo_text("_sd_sam_type", "sam_model_type")
            _set_combo_text("_sd_delta_type", "delta_type")
            _set_text("_sd_delta_ckpt", "delta_checkpoint")
            _set_int("_sd_middle_dim", "middle_dim")
            _set_float("_sd_scaling_factor", "scaling_factor")
            _set_int("_sd_rank", "rank")
            _set_checked("_sd_invert", "invert_mask")
            _set_int("_sd_min_area", "min_area")
            _set_int("_sd_dilate", "dilate")

            _set_text("_sd_gdino_ckpt", "dino_checkpoint")
            _set_combo_data("_sd_gdino_cfg", "dino_config_id")
            _set_combo_text("_sd_device", "device")
            _set_text("_sd_queries", "text_queries")
            _set_float("_sd_box_thr", "box_threshold")
            _set_float("_sd_text_thr", "text_threshold")
            _set_int("_sd_max_dets", "max_dets")
            self._isolate_last_labels = str(settings.get("isolate_labels") or "")
            self._isolate_last_crop = bool(settings.get("isolate_crop"))
            self._isolate_last_white = bool(settings.get("isolate_outside_white"))

            _set_text("_unet_model_edit", "unet_model")
            _set_float("_unet_threshold", "unet_threshold")
            _set_checked("_unet_post", "unet_post")
            _set_combo_text("_unet_mode", "unet_mode")
            _set_int("_unet_input_size", "unet_input_size")
            _set_int("_unet_overlap", "unet_overlap")
            _set_int("_unet_tile_batch", "unet_tile_batch")

            _set_int("_brush_slider", "brush_radius")
            _set_int("_overlay_slider", "overlay_opacity")
        finally:
            self._loading_settings = False

    def _save_settings(self) -> None:
        data = self._collect_settings()
        self._write_env_settings(data)

    def _load_settings(self) -> None:
        defaults = self._collect_settings()
        key_types: dict[str, type] = {
            "sam_checkpoint": str,
            "sam_model_type": str,
            "delta_type": str,
            "delta_checkpoint": str,
            "middle_dim": int,
            "scaling_factor": float,
            "rank": int,
            "invert_mask": bool,
            "min_area": int,
            "dilate": int,
            "dino_checkpoint": str,
            "dino_config_id": str,
            "device": str,
            "text_queries": str,
            "box_threshold": float,
            "text_threshold": float,
            "max_dets": int,
            "isolate_labels": str,
            "isolate_crop": bool,
            "isolate_outside_white": bool,
            "unet_model": str,
            "unet_threshold": float,
            "unet_post": bool,
            "unet_mode": str,
            "unet_input_size": int,
            "unet_overlap": int,
            "unet_tile_batch": int,
            "brush_radius": int,
            "overlay_opacity": int,
            "workspace_path": str,
        }

        loaded = dict(defaults)
        try:
            env_data = self._load_env_settings(key_types)
            if env_data:
                loaded.update(env_data)
        except Exception:
            pass

        self._loading_settings = True
        try:
            self._apply_settings_dict(loaded)
        finally:
            self._loading_settings = False

        # Always write .settings on startup to keep it in sync.
        try:
            self._write_env_settings(loaded)
        except Exception:
            pass

    def _bind_settings_autosave(self) -> None:
        def _connect_line(attr: str) -> None:
            w = getattr(self, attr, None)
            if isinstance(w, QtWidgets.QLineEdit):
                w.textChanged.connect(self._schedule_save_settings)

        def _connect_combo(attr: str) -> None:
            w = getattr(self, attr, None)
            if isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(self._schedule_save_settings)

        def _connect_check(attr: str) -> None:
            w = getattr(self, attr, None)
            if isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(self._schedule_save_settings)

        def _connect_spin(attr: str) -> None:
            w = getattr(self, attr, None)
            if isinstance(w, QtWidgets.QSpinBox) or isinstance(w, QtWidgets.QSlider):
                w.valueChanged.connect(self._schedule_save_settings)

        def _connect_dspin(attr: str) -> None:
            w = getattr(self, attr, None)
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                w.valueChanged.connect(self._schedule_save_settings)

        for a in ["_sd_sam_ckpt", "_sd_delta_ckpt", "_sd_gdino_ckpt", "_sd_queries"]:
            _connect_line(a)
        for a in ["_unet_model_edit"]:
            _connect_line(a)

        for a in ["_sd_sam_type", "_sd_delta_type", "_sd_gdino_cfg", "_sd_device", "_unet_mode"]:
            _connect_combo(a)

        for a in ["_sd_invert", "_unet_post"]:
            _connect_check(a)

        for a in [
            "_sd_middle_dim",
            "_sd_rank",
            "_sd_min_area",
            "_sd_dilate",
            "_sd_max_dets",
            "_unet_input_size",
            "_unet_overlap",
            "_unet_tile_batch",
        ]:
            _connect_spin(a)
        for a in ["_brush_slider", "_overlay_slider"]:
            _connect_spin(a)

        for a in ["_sd_scaling_factor", "_sd_box_thr", "_sd_text_thr", "_unet_threshold"]:
            _connect_dspin(a)

    def _open_model_settings_dialog(self, initial_page: str | None = None) -> bool:
        current = self._collect_settings()
        dlg = PredictDialog(
            self,
            title="Model Settings",
            mode="settings",
            settings=current,
            has_image=True,
            has_folder=True,
            show_scope=False,
            pages=["SAM", "DINO", "UNet"],
            ok_text="Apply",
            initial_page=initial_page,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        new_settings = dlg.settings_dict()
        self._apply_settings_dict(new_settings)
        try:
            self._save_settings()
        except Exception:
            self._schedule_save_settings()
        return True

    def _missing_settings_for_mode(self, mode: str) -> tuple[str, str] | None:
        mode = str(mode or "").strip().lower()
        if mode in {"sam_dino", "sam_dino_ft"}:
            sam_ckpt = self._sd_sam_ckpt.text().strip()
            if not sam_ckpt or not os.path.isfile(sam_ckpt):
                return ("SAM", "SAM checkpoint is required (file not found).")

            gdino_ckpt = self._sd_gdino_ckpt.text().strip()
            if not gdino_ckpt:
                return ("DINO", "GroundingDINO checkpoint is required.")
            lower = gdino_ckpt.lower()
            if lower.endswith((".pth", ".pt", ".safetensors", ".bin")) and not os.path.exists(gdino_ckpt):
                return ("DINO", f"GroundingDINO checkpoint not found: {gdino_ckpt}")

            if mode == "sam_dino_ft":
                delta_ckpt = self._sd_delta_ckpt.text().strip()
                if not delta_ckpt:
                    return ("SAM", "Delta checkpoint is required (set to 'auto' or choose a file).")
                dl = delta_ckpt.lower()
                if dl != "auto" and dl.endswith((".pth", ".pt", ".safetensors", ".bin")) and not os.path.exists(
                    delta_ckpt
                ):
                    return ("SAM", f"Delta checkpoint not found: {delta_ckpt}")

            return None

        if mode == "unet":
            model_path = self._unet_model_edit.text().strip()
            if not model_path or not os.path.isfile(model_path):
                return ("UNet", "UNet model is required (file not found).")

            # UNet + DINO mode requires DINO checkpoint
            # Check if pending ROI exists? No, this function checks GLOBAL settings readiness.
            # "Predict UNet + DINO" implies DINO is standard workflow now.
            gdino_ckpt = self._sd_gdino_ckpt.text().strip()
            if not gdino_ckpt:
                return ("DINO", "GroundingDINO checkpoint is required for UNet+DINO.")
            if not os.path.exists(gdino_ckpt):
                lower = gdino_ckpt.lower()
                if lower.endswith((".pth", ".pt", ".safetensors", ".bin")):
                    return ("DINO", f"GroundingDINO checkpoint not found: {gdino_ckpt}")

            return None

        return ("SAM", f"Unknown predict mode: {mode}")

    def _ensure_settings_ready(self, mode: str) -> bool:
        missing = self._missing_settings_for_mode(mode)
        if missing is None:
            return True

        initial_page, msg = missing
        QtWidgets.QMessageBox.information(self, "Model settings required", msg)
        if not self._open_model_settings_dialog(initial_page):
            return False

        missing2 = self._missing_settings_for_mode(mode)
        if missing2 is not None:
            _, msg2 = missing2
            QtWidgets.QMessageBox.warning(self, "Model settings", msg2)
            return False
        return True
