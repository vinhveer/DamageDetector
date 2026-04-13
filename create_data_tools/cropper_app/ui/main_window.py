from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from create_data_tools.cropper_app.domain import Roi
from create_data_tools.cropper_app.roi_db import RoiDatabase
from create_data_tools.cropper_app.services.export_service import export_rois
from create_data_tools.cropper_app.ui.image_view import ImageRoiView, SquareRoiItem


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("crop_images")
        self.resize(1500, 900)

        self._base_dir: Path | None = None
        self._db: RoiDatabase | None = None
        self._images: list[Path] = []
        self._current_image: Path | None = None

        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage("File > Open Folder... để bắt đầu")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._db is not None:
            self._db.close()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        self._btn_add_roi = QtWidgets.QPushButton("Thêm ROI", central)
        self._btn_add_roi.clicked.connect(self._add_roi)
        top.addWidget(self._btn_add_roi)
        top.addStretch(1)
        root.addLayout(top)

        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        self._splitter.setChildrenCollapsible(False)

        left = QtWidgets.QWidget(self._splitter)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(QtWidgets.QLabel("Danh sách hình", left))
        self._image_list = QtWidgets.QListWidget(left)
        self._image_list.currentItemChanged.connect(self._on_image_selected)
        left_layout.addWidget(self._image_list, 2)

        left_layout.addWidget(QtWidgets.QLabel("ROIs của hình đang chọn", left))
        self._roi_list = QtWidgets.QListWidget(left)
        self._roi_list.currentItemChanged.connect(self._on_roi_selected_from_list)
        self._roi_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._roi_list.customContextMenuRequested.connect(self._show_roi_context_menu)
        left_layout.addWidget(self._roi_list, 1)

        self._view = ImageRoiView(self._splitter)
        self._view.roiSelectionChanged.connect(self._on_roi_selected_from_view)
        self._view.addRoiRequested.connect(self._add_roi)
        self._view.deleteRequested.connect(self._delete_roi_quiet)

        self._splitter.addWidget(left)
        self._splitter.addWidget(self._view)
        self._splitter.setSizes([420, 1080])
        root.addWidget(self._splitter, 1)

        self.setCentralWidget(central)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("File")

        open_folder = QtGui.QAction("Open Folder...", self)
        open_folder.triggered.connect(self._open_folder_dialog)
        menu.addAction(open_folder)

        export_action = QtGui.QAction("Export hình theo ROI...", self)
        export_action.triggered.connect(self._export_dialog)
        menu.addAction(export_action)

        menu.addSeparator()
        quit_action = QtGui.QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)

    def _db_required(self) -> RoiDatabase | None:
        if self._db is None:
            QtWidgets.QMessageBox.information(self, "Chưa có folder", "Vui lòng chọn File > Open Folder... trước.")
            return None
        return self._db

    def _open_folder_dialog(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Chọn folder hình", str(self._base_dir or Path.cwd()))
        if not folder:
            return
        self._open_folder(Path(folder))

    def _open_folder(self, folder: Path) -> None:
        folder = Path(folder)
        self._base_dir = folder
        if self._db is not None:
            self._db.close()
        self._db = RoiDatabase(folder / "crop_rois.sqlite3")

        self._images = self._scan_images(folder)
        self._image_list.clear()
        for p in self._images:
            item = QtWidgets.QListWidgetItem(str(p.relative_to(folder)))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(p))
            self._image_list.addItem(item)

        if self._images:
            self._image_list.setCurrentRow(0)
        else:
            self._current_image = None
            self._view.clear_image()
            self._roi_list.clear()

        self.statusBar().showMessage(f"Base: {folder} — DB: {self._db.path}")

    def _scan_images(self, folder: Path) -> list[Path]:
        results: list[Path] = []
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in IMAGE_EXTS:
                results.append(p)
        results.sort(key=lambda x: str(x).lower())
        return results

    def _image_key(self, path: Path) -> str:
        if self._base_dir is None:
            return str(path.name)
        return str(path.relative_to(self._base_dir)).replace("\\", "/")

    def _load_image(self, path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            return
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(self, "Không mở được hình", str(path))
            return

        self._current_image = path
        self._view.set_image_pixmap(pixmap)
        self._refresh_rois()

    def _refresh_rois(self) -> None:
        db = self._db_required()
        if db is None or self._current_image is None:
            self._roi_list.clear()
            self._view.set_rois([])
            return

        key = self._image_key(self._current_image)
        rois = db.list_rois(key)

        items: list[SquareRoiItem] = []
        for roi in rois:
            item = SquareRoiItem(roi.id, x=roi.x, y=roi.y, size=roi.size, image_bounds=self._view.scene().sceneRect())
            item.geometryCommitted.connect(self._on_roi_geometry_committed)
            items.append(item)
        self._view.set_rois(items)

        self._roi_list.clear()
        for roi in rois:
            text = f"{roi.name} — x={roi.x} y={roi.y} s={roi.size}"
            li = QtWidgets.QListWidgetItem(text)
            li.setData(QtCore.Qt.ItemDataRole.UserRole, int(roi.id))
            self._roi_list.addItem(li)

    def _on_image_selected(self, current: QtWidgets.QListWidgetItem | None, previous: QtWidgets.QListWidgetItem | None) -> None:  # noqa: ANN001
        del previous
        if current is None:
            return
        path_str = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        self._load_image(Path(str(path_str)))

    def _add_roi(self) -> None:
        db = self._db_required()
        if db is None or self._current_image is None:
            return

        bounds = self._view.scene().sceneRect()
        if bounds.isNull() or bounds.width() <= 0 or bounds.height() <= 0:
            return

        default_size = int(max(32, min(bounds.width(), bounds.height()) * 0.2))
        x = int((bounds.width() - default_size) / 2)
        y = int((bounds.height() - default_size) / 2)

        image_key = self._image_key(self._current_image)
        roi_count = len(db.list_rois(image_key))
        roi = db.create_roi(image_key, name=f"ROI {roi_count + 1}", x=x, y=y, size=default_size)
        self._refresh_rois()

        item = self._view.roi_item(roi.id)
        if item is not None:
            item.setSelected(True)
            self._view.centerOn(item)

        self._select_roi_in_list(roi.id)

    def _select_roi_in_list(self, roi_id: int) -> None:
        for i in range(self._roi_list.count()):
            item = self._roi_list.item(i)
            if int(item.data(QtCore.Qt.ItemDataRole.UserRole)) == int(roi_id):
                self._roi_list.setCurrentRow(i)
                return

    def _on_roi_selected_from_list(self, current: QtWidgets.QListWidgetItem | None, previous: QtWidgets.QListWidgetItem | None) -> None:  # noqa: ANN001
        del previous
        if current is None:
            return
        roi_id = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if roi_id is None:
            return
        item = self._view.roi_item(int(roi_id))
        if item is not None:
            item.setSelected(True)
            self._view.centerOn(item)

    def _on_roi_selected_from_view(self, roi_id: int | None) -> None:
        if roi_id is None:
            self._roi_list.setCurrentRow(-1)
            return
        self._select_roi_in_list(int(roi_id))

    def _on_roi_geometry_committed(self, roi_id: int, geom) -> None:  # noqa: ANN001
        db = self._db_required()
        if db is None:
            return
        db.update_roi(int(roi_id), x=int(geom.x), y=int(geom.y), size=int(geom.size))
        self._refresh_rois()
        self._select_roi_in_list(int(roi_id))

    def _show_roi_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._roi_list.itemAt(pos)
        if item is None:
            return
        roi_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if roi_id is None:
            return

        menu = QtWidgets.QMenu(self)
        delete_action = menu.addAction("Xóa ROI")
        action = menu.exec(self._roi_list.mapToGlobal(pos))
        if action == delete_action:
            self._delete_roi(int(roi_id))

    def _delete_roi(self, roi_id: int) -> None:
        db = self._db_required()
        if db is None:
            return
        if (
            QtWidgets.QMessageBox.question(self, "Xóa ROI", f"Xóa ROI {roi_id} ?")
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return
        db.delete_roi(int(roi_id))
        self._refresh_rois()

    def _delete_roi_quiet(self, roi_id: int) -> None:
        db = self._db_required()
        if db is None:
            return
        db.delete_roi(int(roi_id))
        self._refresh_rois()

    def _export_dialog(self) -> None:
        db = self._db_required()
        if db is None or self._base_dir is None:
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(self, "Chọn folder export", str(self._base_dir))
        if not out:
            return

        out_dir = Path(out)
        items: list[tuple[Path, list[Roi]]] = []
        for img in self._images:
            rois = db.list_rois(self._image_key(img))
            if rois:
                items.append((img, rois))

        result = export_rois(base_dir=self._base_dir, out_dir=out_dir, items=items)
        self.statusBar().showMessage(f"Exported: {result.exported} — Skipped: {result.skipped} — Out: {out_dir}")
