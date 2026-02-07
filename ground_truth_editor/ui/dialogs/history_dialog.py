from __future__ import annotations

import csv
from pathlib import Path

from PySide6 import QtCore, QtWidgets


class HistoryBaseDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, title: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 500)
        self._layout = QtWidgets.QVBoxLayout(self)
        
    def _create_tree(self, columns: list[str]) -> QtWidgets.QTreeWidget:
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(columns)
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(False)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        return tree


class FolderHistoryDialog(HistoryBaseDialog):
    def __init__(self, parent: QtWidgets.QWidget, results_root: Path) -> None:
        super().__init__(parent, "Folder History")
        self._results_root = results_root
        self._selected_run_file: Path | None = None
        self._selected_image_rel: str | None = None

        # Splitter: Left = Runs, Right = Images in Run
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._layout.addWidget(splitter)

        # Runs Panel
        runs_widget = QtWidgets.QWidget()
        runs_layout = QtWidgets.QVBoxLayout(runs_widget)
        runs_layout.setContentsMargins(0, 0, 0, 0)
        runs_layout.addWidget(QtWidgets.QLabel("Prediction Scans (Lần quét):"))
        
        self._runs_tree = self._create_tree(["Run ID", "Date", "Model", "Images"])
        self._runs_tree.itemSelectionChanged.connect(self._on_run_selected)
        runs_layout.addWidget(self._runs_tree, 2)
        
        runs_layout.addWidget(QtWidgets.QLabel("Scan Details:"))
        self._details_text = QtWidgets.QTextEdit()
        self._details_text.setReadOnly(True)
        runs_layout.addWidget(self._details_text, 1)
        
        splitter.addWidget(runs_widget)

        # Images Panel
        imgs_widget = QtWidgets.QWidget()
        imgs_layout = QtWidgets.QVBoxLayout(imgs_widget)
        imgs_layout.setContentsMargins(0, 0, 0, 0)
        imgs_layout.addWidget(QtWidgets.QLabel("Images in Run:"))

        self._imgs_tree = self._create_tree(["Image", "Detections", "Max Score"])
        self._imgs_tree.itemDoubleClicked.connect(self._on_image_double_clicked)
        imgs_layout.addWidget(self._imgs_tree)
        splitter.addWidget(imgs_widget)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self._apply_btn = QtWidgets.QPushButton("Apply Scan (Load All Results)")
        self._apply_btn.clicked.connect(self._on_apply)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self._apply_btn)
        self._layout.addLayout(btn_layout)
        
        splitter.setSizes([400, 400])
        self._populate_runs()

    def _populate_runs(self) -> None:
        if not self._results_root.exists():
            return

        # Find all run CSVs
        run_files = sorted(self._results_root.glob("*_lan_quet_workspace.csv"), reverse=True)
        
        for p in run_files:
            # Parse CSV to get summary (Model, Date from first row, Image count)
            try:
                model = "Unknown"
                created = "Unknown"
                images = set()
                with p.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if model == "Unknown":
                            model = row.get("model") or "Unknown"
                        if created == "Unknown":
                            created = row.get("created_at") or "Unknown"
                        img = row.get("image_rel") or row.get("image_path")
                        if img:
                            images.add(img)
                
                item = QtWidgets.QTreeWidgetItem(self._runs_tree)
                name = p.name.replace("_lan_quet_workspace.csv", "")
                if len(model) > 20:
                     model = model[:17] + "..."
                item.setText(0, name)
                item.setText(1, created)
                item.setText(2, model)
                item.setText(3, str(len(images)))
                item.setData(0, QtCore.Qt.UserRole, str(p))
            except Exception:
                continue

    def _on_run_selected(self) -> None:
        self._imgs_tree.clear()
        self._details_text.clear()
        self._selected_run_file = None
        self._selected_image_rel = None

        sel = self._runs_tree.selectedItems()
        if not sel:
            return

        path_str = sel[0].data(0, QtCore.Qt.UserRole)
        self._selected_run_file = Path(path_str)
        
        # Load details/metadata if exists
        run_name = self._selected_run_file.name.replace("_lan_quet_workspace.csv", "")
        info_path = self._selected_run_file.parent / f"{run_name}_info.txt"
        if info_path.exists():
             try:
                 text = info_path.read_text(encoding="utf-8")
                 self._details_text.setText(text)
             except:
                 self._details_text.setText("(Could not read info.txt)")
        else:
             self._details_text.setText("(No info.txt saved for this run)")

        # Parse again to group by image
        by_image = {}
        try:
            with self._selected_run_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img = row.get("image_rel") or row.get("image_path")
                    if not img:
                        continue
                    by_image.setdefault(img, []).append(row)
        except Exception:
            return

        for img_rel, rows in sorted(by_image.items()):
            count = len(rows)
            max_score = 0.0
            for r in rows:
                try:
                    s = float(r.get("score") or 0)
                    if s > max_score:
                        max_score = s
                except:
                    pass
            
            item = QtWidgets.QTreeWidgetItem(self._imgs_tree)
            item.setText(0, img_rel)
            item.setText(1, str(count))
            item.setText(2, f"{max_score:.2f}")
            item.setData(0, QtCore.Qt.UserRole, img_rel)

        # Connect selection to "Done"
        # We use itemClicked so single click works
        self._imgs_tree.itemClicked.connect(self._on_image_clicked)

    def _on_image_clicked(self, item: QtWidgets.QTreeWidgetItem, col: int) -> None:
         self._selected_image_rel = item.data(0, QtCore.Qt.UserRole)
         # "Simple select -> Done"
         self.accept()

    def _on_image_double_clicked(self, item: QtWidgets.QTreeWidgetItem, col: int) -> None:
        # Redundant but safe
        self._on_image_clicked(item, col)

    def _on_apply(self) -> None:
        if self._selected_run_file:
            # Reset image selection so we treat it as 'whole run'
            self._selected_image_rel = None
            self.accept()

    def get_result(self) -> tuple[Path, str | None] | None:
        if self._selected_run_file:
            return self._selected_run_file, self._selected_image_rel
        return None


class ImageHistoryDialog(HistoryBaseDialog):
    def __init__(self, parent: QtWidgets.QWidget, data_csv: Path) -> None:
        super().__init__(parent, "Image History")
        self._data_csv = data_csv
        self._selected_run_id: str | None = None
        
        self._layout.addWidget(QtWidgets.QLabel("Detections for this image across all runs:"))
        
        self._tree = self._create_tree(["Run ID", "Date", "Model", "Detections", "Max Score"])
        self._tree.itemDoubleClicked.connect(self.accept)
        self._layout.addWidget(self._tree)
        
        self._layout.addWidget(self._tree)
        
        # Simplify: Single click on Run ID -> Show Run Only
        # But we also need "Show All".
        # Let's add top-level item "All" to the list instead of a separate button?
        # Or just a "Show All" button is fine, but make list items single-click.
        
        self._show_all_btn = QtWidgets.QPushButton("Show All (No Filter)")
        self._show_all_btn.clicked.connect(self._on_show_all)
        self._layout.addWidget(self._show_all_btn)
        
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._populate()

    def _populate(self) -> None:
        if not self._data_csv.exists():
            return
            
        # Group by run_id
        by_run = {}
        try:
            with self._data_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("run_id") or "unknown"
                    by_run.setdefault(rid, []).append(row)
        except Exception:
            return
            
        for rid, rows in sorted(by_run.items(), reverse=True):
            count = len(rows)
            max_score = 0.0
            model = "Unknown"
            date = "Unknown"
            
            for r in rows:
                if model == "Unknown":
                    model = r.get("model") or "Unknown"
                if date == "Unknown":
                    date = r.get("created_at") or "Unknown"
                try:
                    s = float(r.get("score") or 0)
                    if s > max_score:
                        max_score = s
                except:
                    pass
            
            item = QtWidgets.QTreeWidgetItem(self._tree)
            item.setText(0, rid)
            item.setText(1, date)
            item.setText(2, model)
            item.setText(3, str(count))
            item.setText(4, f"{max_score:.2f}")
            item.setData(0, QtCore.Qt.UserRole, rid)

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, col: int) -> None:
        self._selected_run_id = item.data(0, QtCore.Qt.UserRole)
        self.accept()

    def _on_show_all(self) -> None:
        self._selected_run_id = "ALL"
        self.accept()

    def get_result(self) -> str | None:
        return self._selected_run_id
