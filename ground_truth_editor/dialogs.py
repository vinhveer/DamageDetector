from PySide6 import QtCore, QtGui, QtWidgets

class ProcessingDialog(QtWidgets.QDialog):
    stopRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None, title: str) -> None:
        super().__init__(parent)
        self._allow_close = False
        self.setWindowTitle(title)
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self.resize(720, 420)

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(title, self)
        header.setWordWrap(True)
        layout.addWidget(header)

        self._log = QtWidgets.QPlainTextEdit(self)
        self._log.setReadOnly(True)
        layout.addWidget(self._log, 1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self._stop_btn = QtWidgets.QPushButton("Stop", self)
        self._stop_btn.clicked.connect(self.stopRequested.emit)
        row.addWidget(self._stop_btn)
        layout.addLayout(row)

    def log_widget(self) -> QtWidgets.QPlainTextEdit:
        return self._log

    def stop_button(self) -> QtWidgets.QPushButton:
        return self._stop_btn

    def allow_close(self) -> None:
        self._allow_close = True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
        else:
            event.ignore()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape and not self._allow_close:
            event.ignore()
            return
        super().keyPressEvent(event)
