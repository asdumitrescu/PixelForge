"""Drag-and-drop zone for image file input."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QLabel, QWidget

from src.constants import INPUT_FILE_FILTER, SUPPORTED_INPUT_FORMATS
from src.gui.styles import DROP_ZONE, DROP_ZONE_ACTIVE


class DropZone(QLabel):
    """A drag-and-drop area that accepts image files."""

    file_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & drop an image here\nor click to browse")
        self.setStyleSheet(DROP_ZONE)
        self.setMinimumSize(400, 300)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self._is_valid_image(urls[0].toLocalFile()):
                event.acceptProposedAction()
                self.setStyleSheet(DROP_ZONE_ACTIVE)
                return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet(DROP_ZONE)

    def dropEvent(self, event) -> None:
        self.setStyleSheet(DROP_ZONE)
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self._is_valid_image(file_path):
                self.file_dropped.emit(file_path)
                event.acceptProposedAction()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", INPUT_FILE_FILTER,
            )
            if file_path:
                self.file_dropped.emit(file_path)

    @staticmethod
    def _is_valid_image(file_path: str) -> bool:
        return Path(file_path).suffix.lower() in SUPPORTED_INPUT_FORMATS
