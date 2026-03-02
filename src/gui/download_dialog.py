"""Modal dialog showing model download progress."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.engine.model_downloader import ModelDownloader
from src.engine.model_registry import ModelEntry
from src.gui.qt_utils import format_file_size
from src.gui.styles import BUTTON_SECONDARY, PROGRESS_BAR
from src.workers.download_worker import DownloadWorker


class DownloadDialog(QDialog):
    """Modal dialog that downloads a model with a progress bar."""

    def __init__(
        self,
        downloader: ModelDownloader,
        entry: ModelEntry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._entry = entry
        self._worker: DownloadWorker | None = None
        self._download_path: str | None = None

        self.setWindowTitle("Downloading Model")
        self.setFixedSize(420, 180)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self._title = QLabel(f"Downloading: {entry.display_name}")
        self._title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self._title)

        self._size_label = QLabel(f"Size: ~{entry.file_size_mb:.0f} MB")
        self._size_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self._size_label)

        self._progress = QProgressBar()
        self._progress.setStyleSheet(PROGRESS_BAR)
        self._progress.setTextVisible(True)
        layout.addWidget(self._progress)

        self._status = QLabel("Starting download...")
        self._status.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._status)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(BUTTON_SECONDARY)
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # Start download
        self._worker = DownloadWorker(downloader, entry)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @property
    def download_path(self) -> str | None:
        return self._download_path

    def _on_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(downloaded)
            pct = int(downloaded / total * 100)
            self._progress.setFormat(f"{pct}%")
            self._status.setText(
                f"{format_file_size(downloaded)} / {format_file_size(total)}"
            )

    def _on_finished(self, path: str) -> None:
        self._download_path = path
        self.accept()

    def _on_error(self, msg: str) -> None:
        self._status.setText(f"Error: {msg}")
        self._status.setStyleSheet("color: #f38ba8; font-size: 11px;")
        self._cancel_btn.setText("Close")

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        self.reject()

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        super().closeEvent(event)
