"""Background worker thread for model downloads."""

from __future__ import annotations

import logging
import traceback

from PySide6.QtCore import QThread, Signal

from src.engine.model_downloader import ModelDownloader
from src.engine.model_registry import ModelEntry

logger = logging.getLogger(__name__)


class DownloadWorker(QThread):
    """Downloads a model in a background thread with progress signals.

    Signals:
        progress(int, int): (bytes_downloaded, bytes_total)
        finished(str): File path of the downloaded model.
        error(str): Error message on failure.
    """

    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, downloader: ModelDownloader, entry: ModelEntry) -> None:
        super().__init__()
        self._downloader = downloader
        self._entry = entry

    def run(self) -> None:
        try:
            path = self._downloader.download(
                self._entry,
                progress_callback=self._on_progress,
            )
            self.finished.emit(str(path))
        except Exception as e:
            logger.error("Download failed: %s\n%s", e, traceback.format_exc())
            self.error.emit(str(e))

    def _on_progress(self, downloaded: int, total: int) -> None:
        self.progress.emit(downloaded, total)
