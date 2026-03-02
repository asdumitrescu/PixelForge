"""Background worker thread for image upscaling."""

from __future__ import annotations

import logging
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.engine.upscaler import Upscaler

logger = logging.getLogger(__name__)


class UpscaleWorker(QThread):
    """Runs image upscaling in a background thread to keep the GUI responsive.

    Signals:
        progress(int, int, float): (current_tile, total_tiles, eta_seconds)
        finished(np.ndarray): The upscaled image.
        error(str): Error message on failure.
        tile_size_reduced(int): New tile size when OOM recovery triggers.
        device_fallback(str): Device name when falling back (e.g., "cpu").
    """

    progress = Signal(int, int, float)
    finished = Signal(object)  # np.ndarray (Signal doesn't support numpy directly)
    error = Signal(str)
    tile_size_reduced = Signal(int)
    device_fallback = Signal(str)

    def __init__(self, upscaler: Upscaler, image: np.ndarray) -> None:
        super().__init__()
        self._upscaler = upscaler
        self._image = image

    def run(self) -> None:
        """Execute the upscaling pipeline in the background thread."""
        try:
            result = self._upscaler.upscale(
                self._image,
                progress_callback=self._on_progress,
            )
            self.finished.emit(result)
        except InterruptedError:
            logger.info("Upscaling cancelled by user")
            self.error.emit("Cancelled")
        except Exception as e:
            logger.error("Upscaling failed: %s\n%s", e, traceback.format_exc())
            self.error.emit(str(e))

    def cancel(self) -> None:
        """Request cancellation."""
        self._upscaler.cancel()

    def _on_progress(self, current: int, total: int, eta: float) -> None:
        """Forward progress from the upscaler to the GUI thread via signal."""
        self.progress.emit(current, total, eta)
