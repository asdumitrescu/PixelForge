"""Background worker for image upscaling using Python threading (not QThread).

Why not QThread:
  QThread::~QThread() calls abort() if the thread is still running. Python's
  cyclic GC can trigger this destructor mid-CUDA-inference, crashing the app.
  Python threading.Thread has no such check — safe to GC at any time.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal

from src.engine.upscaler import Upscaler

logger = logging.getLogger(__name__)


class UpscaleWorker(QObject):
    """Runs image upscaling in a daemon thread while emitting Qt signals.

    Public API is intentionally compatible with the previous QThread version
    (start / cancel / isRunning / wait) so call-sites need no changes.

    Signals:
        progress(int, int, float): (current_tile, total_tiles, eta_seconds)
        finished(object):          The upscaled np.ndarray.
        error(str):                Error message on failure or cancellation.
        stage(str):                Human-readable pipeline phase name.
    """

    progress = Signal(int, int, float)
    finished = Signal(object)
    error = Signal(str)
    stage = Signal(str)

    def __init__(
        self,
        upscaler: Upscaler,
        image: np.ndarray,
        face_model_path: Path | None = None,
        device: torch.device | None = None,
        denoise: bool = False,
    ) -> None:
        super().__init__()
        self._upscaler = upscaler
        self._image = image
        self._face_model_path = face_model_path
        self._device = device or torch.device("cpu")
        self._denoise = denoise
        self._thread: threading.Thread | None = None

    # --- QThread-compatible API ---

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="UpscaleWorker")
        self._thread.start()

    def cancel(self) -> None:
        self._upscaler.cancel()

    def isRunning(self) -> bool:  # noqa: N802 — match Qt naming
        return self._thread is not None and self._thread.is_alive()

    def wait(self, msecs: int | None = None) -> None:
        if self._thread is not None:
            timeout = msecs / 1000.0 if msecs is not None else None
            self._thread.join(timeout=timeout)

    # --- Internal ---

    def _run(self) -> None:
        try:
            image = self._image

            if self._denoise:
                self.stage.emit("Denoising JPEG artifacts...")
                from src.engine.preprocessor import denoise_jpeg
                image = denoise_jpeg(image)

            self.stage.emit("Upscaling...")
            result = self._upscaler.upscale(
                image,
                progress_callback=self._on_progress,
            )

            if self._face_model_path is not None:
                result = self._run_face_enhancement(result)

            self.finished.emit(result)
        except InterruptedError:
            logger.info("Upscaling cancelled by user")
            self.error.emit("Cancelled")
        except Exception as e:
            logger.error("Upscaling failed: %s\n%s", e, traceback.format_exc())
            self.error.emit(str(e))

    def _run_face_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Load GFPGAN and enhance faces in the upscaled image."""
        self.stage.emit("Enhancing faces...")

        from src.engine.face_enhancer import FaceEnhancer

        enhancer = FaceEnhancer(self._face_model_path, self._device)
        return enhancer.enhance(image)

    def _on_progress(self, current: int, total: int, eta: float) -> None:
        self.progress.emit(current, total, eta)
