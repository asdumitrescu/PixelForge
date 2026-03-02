"""Background worker thread for model loading."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from src.engine.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ModelLoadWorker(QThread):
    """Loads a model in a background thread to avoid GUI freeze.

    Signals:
        finished(): Model loaded successfully.
        error(str): Error message on failure.
    """

    finished = Signal()
    error = Signal(str)

    def __init__(
        self, model_manager: ModelManager, model_path: Path, use_half: bool
    ) -> None:
        super().__init__()
        self._model_manager = model_manager
        self._model_path = model_path
        self._use_half = use_half

    def run(self) -> None:
        try:
            self._model_manager.load_model(self._model_path, self._use_half)
            self.finished.emit()
        except Exception as e:
            logger.error("Model load failed: %s\n%s", e, traceback.format_exc())
            self.error.emit(str(e))
