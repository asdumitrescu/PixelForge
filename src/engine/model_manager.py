"""Model loading and management via Spandrel."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from src.engine.device import DeviceManager

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Info about a model file found on disk."""

    name: str
    path: Path
    file_size_mb: float
    format: str  # "pth", "safetensors", etc.


class ModelManager:
    """Loads, caches, and manages super-resolution models via Spandrel."""

    SUPPORTED_EXTENSIONS = (".pth", ".safetensors", ".ckpt", ".pt")

    def __init__(self, models_dir: Path, device_manager: DeviceManager) -> None:
        self._models_dir = models_dir
        self._device_manager = device_manager
        self._current_model: object | None = None  # Spandrel ImageModelDescriptor
        self._current_model_path: Path | None = None
        self._use_half: bool = False

    @property
    def current_model(self) -> object | None:
        """The currently loaded Spandrel model descriptor, or None."""
        return self._current_model

    @property
    def current_model_path(self) -> Path | None:
        return self._current_model_path

    @property
    def model_scale(self) -> int:
        """Scale factor of the currently loaded model."""
        if self._current_model is not None:
            return self._current_model.scale  # type: ignore[union-attr]
        return 4  # default

    def load_model(self, model_path: Path, use_half: bool = True) -> None:
        """Load a model from a .pth/.safetensors file using Spandrel.

        Args:
            model_path: Path to the model weights file.
            use_half: Use fp16 for reduced VRAM (no speed benefit on Pascal GPUs).
        """
        import spandrel

        # Unload previous model first
        if self._current_model is not None:
            self.unload_model()

        logger.info("Loading model: %s", model_path.name)

        # Spandrel auto-detects architecture from the weight file
        model = spandrel.ModelLoader().load_from_file(str(model_path))

        device = self._device_manager.device
        model = model.to(device)
        model.eval()

        if use_half and device.type == "cuda" and self._device_manager.supports_half():
            model = model.half()
            self._use_half = True
            logger.info("Model loaded in fp16 (half precision)")
        else:
            self._use_half = False
            logger.info("Model loaded in fp32 (full precision)")

        self._current_model = model
        self._current_model_path = model_path
        logger.info(
            "Model ready: %s | scale=%dx | device=%s",
            model_path.name,
            model.scale,
            device,
        )

    def unload_model(self) -> None:
        """Unload the current model and free GPU memory."""
        if self._current_model is not None:
            del self._current_model
            self._current_model = None
            self._current_model_path = None
            self._device_manager.clear_cache()
            logger.info("Model unloaded, GPU cache cleared")

    def get_available_models(self) -> list[ModelInfo]:
        """Scan the models directory for compatible weight files."""
        if not self._models_dir.exists():
            return []

        models: list[ModelInfo] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for path in self._models_dir.glob(f"*{ext}"):
                size_mb = path.stat().st_size / (1024 * 1024)
                models.append(
                    ModelInfo(
                        name=path.stem,
                        path=path,
                        file_size_mb=round(size_mb, 1),
                        format=ext.lstrip("."),
                    )
                )
        return sorted(models, key=lambda m: m.name)

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._current_model is not None

    def move_to_device(self, device: torch.device) -> None:
        """Move the current model to a different device (e.g., CPU fallback)."""
        if self._current_model is not None:
            self._current_model = self._current_model.to(device)  # type: ignore[union-attr]
            logger.info("Model moved to %s", device)
