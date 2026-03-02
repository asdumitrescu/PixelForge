"""Core upscaling pipeline — orchestrates tiling, inference, and OOM recovery."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np
import torch

from src.constants import MIN_TILE_SIZE, TILE_REDUCE_FACTOR
from src.engine.device import DeviceManager
from src.engine.image_utils import (
    image_to_tensor,
    merge_alpha,
    split_alpha,
    tensor_to_image,
)
from src.engine.model_manager import ModelManager
from src.engine.tiler import Tiler

logger = logging.getLogger(__name__)

# Type alias for progress callback: (current_tile, total_tiles, eta_seconds)
ProgressCallback = Callable[[int, int, float], None]


class Upscaler:
    """Orchestrates the full image upscaling pipeline with tile-based inference.

    Handles:
    - Alpha channel separation and re-merge
    - Tile-based processing for VRAM safety
    - OOM recovery with automatic tile size reduction
    - CPU fallback as last resort
    - Cancellation support
    - Per-tile progress reporting with ETA
    """

    def __init__(
        self,
        model_manager: ModelManager,
        device_manager: DeviceManager,
        tile_size: int = 256,
        tile_pad: int = 32,
    ) -> None:
        self._model_manager = model_manager
        self._device_manager = device_manager
        self._tile_size = tile_size
        self._tile_pad = tile_pad
        self._cancelled = False

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @tile_size.setter
    def tile_size(self, value: int) -> None:
        self._tile_size = max(MIN_TILE_SIZE, value)

    def cancel(self) -> None:
        """Request cancellation of the current upscale operation."""
        self._cancelled = True

    def upscale(
        self,
        image: np.ndarray,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        """Upscale an image using the loaded model with tile-based processing.

        Args:
            image: HWC uint8 numpy array (RGB or RGBA).
            progress_callback: Optional (current_tile, total_tiles, eta_seconds) callback.

        Returns:
            Upscaled HWC uint8 numpy array.

        Raises:
            RuntimeError: If no model is loaded or processing fails completely.
            InterruptedError: If cancelled by user.
        """
        self._cancelled = False

        if not self._model_manager.is_loaded():
            raise RuntimeError("No model loaded. Load a model before upscaling.")

        model = self._model_manager.current_model
        scale = self._model_manager.model_scale
        use_half = self._model_manager._use_half

        # Separate alpha channel if RGBA
        rgb, alpha = split_alpha(image)
        h, w = rgb.shape[:2]

        logger.info(
            "Upscaling %dx%d → %dx%d (scale=%dx, tile=%dpx)",
            w, h, w * scale, h * scale, scale, self._tile_size,
        )

        # Create output canvas
        canvas = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

        # Process tiles
        tiler = Tiler(self._tile_size, self._tile_pad)
        tiles = tiler.calculate_tiles(h, w)
        total = len(tiles)
        tile_times: list[float] = []

        for tile_info in tiles:
            if self._cancelled:
                raise InterruptedError("Upscaling cancelled by user.")

            t_start = time.monotonic()

            # Extract tile with padding
            tile_data = tiler.extract_tile(rgb, tile_info)

            # Process tile (with OOM recovery)
            tile_output = self._process_tile(tile_data, model, use_half, scale)

            # Place result into canvas
            tiler.place_tile(canvas, tile_output, tile_info, scale)

            # Track timing for ETA
            elapsed = time.monotonic() - t_start
            tile_times.append(elapsed)
            avg_time = sum(tile_times) / len(tile_times)
            remaining = (total - tile_info.index - 1) * avg_time

            if progress_callback:
                progress_callback(tile_info.index + 1, total, remaining)

        # Merge alpha channel back if present
        result = merge_alpha(canvas, alpha, scale)

        logger.info("Upscaling complete: %dx%d", result.shape[1], result.shape[0])
        return result

    def _process_tile(
        self,
        tile_data: np.ndarray,
        model: object,
        use_half: bool,
        scale: int,
    ) -> np.ndarray:
        """Process a single tile through the model with OOM recovery.

        On OOM: reduces tile size, clears cache, retries. Falls back to CPU as last resort.
        """
        device = self._device_manager.device
        tensor = image_to_tensor(tile_data, device, half=use_half)

        try:
            with torch.no_grad():
                output_tensor = model(tensor)  # type: ignore[operator]
        except torch.cuda.OutOfMemoryError:
            return self._handle_oom_and_retry(tile_data, model, use_half, scale)

        result = tensor_to_image(output_tensor)

        # Clean up tensor references
        del tensor, output_tensor
        return result

    def _handle_oom_and_retry(
        self,
        tile_data: np.ndarray,
        model: object,
        use_half: bool,
        scale: int,
    ) -> np.ndarray:
        """Handle OOM by reducing tile size or falling back to CPU.

        Strategy:
        1. Clear VRAM cache
        2. Reduce tile size by TILE_REDUCE_FACTOR (25% smaller)
        3. If tile_size < MIN_TILE_SIZE, fall back to CPU
        4. Retry inference
        """
        self._device_manager.clear_cache()
        new_size = int(self._tile_size * TILE_REDUCE_FACTOR)

        if new_size >= MIN_TILE_SIZE:
            logger.warning(
                "OOM! Reducing tile size: %d → %d", self._tile_size, new_size
            )
            self._tile_size = new_size
            # Retry on GPU with smaller conceptual tile size
            # But for this specific tile_data (already extracted), process on GPU
            device = self._device_manager.device
            tensor = image_to_tensor(tile_data, device, half=use_half)
            try:
                with torch.no_grad():
                    output_tensor = model(tensor)  # type: ignore[operator]
                result = tensor_to_image(output_tensor)
                del tensor, output_tensor
                return result
            except torch.cuda.OutOfMemoryError:
                # Still OOM — fall through to CPU
                self._device_manager.clear_cache()

        # CPU fallback
        logger.warning("GPU exhausted. Falling back to CPU for this tile.")
        cpu_device = torch.device("cpu")
        self._model_manager.move_to_device(cpu_device)

        tensor = image_to_tensor(tile_data, cpu_device, half=False)
        with torch.no_grad():
            output_tensor = model(tensor)  # type: ignore[operator]
        result = tensor_to_image(output_tensor)
        del tensor, output_tensor

        # Move model back to GPU for next tiles
        if self._device_manager.device.type == "cuda":
            self._model_manager.move_to_device(self._device_manager.device)

        return result
