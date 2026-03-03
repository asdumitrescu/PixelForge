"""Face enhancement via GFPGAN — optional post-processing after super-resolution."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.engine.model_registry import ModelEntry

logger = logging.getLogger(__name__)

# GFPGAN v1.4 model entry — reuses ModelEntry so existing download
# infrastructure (ModelDownloader, DownloadDialog) works without changes.
GFPGAN_MODEL = ModelEntry(
    id="gfpgan-v1.4",
    display_name="GFPGAN v1.4 (Face Enhancement)",
    filename="GFPGANv1.4.pth",
    url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    scale=1,
    description="Face restoration and enhancement. Applied after upscaling.",
    file_size_mb=349.0,
)


class FaceEnhancer:
    """Detects and restores faces in an image using GFPGAN.

    Intended to run *after* super-resolution — the upscaled image has better
    resolution but faces may still show artifacts. GFPGAN sharpens facial
    features and removes degradation.
    """

    def __init__(self, model_path: Path, device: torch.device) -> None:
        try:
            from gfpgan import GFPGANer
        except ImportError:
            raise RuntimeError(
                "GFPGAN is not installed. Run: pip install gfpgan"
            ) from None

        device_str = str(device)
        self._restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,  # We already upscaled — just restore faces at current resolution
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device_str,
        )
        logger.info("FaceEnhancer loaded from %s on %s", model_path.name, device_str)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Detect and enhance all faces in the image.

        Args:
            image: HWC uint8 RGB numpy array.

        Returns:
            HWC uint8 RGB numpy array with enhanced faces composited back.
            If no faces are detected, returns the original image unchanged.
        """
        # GFPGANer expects BGR input
        bgr = image[:, :, ::-1].copy()

        _, _, restored_bgr = self._restorer.enhance(
            bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5,  # 0 = original, 1 = fully restored
        )

        if restored_bgr is None:
            logger.info("No faces detected — returning original image")
            return image

        # BGR → RGB
        return restored_bgr[:, :, ::-1].copy()
