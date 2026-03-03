"""Image pre-processing — JPEG artifact removal before upscaling."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def denoise_jpeg(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """Remove JPEG compression block artifacts using non-local means denoising.

    Uses OpenCV's fastNlMeansDenoisingColored which is specifically effective
    at removing the 8x8 block artifacts from JPEG compression while preserving
    edges and detail.

    Args:
        image: HWC uint8 RGB numpy array.
        strength: Filter strength (higher = more denoising, 5-15 typical).
                  10 is a good default for moderately compressed JPEG.

    Returns:
        Denoised HWC uint8 RGB numpy array.
    """
    h, w = image.shape[:2]
    logger.info("Denoising JPEG artifacts: %dx%d (strength=%d)", w, h, strength)

    # fastNlMeansDenoisingColored works on BGR — convert
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Positional args: src, dst, h, hColor, templateWindowSize, searchWindowSize
    # (keyword names changed across OpenCV versions — use positional for safety)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        bgr, None, strength, strength, 7, 21
    )
    result = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

    logger.info("Denoising complete")
    return result
