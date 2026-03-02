"""Image I/O and tensor conversion utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path: Path) -> tuple[np.ndarray, dict]:
    """Load an image file and return (HWC uint8 array, metadata dict).

    Metadata includes ICC profile, original format, and size info.
    """
    img = Image.open(path)
    metadata: dict = {
        "format": img.format,
        "mode": img.mode,
        "size": img.size,  # (width, height)
    }

    # Preserve ICC profile if present
    icc = img.info.get("icc_profile")
    if icc:
        metadata["icc_profile"] = icc

    # Convert to RGB or RGBA
    if img.mode == "RGBA":
        arr = np.array(img)  # Keep RGBA
    elif img.mode in ("L", "LA", "P", "PA"):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")
        arr = np.array(img)
    else:
        img = img.convert("RGB")
        arr = np.array(img)

    return arr, metadata


def save_image(
    image: np.ndarray,
    path: Path,
    fmt: str = "PNG",
    quality: int = 95,
    metadata: dict | None = None,
) -> None:
    """Save a HWC uint8 numpy array as an image file.

    Args:
        image: HWC uint8 numpy array (RGB or RGBA).
        path: Output file path.
        fmt: Image format ("PNG", "JPEG", "WebP").
        quality: JPEG/WebP quality (1-100).
        metadata: Optional metadata dict (icc_profile, etc.).
    """
    if image.shape[2] == 4:
        img = Image.fromarray(image, mode="RGBA")
    else:
        img = Image.fromarray(image, mode="RGB")

    save_kwargs: dict = {}

    if metadata and "icc_profile" in metadata:
        save_kwargs["icc_profile"] = metadata["icc_profile"]

    if fmt.upper() == "JPEG":
        # JPEG doesn't support alpha — drop it
        if img.mode == "RGBA":
            img = img.convert("RGB")
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif fmt.upper() == "WEBP":
        save_kwargs["quality"] = quality
    elif fmt.upper() == "PNG":
        save_kwargs["compress_level"] = 6  # balanced speed vs size

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=fmt.upper(), **save_kwargs)


def image_to_tensor(
    image: np.ndarray,
    device: torch.device,
    half: bool = False,
) -> torch.Tensor:
    """Convert HWC uint8 numpy array to BCHW float32 tensor on device.

    Args:
        image: HWC uint8 numpy array (RGB, 3 channels).
        device: Target torch device.
        half: Use fp16 precision.

    Returns:
        BCHW float32 (or float16) tensor in [0, 1] range.
    """
    # HWC -> CHW -> BCHW
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0)  # Add batch dim

    if half:
        tensor = tensor.half()

    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert BCHW float tensor to HWC uint8 numpy array.

    Args:
        tensor: BCHW float tensor in [0, 1] range.

    Returns:
        HWC uint8 numpy array.
    """
    # BCHW -> CHW -> HWC
    output = tensor.squeeze(0).float().clamp(0, 1)
    output = (output * 255.0).round().byte()
    output = output.cpu().numpy().transpose(1, 2, 0)
    return output


def split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Separate RGB from alpha channel if RGBA.

    Returns:
        (rgb_array, alpha_array) where alpha is None if input is RGB.
    """
    if image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]
    return image, None


def merge_alpha(rgb: np.ndarray, alpha: np.ndarray | None, scale: int) -> np.ndarray:
    """Merge RGB with upscaled alpha channel.

    Alpha is upscaled via bicubic interpolation to match the RGB dimensions.

    Args:
        rgb: HWC uint8 RGB array (upscaled).
        alpha: HW uint8 alpha channel (original resolution), or None.
        scale: The scale factor that was applied to RGB.

    Returns:
        RGBA or RGB array depending on whether alpha was provided.
    """
    if alpha is None:
        return rgb

    # Upscale alpha via bicubic interpolation
    h, w = alpha.shape
    new_h, new_w = h * scale, w * scale
    alpha_img = Image.fromarray(alpha, mode="L")
    alpha_upscaled = alpha_img.resize((new_w, new_h), Image.BICUBIC)
    alpha_arr = np.array(alpha_upscaled)

    # Merge: RGB + Alpha -> RGBA
    rgba = np.concatenate([rgb, alpha_arr[:, :, np.newaxis]], axis=2)
    return rgba
