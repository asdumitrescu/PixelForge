"""Static registry of known super-resolution models with download URLs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    """Metadata for a known super-resolution model."""

    id: str
    display_name: str
    filename: str
    url: str
    scale: int
    description: str
    file_size_mb: float
    sha256: str | None = None


# Verified model download URLs from official GitHub releases
MODEL_REGISTRY: dict[str, ModelEntry] = {
    "realesrgan-x4plus": ModelEntry(
        id="realesrgan-x4plus",
        display_name="Real-ESRGAN x4 Plus",
        filename="RealESRGAN_x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        scale=4,
        description="Best for real-world photographs. Handles compression artifacts, noise, blur.",
        file_size_mb=67.0,
        sha256=None,
    ),
    "realesrgan-x4plus-anime": ModelEntry(
        id="realesrgan-x4plus-anime",
        display_name="Real-ESRGAN x4 Anime",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        scale=4,
        description="Optimized for anime, manga, and digital illustration. Smaller model.",
        file_size_mb=17.0,
        sha256=None,
    ),
    "realesrgan-x2plus": ModelEntry(
        id="realesrgan-x2plus",
        display_name="Real-ESRGAN x2 Plus",
        filename="RealESRGAN_x2plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        scale=2,
        description="2x upscale for photographs. Good when 4x is too much.",
        file_size_mb=67.0,
        sha256=None,
    ),
}


def get_model_entry(model_id: str) -> ModelEntry | None:
    """Look up a model by its ID."""
    return MODEL_REGISTRY.get(model_id)


def get_models_for_scale(scale: int) -> list[ModelEntry]:
    """Return all models that support a given scale factor."""
    return [m for m in MODEL_REGISTRY.values() if m.scale == scale]


def get_all_model_ids() -> list[str]:
    """Return all registered model IDs."""
    return list(MODEL_REGISTRY.keys())
