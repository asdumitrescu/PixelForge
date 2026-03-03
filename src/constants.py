"""Application-wide constants and defaults."""

from pathlib import Path

APP_NAME = "PixelForge"
APP_VERSION = "0.1.0"

# Inference defaults
DEFAULT_TILE_SIZE = 128  # Smaller default — leaves VRAM for display compositor
DEFAULT_TILE_PAD = 32
MIN_TILE_SIZE = 128
TILE_REDUCE_FACTOR = 0.75  # Multiply tile_size by this on OOM retry

# Supported image formats
SUPPORTED_INPUT_FORMATS = (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp")
SUPPORTED_OUTPUT_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WebP": ".webp"}

# File filter strings for Qt file dialogs
INPUT_FILE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp);;"
    "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;TIFF (*.tiff *.tif);;All Files (*)"
)
OUTPUT_FILE_FILTER = "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)"

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
