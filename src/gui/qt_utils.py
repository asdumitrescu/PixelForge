"""Qt-to-numpy conversion helpers and formatting utilities."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def numpy_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Convert a HWC uint8 numpy array to QPixmap.

    Supports RGB (3 channels) and RGBA (4 channels).
    """
    h, w = image.shape[:2]

    if image.shape[2] == 4:
        # RGBA
        bytes_per_line = 4 * w
        qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
    else:
        # RGB
        bytes_per_line = 3 * w
        qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    return QPixmap.fromImage(qimage.copy())  # .copy() to detach from numpy buffer


def qpixmap_to_numpy(pixmap: QPixmap) -> np.ndarray:
    """Convert a QPixmap to a HWC uint8 numpy array (RGB)."""
    qimage = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w = qimage.width()
    h = qimage.height()
    ptr = qimage.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3)
    return arr.copy()


def format_file_size(size_bytes: int) -> str:
    """Format bytes into human-readable string (e.g., '12.5 MB')."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    if seconds < 0:
        return "calculating..."
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def format_dimensions(width: int, height: int) -> str:
    """Format image dimensions (e.g., '1920 x 1080')."""
    return f"{width} x {height}"


def format_scale_label(input_w: int, input_h: int, scale: int) -> str:
    """Format a scale description (e.g., '960x540 -> 3840x2160 (4K)')."""
    out_w = input_w * scale
    out_h = input_h * scale
    label = f"{input_w}x{input_h} -> {out_w}x{out_h}"

    # Add common resolution names
    if out_w >= 3840 and out_h >= 2160:
        label += " (4K+)"
    elif out_w >= 2560 and out_h >= 1440:
        label += " (1440p+)"
    elif out_w >= 1920 and out_h >= 1080:
        label += " (1080p+)"

    return label
