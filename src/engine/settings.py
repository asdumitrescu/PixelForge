"""Persistent application settings using QSettings."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QSettings

from src.constants import DEFAULT_TILE_PAD, DEFAULT_TILE_SIZE


class AppSettings:
    """Persistent user preferences stored via QSettings (platform-native storage).

    On Linux: ~/.config/PixelForge/PixelForge.conf
    """

    def __init__(self) -> None:
        self._settings = QSettings("PixelForge", "PixelForge")

    # --- Inference ---

    @property
    def tile_size(self) -> int:
        return int(self._settings.value("inference/tile_size", DEFAULT_TILE_SIZE))

    @tile_size.setter
    def tile_size(self, value: int) -> None:
        self._settings.setValue("inference/tile_size", value)

    @property
    def tile_pad(self) -> int:
        return int(self._settings.value("inference/tile_pad", DEFAULT_TILE_PAD))

    @tile_pad.setter
    def tile_pad(self, value: int) -> None:
        self._settings.setValue("inference/tile_pad", value)

    @property
    def use_half(self) -> bool:
        return self._settings.value("inference/use_half", True, type=bool)

    @use_half.setter
    def use_half(self, value: bool) -> None:
        self._settings.setValue("inference/use_half", value)

    @property
    def default_model(self) -> str:
        return str(self._settings.value("inference/default_model", "realesrgan-x4plus"))

    @default_model.setter
    def default_model(self, value: str) -> None:
        self._settings.setValue("inference/default_model", value)

    # --- Output ---

    @property
    def output_format(self) -> str:
        return str(self._settings.value("output/format", "PNG"))

    @output_format.setter
    def output_format(self, value: str) -> None:
        self._settings.setValue("output/format", value)

    @property
    def jpeg_quality(self) -> int:
        return int(self._settings.value("output/jpeg_quality", 95))

    @jpeg_quality.setter
    def jpeg_quality(self, value: int) -> None:
        self._settings.setValue("output/jpeg_quality", value)

    @property
    def webp_quality(self) -> int:
        return int(self._settings.value("output/webp_quality", 90))

    @webp_quality.setter
    def webp_quality(self, value: int) -> None:
        self._settings.setValue("output/webp_quality", value)

    # --- Paths ---

    @property
    def last_input_dir(self) -> str:
        return str(self._settings.value("paths/last_input_dir", ""))

    @last_input_dir.setter
    def last_input_dir(self, value: str) -> None:
        self._settings.setValue("paths/last_input_dir", value)

    @property
    def last_output_dir(self) -> str:
        return str(self._settings.value("paths/last_output_dir", ""))

    @last_output_dir.setter
    def last_output_dir(self, value: str) -> None:
        self._settings.setValue("paths/last_output_dir", value)

    # --- Window ---

    @property
    def window_geometry(self) -> QByteArray:
        return self._settings.value("window/geometry", QByteArray())

    @window_geometry.setter
    def window_geometry(self, value: QByteArray) -> None:
        self._settings.setValue("window/geometry", value)

    @property
    def window_state(self) -> QByteArray:
        return self._settings.value("window/state", QByteArray())

    @window_state.setter
    def window_state(self, value: QByteArray) -> None:
        self._settings.setValue("window/state", value)
