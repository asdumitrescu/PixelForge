"""Smoke tests for application constants."""

from src.constants import (
    APP_NAME,
    DEFAULT_TILE_PAD,
    DEFAULT_TILE_SIZE,
    MIN_TILE_SIZE,
    MODELS_DIR,
    SUPPORTED_INPUT_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
    TILE_REDUCE_FACTOR,
)


def test_app_name() -> None:
    assert APP_NAME == "PixelForge"


def test_supported_input_formats_not_empty() -> None:
    assert len(SUPPORTED_INPUT_FORMATS) > 0
    assert ".png" in SUPPORTED_INPUT_FORMATS
    assert ".jpg" in SUPPORTED_INPUT_FORMATS


def test_supported_output_formats_not_empty() -> None:
    assert len(SUPPORTED_OUTPUT_FORMATS) > 0
    assert "PNG" in SUPPORTED_OUTPUT_FORMATS


def test_default_tile_size_in_range() -> None:
    assert 128 <= DEFAULT_TILE_SIZE <= 512


def test_tile_pad_positive() -> None:
    assert DEFAULT_TILE_PAD > 0


def test_min_tile_size_less_than_default() -> None:
    assert MIN_TILE_SIZE < DEFAULT_TILE_SIZE


def test_tile_reduce_factor_between_0_and_1() -> None:
    assert 0 < TILE_REDUCE_FACTOR < 1


def test_models_dir_path_valid() -> None:
    assert MODELS_DIR.name == "models"
