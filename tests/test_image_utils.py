"""Tests for image I/O and tensor conversion utilities."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from PIL import Image

from src.engine.image_utils import (
    image_to_tensor,
    load_image,
    merge_alpha,
    save_image,
    split_alpha,
    tensor_to_image,
)


@pytest.fixture
def temp_dir():
    with TemporaryDirectory() as d:
        yield Path(d)


def make_test_image(w: int, h: int, mode: str = "RGB") -> Image.Image:
    """Create a solid color test image."""
    if mode == "RGBA":
        return Image.new("RGBA", (w, h), (100, 150, 200, 128))
    return Image.new("RGB", (w, h), (100, 150, 200))


class TestLoadImage:
    def test_load_png(self, temp_dir: Path) -> None:
        path = temp_dir / "test.png"
        make_test_image(64, 48).save(path, "PNG")

        arr, meta = load_image(path)
        assert arr.shape == (48, 64, 3)
        assert arr.dtype == np.uint8
        assert meta["format"] == "PNG"
        assert meta["size"] == (64, 48)

    def test_load_jpeg(self, temp_dir: Path) -> None:
        path = temp_dir / "test.jpg"
        make_test_image(64, 48).save(path, "JPEG")

        arr, meta = load_image(path)
        assert arr.shape == (48, 64, 3)
        assert meta["format"] == "JPEG"

    def test_load_rgba(self, temp_dir: Path) -> None:
        path = temp_dir / "test.png"
        make_test_image(64, 48, "RGBA").save(path, "PNG")

        arr, meta = load_image(path)
        assert arr.shape == (48, 64, 4)  # RGBA preserved

    def test_load_grayscale(self, temp_dir: Path) -> None:
        path = temp_dir / "test.png"
        Image.new("L", (32, 32), 128).save(path, "PNG")

        arr, meta = load_image(path)
        assert arr.shape == (32, 32, 3)  # Converted to RGB


class TestSaveImage:
    def test_save_png(self, temp_dir: Path) -> None:
        arr = np.full((48, 64, 3), 128, dtype=np.uint8)
        path = temp_dir / "out.png"
        save_image(arr, path, fmt="PNG")

        assert path.exists()
        reloaded = Image.open(path)
        assert reloaded.size == (64, 48)

    def test_save_jpeg(self, temp_dir: Path) -> None:
        arr = np.full((48, 64, 3), 128, dtype=np.uint8)
        path = temp_dir / "out.jpg"
        save_image(arr, path, fmt="JPEG", quality=90)

        assert path.exists()

    def test_save_webp(self, temp_dir: Path) -> None:
        arr = np.full((48, 64, 3), 128, dtype=np.uint8)
        path = temp_dir / "out.webp"
        save_image(arr, path, fmt="WebP", quality=90)

        assert path.exists()

    def test_save_rgba_as_png(self, temp_dir: Path) -> None:
        arr = np.full((48, 64, 4), 128, dtype=np.uint8)
        path = temp_dir / "out.png"
        save_image(arr, path, fmt="PNG")

        reloaded = Image.open(path)
        assert reloaded.mode == "RGBA"

    def test_save_rgba_as_jpeg_drops_alpha(self, temp_dir: Path) -> None:
        arr = np.full((48, 64, 4), 128, dtype=np.uint8)
        path = temp_dir / "out.jpg"
        save_image(arr, path, fmt="JPEG")

        reloaded = Image.open(path)
        assert reloaded.mode == "RGB"

    def test_save_creates_parent_dirs(self, temp_dir: Path) -> None:
        arr = np.full((32, 32, 3), 128, dtype=np.uint8)
        path = temp_dir / "sub" / "dir" / "out.png"
        save_image(arr, path)
        assert path.exists()


class TestTensorConversion:
    def test_image_to_tensor_shape(self) -> None:
        import torch

        arr = np.full((48, 64, 3), 128, dtype=np.uint8)
        tensor = image_to_tensor(arr, torch.device("cpu"))

        assert tensor.shape == (1, 3, 48, 64)  # BCHW
        assert tensor.dtype == torch.float32
        assert 0 <= tensor.min() <= tensor.max() <= 1.0

    def test_image_to_tensor_half(self) -> None:
        import torch

        arr = np.full((32, 32, 3), 200, dtype=np.uint8)
        tensor = image_to_tensor(arr, torch.device("cpu"), half=True)

        assert tensor.dtype == torch.float16

    def test_tensor_roundtrip(self) -> None:
        import torch

        original = np.array([[[100, 150, 200]]], dtype=np.uint8)  # 1x1 RGB
        tensor = image_to_tensor(original, torch.device("cpu"))
        recovered = tensor_to_image(tensor)

        # Allow +/- 1 due to float precision
        np.testing.assert_allclose(recovered, original, atol=1)

    def test_tensor_to_image_clamps(self) -> None:
        import torch

        # Values outside [0, 1] should be clamped
        tensor = torch.tensor([[[[1.5, -0.5]]]], dtype=torch.float32)  # B=1, C=1, H=1, W=2
        result = tensor_to_image(tensor)
        assert result.max() == 255
        assert result.min() == 0


class TestAlphaHandling:
    def test_split_rgb_returns_none_alpha(self) -> None:
        arr = np.full((32, 32, 3), 128, dtype=np.uint8)
        rgb, alpha = split_alpha(arr)
        assert rgb.shape == (32, 32, 3)
        assert alpha is None

    def test_split_rgba_separates(self) -> None:
        arr = np.full((32, 32, 4), 128, dtype=np.uint8)
        arr[:, :, 3] = 200  # Distinct alpha
        rgb, alpha = split_alpha(arr)

        assert rgb.shape == (32, 32, 3)
        assert alpha is not None
        assert alpha.shape == (32, 32)
        assert (alpha == 200).all()

    def test_merge_with_none_alpha_returns_rgb(self) -> None:
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = merge_alpha(rgb, None, 2)
        assert result.shape == (64, 64, 3)

    def test_merge_upscales_alpha(self) -> None:
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)  # Already "upscaled" 2x
        alpha = np.full((32, 32), 200, dtype=np.uint8)  # Original resolution

        result = merge_alpha(rgb, alpha, 2)
        assert result.shape == (64, 64, 4)
        assert result[:, :, 3].mean() == pytest.approx(200, abs=1)
