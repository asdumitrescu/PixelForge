"""Tests for tile grid calculation, extraction, and stitching."""

import numpy as np
import pytest

from src.engine.tiler import Tiler


@pytest.fixture
def tiler() -> Tiler:
    return Tiler(tile_size=64, tile_pad=8)


def make_image(h: int, w: int, channels: int = 3) -> np.ndarray:
    """Create a test image with unique pixel values for verification."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(h, w, channels), dtype=np.uint8)


def make_gradient(h: int, w: int) -> np.ndarray:
    """Create a gradient image for visual seam detection."""
    y = np.linspace(0, 255, h, dtype=np.uint8)
    x = np.linspace(0, 255, w, dtype=np.uint8)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([yy, xx, (yy + xx) // 2], axis=2).astype(np.uint8)


class TestTileCalculation:
    def test_single_tile_small_image(self, tiler: Tiler) -> None:
        """Image smaller than tile_size should produce 1 tile."""
        tiles = tiler.calculate_tiles(32, 32)
        assert len(tiles) == 1

    def test_exact_fit(self, tiler: Tiler) -> None:
        """Image exactly 2x2 tiles should produce 4 tiles."""
        tiles = tiler.calculate_tiles(128, 128)
        assert len(tiles) == 4

    def test_non_exact_fit(self, tiler: Tiler) -> None:
        """Image not divisible by tile_size should still cover all pixels."""
        tiles = tiler.calculate_tiles(100, 100)
        # ceil(100/64) = 2 in each dim = 4 tiles
        assert len(tiles) == 4

    def test_tile_grid_covers_full_image(self, tiler: Tiler) -> None:
        """Every pixel should be covered by at least one tile."""
        h, w = 200, 300
        tiles = tiler.calculate_tiles(h, w)

        covered = np.zeros((h, w), dtype=bool)
        for t in tiles:
            covered[t.y_start : t.y_end, t.x_start : t.x_end] = True

        assert covered.all(), "Some pixels are not covered by any tile"

    def test_tile_indices_sequential(self, tiler: Tiler) -> None:
        tiles = tiler.calculate_tiles(200, 200)
        indices = [t.index for t in tiles]
        assert indices == list(range(len(tiles)))

    def test_total_tiles_matches(self, tiler: Tiler) -> None:
        h, w = 200, 300
        total = tiler.total_tiles(h, w)
        tiles = tiler.calculate_tiles(h, w)
        assert total == len(tiles)


class TestTileExtraction:
    def test_extract_tile_shape(self, tiler: Tiler) -> None:
        img = make_image(200, 200)
        tiles = tiler.calculate_tiles(200, 200)

        # Middle tile should include full padding
        middle_tile = tiles[0]  # top-left has clamped padding
        extracted = tiler.extract_tile(img, middle_tile)
        assert extracted.ndim == 3
        assert extracted.shape[2] == 3

    def test_extract_with_padding(self) -> None:
        """A tile in the middle of the image should include full padding."""
        tiler = Tiler(tile_size=64, tile_pad=16)
        img = make_image(256, 256)
        tiles = tiler.calculate_tiles(256, 256)

        # Pick a middle tile (not edge)
        middle = [t for t in tiles if t.pad_top == 16 and t.pad_left == 16]
        assert len(middle) > 0, "No fully-padded middle tiles found"

        tile = middle[0]
        extracted = tiler.extract_tile(img, tile)
        expected_h = (tile.y_end - tile.y_start) + tile.pad_top + tile.pad_bottom
        expected_w = (tile.x_end - tile.x_start) + tile.pad_left + tile.pad_right
        assert extracted.shape[0] == expected_h
        assert extracted.shape[1] == expected_w

    def test_edge_tile_has_clamped_padding(self, tiler: Tiler) -> None:
        """Top-left tile should have 0 top and left padding."""
        tiles = tiler.calculate_tiles(200, 200)
        top_left = tiles[0]
        assert top_left.pad_top == 0
        assert top_left.pad_left == 0


class TestTileStitching:
    def test_round_trip_identity(self) -> None:
        """Extract all tiles and place back without processing. Result should match original."""
        tiler = Tiler(tile_size=64, tile_pad=16)
        img = make_image(200, 300)
        h, w = img.shape[:2]
        scale = 1  # No upscaling, just identity

        canvas = np.zeros_like(img)
        tiles = tiler.calculate_tiles(h, w)

        for tile_info in tiles:
            extracted = tiler.extract_tile(img, tile_info)
            tiler.place_tile(canvas, extracted, tile_info, scale)

        np.testing.assert_array_equal(canvas, img)

    def test_round_trip_with_scale(self) -> None:
        """Simulate 2x upscale: expand each tile and verify canvas dimensions."""
        tiler = Tiler(tile_size=32, tile_pad=8)
        img = make_image(64, 96)
        h, w = img.shape[:2]
        scale = 2

        canvas = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
        tiles = tiler.calculate_tiles(h, w)

        for tile_info in tiles:
            extracted = tiler.extract_tile(img, tile_info)
            # Simulate upscale by repeating pixels
            upscaled = np.repeat(np.repeat(extracted, scale, axis=0), scale, axis=1)
            tiler.place_tile(canvas, upscaled, tile_info, scale)

        assert canvas.shape == (h * scale, w * scale, 3)
        # Verify non-zero (canvas was filled)
        assert canvas.sum() > 0

    @pytest.mark.parametrize("tile_size", [32, 64, 128, 192, 256])
    def test_different_tile_sizes(self, tile_size: int) -> None:
        """Various tile sizes should all produce seamless round-trip."""
        tiler = Tiler(tile_size=tile_size, tile_pad=16)
        img = make_image(300, 400)
        h, w = img.shape[:2]

        canvas = np.zeros_like(img)
        tiles = tiler.calculate_tiles(h, w)

        for tile_info in tiles:
            extracted = tiler.extract_tile(img, tile_info)
            tiler.place_tile(canvas, extracted, tile_info, 1)

        np.testing.assert_array_equal(canvas, img)
