"""Tile-based image processing for memory-safe super-resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TileInfo:
    """Position and padding info for a single tile."""

    row: int
    col: int
    # Input image coordinates (before padding)
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    # Actual padding applied (may be less at image borders)
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int
    # Index for progress tracking
    index: int


class Tiler:
    """Splits images into overlapping tiles and stitches results back together.

    The overlap (tile_pad) ensures seamless boundaries between tiles.
    After model inference, the padded regions are cropped from each tile's output,
    and only the core region is placed into the output canvas.
    """

    def __init__(self, tile_size: int = 256, tile_pad: int = 32) -> None:
        self.tile_size = tile_size
        self.tile_pad = tile_pad

    def calculate_tiles(self, height: int, width: int) -> list[TileInfo]:
        """Calculate the tile grid for an image of given dimensions.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            List of TileInfo describing each tile's position and padding.
        """
        rows = math.ceil(height / self.tile_size)
        cols = math.ceil(width / self.tile_size)
        tiles: list[TileInfo] = []
        index = 0

        for r in range(rows):
            for c in range(cols):
                y_start = r * self.tile_size
                x_start = c * self.tile_size
                y_end = min(y_start + self.tile_size, height)
                x_end = min(x_start + self.tile_size, width)

                # Calculate padding (clamped to image boundaries)
                pad_top = min(y_start, self.tile_pad)
                pad_bottom = min(height - y_end, self.tile_pad)
                pad_left = min(x_start, self.tile_pad)
                pad_right = min(width - x_end, self.tile_pad)

                tiles.append(
                    TileInfo(
                        row=r,
                        col=c,
                        x_start=x_start,
                        y_start=y_start,
                        x_end=x_end,
                        y_end=y_end,
                        pad_left=pad_left,
                        pad_right=pad_right,
                        pad_top=pad_top,
                        pad_bottom=pad_bottom,
                        index=index,
                    )
                )
                index += 1

        return tiles

    def total_tiles(self, height: int, width: int) -> int:
        """Return total number of tiles for progress calculation."""
        rows = math.ceil(height / self.tile_size)
        cols = math.ceil(width / self.tile_size)
        return rows * cols

    def extract_tile(self, image: np.ndarray, tile: TileInfo) -> np.ndarray:
        """Extract a tile from the image with overlap padding.

        Args:
            image: HWC uint8 numpy array.
            tile: TileInfo describing the tile position.

        Returns:
            The extracted tile (with padding) as HWC uint8 array.
        """
        y_from = tile.y_start - tile.pad_top
        y_to = tile.y_end + tile.pad_bottom
        x_from = tile.x_start - tile.pad_left
        x_to = tile.x_end + tile.pad_right

        return image[y_from:y_to, x_from:x_to].copy()

    def place_tile(
        self,
        canvas: np.ndarray,
        tile_output: np.ndarray,
        tile: TileInfo,
        scale: int,
    ) -> None:
        """Place a processed tile into the output canvas, cropping the overlap padding.

        Args:
            canvas: Output HWC uint8 array to write into.
            tile_output: Processed tile (HWC uint8) including scaled padding.
            tile: TileInfo for this tile.
            scale: The model's upscale factor.
        """
        # Calculate the crop region to remove padding from the output
        crop_top = tile.pad_top * scale
        crop_bottom = tile_output.shape[0] - tile.pad_bottom * scale
        crop_left = tile.pad_left * scale
        crop_right = tile_output.shape[1] - tile.pad_right * scale

        # Handle edge case where pad_bottom or pad_right is 0
        if tile.pad_bottom == 0:
            crop_bottom = tile_output.shape[0]
        if tile.pad_right == 0:
            crop_right = tile_output.shape[1]

        cropped = tile_output[crop_top:crop_bottom, crop_left:crop_right]

        # Place into canvas at the scaled tile position
        out_y = tile.y_start * scale
        out_x = tile.x_start * scale
        h, w = cropped.shape[:2]

        canvas[out_y : out_y + h, out_x : out_x + w] = cropped
