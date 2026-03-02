"""Model weight downloading with progress tracking and integrity verification."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from collections.abc import Callable
from pathlib import Path

from src.engine.model_registry import ModelEntry

logger = logging.getLogger(__name__)

DownloadProgress = Callable[[int, int], None]  # (bytes_downloaded, bytes_total)


class ModelDownloader:
    """Downloads model weights from URLs with progress and integrity verification."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir

    def download(
        self,
        entry: ModelEntry,
        progress_callback: DownloadProgress | None = None,
    ) -> Path:
        """Download a model if not already present.

        Downloads to a .part file first, then renames atomically on completion.
        Verifies SHA256 hash if available.

        Args:
            entry: ModelEntry with URL and filename.
            progress_callback: Optional (bytes_downloaded, bytes_total) callback.

        Returns:
            Path to the downloaded model file.
        """
        target = self._models_dir / entry.filename

        if target.exists():
            logger.info("Model already downloaded: %s", entry.filename)
            return target

        self._models_dir.mkdir(parents=True, exist_ok=True)
        part_file = target.with_suffix(target.suffix + ".part")

        logger.info("Downloading %s from %s", entry.filename, entry.url)

        def reporthook(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if progress_callback:
                progress_callback(min(downloaded, total_size), total_size)

        try:
            urllib.request.urlretrieve(entry.url, str(part_file), reporthook=reporthook)
        except Exception:
            # Clean up partial download
            if part_file.exists():
                part_file.unlink()
            raise

        # Verify SHA256 if available
        if entry.sha256:
            file_hash = self._compute_sha256(part_file)
            if file_hash != entry.sha256:
                part_file.unlink()
                raise ValueError(
                    f"SHA256 mismatch for {entry.filename}: "
                    f"expected {entry.sha256}, got {file_hash}"
                )
            logger.info("SHA256 verified: %s", entry.filename)

        # Atomic rename
        part_file.rename(target)
        logger.info("Download complete: %s", target)
        return target

    def is_downloaded(self, entry: ModelEntry) -> bool:
        return (self._models_dir / entry.filename).exists()

    def get_model_path(self, entry: ModelEntry) -> Path:
        return self._models_dir / entry.filename

    def delete_model(self, entry: ModelEntry) -> bool:
        path = self._models_dir / entry.filename
        if path.exists():
            path.unlink()
            logger.info("Deleted model: %s", entry.filename)
            return True
        return False

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
