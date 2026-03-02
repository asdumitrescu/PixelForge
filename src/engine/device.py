"""GPU/CPU device detection and VRAM management."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    """Information about the active compute device."""

    device: torch.device
    name: str
    vram_total_mb: int
    vram_free_mb: int
    compute_capability: tuple[int, int] | None
    is_cuda: bool


class DeviceManager:
    """Detects and manages the compute device (CUDA / CPU)."""

    def __init__(self) -> None:
        self._device = self.detect_device()

    @property
    def device(self) -> torch.device:
        return self._device

    @staticmethod
    def detect_device() -> torch.device:
        """Auto-detect best available device: CUDA > CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def get_device_info(self) -> DeviceInfo:
        """Return detailed info about the current device."""
        if self._device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            return DeviceInfo(
                device=self._device,
                name=props.name,
                vram_total_mb=total // (1024 * 1024),
                vram_free_mb=free // (1024 * 1024),
                compute_capability=(props.major, props.minor),
                is_cuda=True,
            )
        return DeviceInfo(
            device=self._device,
            name="CPU",
            vram_total_mb=0,
            vram_free_mb=0,
            compute_capability=None,
            is_cuda=False,
        )

    def supports_half(self) -> bool:
        """Check if device supports fp16 storage (CUDA compute >= 5.3)."""
        if not self._device.type == "cuda":
            return False
        props = torch.cuda.get_device_properties(0)
        return (props.major, props.minor) >= (5, 3)

    def get_vram_free_mb(self) -> int:
        """Return free VRAM in MB. Returns 0 for CPU."""
        if self._device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(0)
            return free // (1024 * 1024)
        return 0

    def clear_cache(self) -> None:
        """Free cached GPU memory."""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def force_cpu(self) -> None:
        """Force device to CPU (for OOM fallback)."""
        self._device = torch.device("cpu")

    def reset_device(self) -> None:
        """Re-detect the best available device."""
        self._device = self.detect_device()
