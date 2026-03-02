"""Tests for device detection and management."""

import torch

from src.engine.device import DeviceInfo, DeviceManager


def test_detect_device_returns_valid() -> None:
    device = DeviceManager.detect_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "cpu")


def test_device_manager_creates_device() -> None:
    dm = DeviceManager()
    assert dm.device is not None
    assert dm.device.type in ("cuda", "cpu")


def test_device_info_has_required_fields() -> None:
    dm = DeviceManager()
    info = dm.get_device_info()
    assert isinstance(info, DeviceInfo)
    assert isinstance(info.name, str)
    assert len(info.name) > 0
    assert isinstance(info.vram_total_mb, int)
    assert isinstance(info.vram_free_mb, int)
    assert isinstance(info.is_cuda, bool)


def test_supports_half_returns_bool() -> None:
    dm = DeviceManager()
    result = dm.supports_half()
    assert isinstance(result, bool)


def test_get_vram_free_returns_int() -> None:
    dm = DeviceManager()
    free = dm.get_vram_free_mb()
    assert isinstance(free, int)
    assert free >= 0


def test_force_cpu() -> None:
    dm = DeviceManager()
    dm.force_cpu()
    assert dm.device.type == "cpu"


def test_reset_device() -> None:
    dm = DeviceManager()
    dm.force_cpu()
    dm.reset_device()
    # After reset, should be back to best available
    assert dm.device.type in ("cuda", "cpu")


def test_clear_cache_does_not_raise() -> None:
    dm = DeviceManager()
    # Should not raise regardless of device type
    dm.clear_cache()
