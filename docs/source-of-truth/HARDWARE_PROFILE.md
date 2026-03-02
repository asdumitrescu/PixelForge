# HARDWARE_PROFILE.md | PixelForge | Verified 2026-03-02

Primary development and target hardware profile.
All tile sizes, memory budgets, and performance estimates are based on this hardware.
Update this file when hardware changes or when new benchmarks are available.

---

## Primary System

| Component | Specification | Notes |
|---|---|---|
| CPU | Intel Core i7-7700HQ @ 2.80GHz | 4 cores / 8 threads | Turbo to 3.8GHz |
| GPU | NVIDIA GeForce GTX 1050 Mobile | Pascal architecture (GP107M) |
| VRAM | 4 GB GDDR5 | CRITICAL CONSTRAINT |
| CUDA Compute | 6.1 | fp16 storage supported, no fp16 compute acceleration (no Tensor Cores) |
| RAM | 16 GB DDR4 | Sufficient for image processing buffers |
| Disk | ~23 GB free on / | Model weights ~100-500MB each | output images can be large |
| OS | Linux (Ubuntu-based) | Kernel 6.8.0-100-generic |
| Driver | NVIDIA 535.288.01 | CUDA toolkit compatible |

---

## GPU Capabilities & Limitations

### What GTX 1050 CAN Do
- CUDA inference with PyTorch
- fp16 tensor storage (saves ~50% VRAM vs fp32)
- Tile-based super-resolution with tiles up to ~400px
- Real-ESRGAN, ESRGAN, and compact models comfortably
- Process images of any size via tiling

### What GTX 1050 CANNOT Do Efficiently
- fp16 compute acceleration (no Tensor Cores → fp16 is NOT faster, only smaller)
- Process tiles > 512px without OOM risk
- Run HAT or DAT on large tiles (very VRAM-hungry transformers)
- Hold a full 4K image in VRAM without tiling

### VRAM Budget Table

| Tile Size | Real-ESRGAN (x4) | SwinIR (x4) | HAT (x4) | Safe? |
|---|---|---|---|---|
| 192px | ~0.8 GB | ~1.0 GB | ~1.2 GB | YES |
| 256px | ~1.5 GB | ~1.8 GB | ~2.2 GB | YES (default) |
| 320px | ~2.0 GB | ~2.5 GB | ~3.0 GB | YES |
| 400px | ~2.5 GB | ~3.2 GB | ~3.8 GB | MARGINAL |
| 512px | ~3.5 GB | ~4.5 GB | ~5.5 GB | RISKY (Real-ESRGAN only) |

Note: These are approximate | actual usage depends on model variant, batch size, and PyTorch overhead.
Always maintain ~500MB VRAM headroom for PyTorch internals and OS.

### OOM Recovery Strategy

1. Catch `torch.cuda.OutOfMemoryError`
2. Call `torch.cuda.empty_cache()`
3. Reduce tile_size by 25% (e.g., 400 → 300 → 225)
4. Retry with smaller tiles
5. If tile_size < 128 and still OOM → fall back to CPU with user warning
6. NEVER crash on OOM | always degrade gracefully

---

## Performance Estimates (GTX 1050)

These are rough estimates | actual benchmarks should be recorded once implemented.

| Input | Output | Model | Tile | Est. Time |
|---|---|---|---|---|
| 960x540 | 3840x2160 (4K) | Real-ESRGAN x4 | 256px | ~30-60s |
| 1280x720 | 5120x2880 | Real-ESRGAN x4 | 256px | ~60-120s |
| 1920x1080 | 3840x2160 (4K) | Real-ESRGAN x2 | 256px | ~15-30s |
| 640x480 | 2560x1920 | Real-ESRGAN x4 | 256px | ~10-20s |

Note: HAT/SwinIR will be 3-5x slower than Real-ESRGAN on this hardware.

---

## CPU Fallback

- Intel i7-7700HQ can do inference but VERY slowly
- Expect 10-30x slower than GPU for Real-ESRGAN
- Acceptable for: small images (< 512px), one-off tasks, testing
- MUST warn user when falling back to CPU
- Consider ONNX Runtime export for ~2-3x CPU speedup (future optimization)

---

## Disk Space Considerations

| Item | Size | Notes |
|---|---|---|
| Real-ESRGAN x4plus weights | ~67 MB | Default model |
| Real-ESRGAN Anime 6B weights | ~17 MB | Smaller model |
| SwinIR weights | ~12-48 MB | Varies by variant |
| HAT weights | ~40-160 MB | Varies by size (S/M/L) |
| PyTorch + dependencies | ~2-3 GB | pip install |
| PySide6 | ~100-200 MB | pip install |
| Total minimum | ~3 GB | With one model |
| Total with all models | ~4-5 GB | With multiple models |

With 23 GB free, disk space is sufficient but not abundant.
Warn user if disk space drops below 5 GB during model downloads.
