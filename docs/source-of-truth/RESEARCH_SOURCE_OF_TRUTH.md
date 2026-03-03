# RESEARCH_SOURCE_OF_TRUTH.md | PixelForge | Verified 2026-03-02

Source of truth for AI super-resolution models | libraries | hardware constraints | architecture decisions
All data verified against official documentation and benchmarks | update this file when specs change

---

## 1. Super-Resolution Models Reference

### Tier 1: Best Quality (Slower)

**HAT (Hybrid Attention Transformer)**
- Source: https://github.com/XPixelGroup/HAT
- Paper: CVPR 2023 → TPAMI 2025 update
- Architecture: Channel attention + window self-attention + overlapping cross-attention
- Scales: x2, x3, x4
- Quality: Highest PSNR/SSIM on standard benchmarks (Set5, Set14, Urban100, Manga109)
- Speed: Slowest of the bunch | highest VRAM usage
- Weights: Google Drive / Kaggle (https://www.kaggle.com/datasets/djokester/hat-pre-trained-weights-for-super-resolution)
- Best for: Maximum quality when speed is not critical | offline batch processing
- GTX 1050 4GB: Requires small tiles (256px) | may be impractical for large images

**DAT (Dual Aggregation Transformer)**
- Source: ICCV 2023 paper
- Architecture: Alternating spatial and channel self-attention
- Quality: DAT+ outperforms SwinIR by 0.41 dB on Manga109 (x2)
- Used as baseline in NTIRE 2024 Challenge (x4)
- Similar VRAM profile to HAT
- GTX 1050 4GB: Same constraints as HAT

**SwinIR (Swin Transformer for Image Restoration)**
- Source: https://github.com/JingyunLiang/SwinIR
- Architecture: Swin Transformer (Microsoft Research)
- Quality: ~9.7/10 in comparative tests
- Speed: ~12 seconds for 1080p→4K on RTX 4090
- Best for: Both photos and digital art | very versatile
- Mature ecosystem with wide community adoption

### Tier 2: Best Practical (Speed + Quality Balance)

**Real-ESRGAN** | RECOMMENDED DEFAULT
- Source: https://github.com/xinntao/Real-ESRGAN
- PyPI: https://pypi.org/project/realesrgan/
- Quality: ~9.2/10 | processes in ~6 seconds (RTX 4090, 1080p→4K)
- Trained on real-world degraded images (compression artifacts, noise, blur)
- Key model variants:
  - `RealESRGAN_x4plus` — photographs and general images
  - `RealESRGAN_x4plus_anime_6B` — anime, manga, digital illustration (smaller model)
  - `realesr-general-x4v3` — general purpose, newer
- ncnn-vulkan port available for cross-platform GPU without CUDA
- Widest community adoption | used by Upscayl, chaiNNer, ComfyUI
- GTX 1050 4GB: Runs well with 256-400px tiles | fp16 for memory savings
- **THIS IS THE DEFAULT MODEL FOR PIXELFORGE**

**BSRGAN**
- Part of ESRGAN family | broader degradation model training
- Good for severely degraded images
- Same RRDBNet architecture as Real-ESRGAN

### Content-Specific Recommendations

| Content Type | Best Model | Runner-Up |
|---|---|---|
| Photographs (real-world) | Real-ESRGAN x4plus | HAT / SwinIR |
| Anime / Manga / Illustration | Real-ESRGAN Anime 6B | Waifu2x |
| Digital art / Mixed | SwinIR | DAT |
| Benchmark / Maximum PSNR | HAT / DAT | SwinIR |
| Speed-critical / Batch | Real-ESRGAN Compact | ESRGAN |

### 4K Upscaling Math

| Input Resolution | Scale | Output Resolution | Category |
|---|---|---|---|
| 960x540 | x4 | 3840x2160 | True 4K |
| 1280x720 | x4 | 5120x2880 | 5K |
| 1920x1080 | x2 | 3840x2160 | True 4K |
| 1920x1080 | x4 | 7680x4320 | 8K |

---

## 2. Python Libraries & Frameworks

### Inference: PyTorch (Primary)

- Nearly ALL state-of-the-art SR models are PyTorch-based
- TensorFlow has negligible presence in super-resolution
- Supports: CUDA (NVIDIA) | ROCm (AMD) | MPS (Apple Silicon) | CPU
- Current stable: PyTorch 2.x
- MUST use `torch.no_grad()` for all inference
- MUST call `torch.cuda.empty_cache()` between images in batch mode

### Spandrel | Universal Model Loader | RECOMMENDED

- Source: https://github.com/chaiNNer-org/spandrel
- PyPI: https://pypi.org/project/spandrel/
- Auto-detects architecture and hyperparameters from .pth / .pt / .ckpt / .safetensors files
- Supports: ESRGAN, Real-ESRGAN, Real-ESRGAN Compact, SwinIR, HAT, DAT, BSRGAN, 50+ architectures
- Two packages:
  - `spandrel` (MIT/Apache licensed architectures)
  - `spandrel_extra_arches` (non-commercial architectures)
- Unified API:
  ```python
  import spandrel
  model = spandrel.ModelLoader().load_from_file("RealESRGAN_x4plus.pth")
  model = model.to("cuda").eval()
  # model.scale, model.input_channels, model.architecture auto-detected
  ```
- Used by: Stable Diffusion WebUI (AUTOMATIC1111), InvokeAI, chaiNNer

### BasicSR | Training Framework

- Source: https://github.com/XPixelGroup/BasicSR
- Training/inference framework from Tencent/XPixelGroup
- Used to train Real-ESRGAN, HAT, SwinIR
- Heavier than Spandrel | only needed if training custom models

### realesrgan (PyPI Package) | Official Real-ESRGAN

- Source: https://pypi.org/project/realesrgan/
- Official package from Real-ESRGAN team
- Built-in tile-based inference | face enhancement (GFPGAN) | CLI tools
- Depends on: basicsr, facexlib
- Good for quick integration | Spandrel is more flexible for multi-model support

### Pillow (PIL) | Image I/O

- Handles: PNG, JPEG, WebP, TIFF, BMP
- ICC profile preservation via save(icc_profile=...) parameter
- RGBA handling: separate alpha channel before model inference

### PySide6 (Qt 6) | GUI Framework

- LGPL license | commercial-friendly
- QGraphicsView for image preview with zoom/pan
- QThread for non-blocking inference
- Signal/slot for progress reporting
- Native file dialogs | high-DPI support
- Cross-platform (Linux, Windows, macOS)

---

## 3. Hardware Profile & Constraints

Full reference: docs/source-of-truth/HARDWARE_PROFILE.md

### Target System (Primary Development)

| Component | Spec |
|---|---|
| CPU | Intel Core i7-7700HQ @ 2.80GHz (4C/8T) |
| GPU | NVIDIA GeForce GTX 1050 Mobile |
| VRAM | 4 GB GDDR5 |
| CUDA Compute | 6.1 (Pascal architecture) |
| RAM | 16 GB DDR4 |
| Disk | ~23 GB free |
| OS | Linux (Ubuntu-based) |

### VRAM Budget Guidelines

| Tile Size | Approx VRAM Usage (Real-ESRGAN x4) | Safe for 4GB? |
|---|---|---|
| 256px | ~1.5 GB | YES (recommended default) |
| 400px | ~2.5 GB | YES |
| 512px | ~3.5 GB | RISKY |
| 768px | ~6+ GB | NO |

### Critical Constraints for GTX 1050

- No Tensor Cores → fp16 saves memory but does NOT speed up inference
- 4 GB VRAM → tile_size=256 default | max safe = 400
- CUDA 6.1 → supports fp16 storage but not fp16 compute acceleration
- Pascal architecture → some newer PyTorch features may not be optimal
- MUST implement graceful OOM handling | catch torch.cuda.OutOfMemoryError
- MUST auto-reduce tile size on OOM and retry

---

## 4. Tile-Based Inference | Implementation Reference

```
Algorithm:
1. Load input image (H x W x C)
2. Calculate tile grid: rows = ceil(H / tile_size), cols = ceil(W / tile_size)
3. For each tile (i, j):
   a. Extract tile with tile_pad pixels overlap on all sides
   b. Normalize to [0, 1] float32
   c. Convert to tensor, move to device
   d. Run model inference (with torch.no_grad())
   e. Crop output by (tile_pad * scale) on all sides
   f. Place cropped output into result canvas
   g. Report progress: (tile_index / total_tiles) * 100
4. Convert result canvas back to uint8
5. Save output image with metadata (ICC profile, format-specific settings)
```

Key parameters:
- tile_size: 256 default (4GB GPU safe) | user-configurable
- tile_pad: 32 pixels (overlap to prevent seam artifacts)
- scale: model-dependent (x2, x3, x4)
- precision: fp16 default on CUDA | fp32 fallback

---

## 6. Face Restoration | GFPGAN

### GFPGAN v1.4 | Post-Processing Face Enhancement

- Source: https://github.com/TencentARC/GFPGAN
- PyPI: `pip install gfpgan` (depends on basicsr, facexlib, opencv-python)
- Architecture: GAN Prior Embedded Network (StyleGAN2 prior + U-Net degradation removal)
- Paper: CVPR 2021 (GFPGAN: Towards Real-World Blind Face Restoration with Generative Facial Prior)

**Model weights:**
- GFPGANv1.4.pth — 349 MB
- URL: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
- VRAM: ~700 MB (fp16) | ~1.4 GB (fp32)

**How it works in PixelForge:**
1. Super-resolution runs first (Real-ESRGAN / HAT / SwinIR — tiled)
2. GFPGAN detects faces in the upscaled image via RetinaFace (facexlib)
3. Each detected face is cropped, enhanced through the GFPGAN model, and composited back
4. `upscale=1` — no additional upscaling, just face restoration
5. `weight=0.5` — blends 50% original + 50% restored for natural look

**GTX 1050 4GB notes:**
- GFPGAN runs sequentially AFTER upscaling (not simultaneously) — avoids VRAM conflict
- Face patches are 512x512 — well within 4GB VRAM budget
- Model load/unload per session (~2-5 seconds overhead)

**Known compatibility issue:**
- `basicsr/data/degradations.py` has a broken import (`torchvision.transforms.functional_tensor`) on PyTorch 2.x + modern torchvision
- This affects training/data-augmentation code only — inference path is unaffected
- If import fails at runtime: update basicsr to latest version or patch the import

---

## 5. Sources

### Super-Resolution Models
- https://github.com/xinntao/Real-ESRGAN
- https://github.com/XPixelGroup/HAT
- https://github.com/JingyunLiang/SwinIR
- https://arxiv.org/abs/2309.05239 (HAT TPAMI 2025)
- https://arxiv.org/abs/2308.03364 (DAT ICCV 2023)

### Face Restoration
- https://github.com/TencentARC/GFPGAN
- https://pypi.org/project/gfpgan/

### Libraries
- https://github.com/chaiNNer-org/spandrel
- https://pypi.org/project/spandrel/
- https://pypi.org/project/realesrgan/
- https://github.com/XPixelGroup/BasicSR
- https://doc.qt.io/qtforpython-6/

### Benchmarks & Comparisons
- https://apatero.com/blog/ai-image-upscaling-battle-esrgan-vs-beyond-2025
- https://apatero.com/blog/fastest-esrgan-upscaling-models-quality-comparison-2025
- NTIRE 2024 Challenge on Image Super-Resolution (x4) | CVPR 2024 Workshop

### Community
- https://github.com/Phhofm/models (600+ community upscaling models)
- https://upscayl.org/ (Electron desktop app using Real-ESRGAN ncnn-vulkan)
- https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan
