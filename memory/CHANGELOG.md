# CHANGELOG | PixelForge

Format: [Date] | Batch description | Files changed | Notes

---

## 2026-03-02 | Stability Fixes — VRAM Safety & Worker Crash Prevention

**Scope:** Bug fixes for OOM crashes and QThread destructor crashes.

**Files changed:**
- `src/constants.py` — `DEFAULT_TILE_SIZE` 256→128 (leaves VRAM headroom for display compositor)
- `src/engine/upscaler.py` — Added `torch.cuda.set_per_process_memory_fraction(0.75)` to cap CUDA at 3 GB on 4GB GPUs
- `src/gui/main_window.py` — Worker null-out via `_release_worker()`, cancel-before-restart, `closeEvent` cleanup
- `src/workers/upscale_worker.py` — Refactored from `QThread` to `threading.Thread` (prevents `QThread::~QThread()` abort() crash during GC)

**Root cause fixed:** `QThread.__del__` calls `abort()` if thread still running when Python GC collects the object. This caused hard crashes during CUDA inference. `threading.Thread` has no such destructor check.

**Known issues:** None.

---

## 2026-03-02 | Feature: Face Enhancement via GFPGAN

**Scope:** New "Enhance faces" post-processing step (optional, toggle in Settings panel).

**Files changed:**
- `memory/CHANGELOG.md` — created (this file)
- `requirements.txt` — added `gfpgan>=1.3.8`, `opencv-python>=4.8`
- `src/engine/face_enhancer.py` — new: `FaceEnhancer` class wrapping GFPGANer; `GFPGAN_MODEL` ModelEntry
- `src/gui/controls_panel.py` — added "Enhance faces (GFPGAN)" checkbox to Settings group
- `src/workers/upscale_worker.py` — added `face_model_path` param + `stage` Signal
- `src/gui/main_window.py` — wired face enhancement: download check, worker construction, stage signal
- `docs/source-of-truth/RESEARCH_SOURCE_OF_TRUTH.md` — added Section 6: Face Restoration

**Architecture decision:** GFPGAN runs *after* super-resolution (sequential, not simultaneous) to avoid VRAM conflict on 4GB GPUs. FaceEnhancer created fresh in worker thread each run.

**Known issues:** `gfpgan` pip package has a basicsr/torchvision import issue for PyTorch 2.x training code. Inference path is unaffected. If `import gfpgan` fails, a clear error with install instructions is shown.
