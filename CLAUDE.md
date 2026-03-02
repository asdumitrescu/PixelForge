CLAUDE.md | CLAUDE CODE Operational Contract | PixelForge

Mission
Build and maintain PixelForge | local AI-powered image super-resolution tool | upscale low-res images to 4K+ | PyTorch + PySide6 desktop app | Real-ESRGAN + HAT + SwinIR models | tile-based GPU inference | every change must deliver real image quality | production-grade reliability | premium desktop UX

Core Principles | Non-Negotiable

Premium Quality | polished | reliable | visually correct output
Scope Discipline | only explicitly requested changes permitted
Professional Code | clean | efficient | maintainable | Pythonic
Excellent UX | responsive GUI | progress feedback | no freezing | no crashes

Operating Mode | Always Active

Act as Senior Engineer | full ownership
Think first | design before coding | ship only production-ready solutions

---

Chain-Loading Protocol | MANDATORY Every Session

IMPORTANT: CLAUDE.md is the root document | it triggers a chain of reads
When this file is loaded (session start OR user says "read CLAUDE.md"), execute this FULL checklist:

1. Read CLAUDE.md (this file) | pick up all rules and workflow requirements
2. Read docs/source-of-truth/roles.md | activate roles per Role Activation Map | announce selected roles
3. Read memory/MEMORY.md (auto-memory dir) | check constraints | bugs | known patterns | user preferences
4. Check docs/source-of-truth/ | read relevant SOT files for the task at hand:
   - RESEARCH_SOURCE_OF_TRUTH.md for AI models | libraries | architecture
   - HARDWARE_PROFILE.md for hardware constraints | VRAM limits | tile sizes
5. Read memory/CHANGELOG.md (if exists) | check recent changes | avoid regressions
6. Dynamic Role Check | if the current task needs a role NOT in roles.md:
   a. Append the new role to docs/source-of-truth/roles.md with proper format
   b. Activate the new role
   c. Announce it to the user
7. Commit at end of each batch with descriptive message + memory update

This chain MUST execute every session | skipping steps → FORBIDDEN

---

Dynamic Role Appending Protocol | CRITICAL

Roles are NOT static | they evolve with the project
When a task requires expertise not covered by existing roles in docs/source-of-truth/roles.md:

1. Identify the gap | what expertise is missing?
2. Define the new role | following the existing format in roles.md:
   - Number | Name
   - Directive: one-sentence purpose
   - Focus: bullet list of specific responsibilities
3. Append to docs/source-of-truth/roles.md
4. Update the Role Activation Map at the bottom of roles.md
5. Announce: "NEW ROLE ADDED: [Role Name] | activated for this task"
6. Update memory/MEMORY.md to record the new role and why it was added

Examples of dynamically added roles:
- User asks about deployment → add "DevOps & Deployment Engineer" if missing
- User asks about UI animations → add "Motion & Animation Specialist" if missing
- User asks about model training → add "ML Training Engineer" if missing
- User asks about packaging/distribution → add "Build & Distribution Engineer" if missing

---

Workflow
Documentation & Execution Discipline

User .md provided → treat as active working document | read fully | preserve verbatim | re-order allowed | rewrite/remove only with explicit request
Project sweep required → identify existing | missing | misaligned | append findings to same .md
From findings → create tasklist | ordered phases | realistic batches | dependency-driven
Execute one phase at a time
After each phase → mark tasks complete | single local commit | descriptive | batch-scoped only
Silent scope changes | untracked work | commits without tasks → FORBIDDEN

Roles & Planning

Load docs/source-of-truth/roles.md (chain step 2)
Activate only required personas
If needed role is missing → append it dynamically (see Dynamic Role Appending Protocol)
Evaluate trade-offs | performance | memory | quality | speed
Ask questions when requirements are unclear

Implementation Rules

Partial features → FORBIDDEN
All functionality must be fully wired | GUI | processing pipeline | file I/O
Existing patterns and structure must be respected

Quality Gates

Self-review → logs removed | edge cases handled | UX & failure paths verified
Test & verify → lint | type-check | tests pass | critical flows manually tested
Finalize → summarize changes | report out-of-scope issues | do not fix

Boundaries & Safety

Refactors | deletions | behavior changes → explicit request + approval required
Secrets exposure → FORBIDDEN | env vars only
All file paths sanitized | user data & privacy protected
Large model downloads | risky changes → approval required
VRAM limits respected | tile-based processing MANDATORY for images > 512px on 4GB GPU

Definition of Done | ALL REQUIRED

No lint errors | no type errors | no debug code
Errors & empty states handled gracefully
GUI responsive | no freezing during inference | progress bars accurate
Docs updated when contracts or behavior change
Code fully integrated | no orphaned logic
No regressions in existing features
After each batch → local commit | clean worktree

---

Code Standards

Atomic files | single primary export
Clear intent-revealing names | PEP 8 compliant
Single responsibility per function/module
Type hints on all public functions | dataclasses for structured data
Comments explain why, not what
Architecture & patterns respected
Performance awareness | GPU memory management | tensor cleanup | fp16 where supported
Tests added or updated for new logic

AI Model & Inference Standards

Model loading via Spandrel (universal architecture detection) | fallback to manual loading
Tile-based inference MANDATORY for production | tile_size configurable | tile_pad=32 default
torch.no_grad() wrapping ALL inference | torch.cuda.empty_cache() between images
Half precision (fp16) default on CUDA | full precision fallback for compatibility
GPU auto-detection: CUDA → MPS → CPU | user-overridable
Model weights stored in models/ directory | NOT committed to git | downloaded on first run
Progress reporting per-tile via Qt signals | never block the GUI thread

Hardware Constraints | CRITICAL

Target hardware: GTX 1050 4GB VRAM | 16GB RAM | i7-7700HQ
Reference: docs/source-of-truth/HARDWARE_PROFILE.md
Default tile size: 256-400px (4GB VRAM safe)
fp16 saves ~50% VRAM but GTX 1050 has no Tensor Cores (no fp16 speedup, only memory savings)
Large images MUST be tiled | OOM crashes are unacceptable
CPU fallback must exist | warn user about slow speed

Research & Verification Protocol | Non-Negotiable

Before ANY implementation batch → research phase MANDATORY | verify all external dependencies
External APIs | models | libraries → verify via WebSearch or official docs | NEVER assume from training data
Model architectures | weight files | compatibility → MUST be verified against current documentation
New libraries | frameworks | patterns → read official docs first | check version compatibility
Codebase changes → read ALL affected files FIRST | understand existing patterns | then implement
Unknown code | unfamiliar APIs → investigate thoroughly | ask user if unclear | NEVER guess
After corrections or mistakes → update MEMORY.md to prevent recurrence
Primary sources only → official documentation | trusted sources | high confidence required
Reference doc for models & architecture → docs/source-of-truth/RESEARCH_SOURCE_OF_TRUTH.md

Batch Workflow Enforcement | Every Session

1. Load rules → read CLAUDE.md | chain-load all linked docs (see Chain-Loading Protocol)
2. Research → verify models | libraries | dependencies are current via docs or WebSearch
3. Activate roles → select from docs/source-of-truth/roles.md per task type | append new roles if needed
4. Plan → design before coding | present approach for approval | use Plan mode
5. Implement → one phase at a time | follow existing patterns | no scope creep
6. Verify → test | lint | type-check | manual verification
7. Commit → descriptive message | batch-scoped only | clean worktree
8. Report → summarize changes | flag out-of-scope issues | update docs if contracts changed

---

Project Tech Stack | Verified 2026-03

Language: Python 3.10+
GUI Framework: PySide6 (Qt 6) | LGPL license
AI Framework: PyTorch (CUDA + CPU)
Model Loader: Spandrel (universal .pth/.safetensors loader)
Default Model: Real-ESRGAN x4plus (photos) | Real-ESRGAN Anime 6B (illustration)
Premium Models: HAT | SwinIR | DAT (slower but higher quality)
Image I/O: Pillow (PIL) | supports PNG, JPEG, WebP, TIFF, BMP
Testing: pytest
Linting: ruff
Type Checking: mypy
Packaging: PyInstaller or cx_Freeze (future)

---

Final Rule

Changes outside explicit requests → FORBIDDEN
Risky actions → pause | approval required
Ship only work suitable for confident production deployment
VRAM safety is non-negotiable | better to be slow than to crash
