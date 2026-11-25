# meshforge

Self-sufficient image → 3D-print pipeline. Its main part is the
**print-prep stage** (mesh repair, auto-splitting, dowel joints, scarring/support
minimization) that runs locally on a generated mesh.

Hardware reality: dev laptop is an Intel Core Ultra 7 255U, 31.5 GB RAM, **no CUDA
GPU**. So heavy generation runs on **free cloud GPU**, geometry runs locally.

## Pipeline

```
Image
  │
  ▼
[A] Generation backbone            raw mesh (.glb)
    Tencent Hunyuan3D-2, run on a FREE GPU notebook (Kaggle T4/P100 or Colab T4).
    notebook_server.py launches a gradio share URL; backend/generate.py drives it
    via gradio_client. Shape-only (~6 GB) by default; --texture optional.
  │
  ▼
[B] Print-prep pipeline  (local, CPU)   ← the differentiator
    1. Orientation optimizer   — search rotations; minimize support-contact area
                                 (= scarring) + overhang + footprint stability.
    2. Auto-split              — cut model into plate-fitting parts along seams
                                 that reduce overhangs; connectivity graph.
    3. Dowel/peg joints        — insert peg + socket pairs at each cut (boolean
                                 via manifold3d), with print-tolerance clearance.
    4. Scarring risk map       — per-face heatmap of likely support scars; report.
  │
  ▼
[C] App shell  (local UI)  +  manual Bambu P1S print step (user does this)
```

## Compute decision (June 2026)

- HF **ZeroGPU is NOT viable**: free to *use* others' Spaces (3.5 min/day) but
  *creating/duplicating* a ZeroGPU Space needs Pro ($9/mo). Ruled out — no payment.
- TRELLIS.2-4B **OOMs on a free 16 GB T4** → also ruled out for the free lane.
- **Chosen free lane:** Hunyuan3D-2 on a self-launched **Kaggle/Colab** GPU notebook
  (`notebook_server.py`), exposed via gradio `share=True` -> public `*.gradio.live`
  URL, driven by `gradio_client`. Shape-only ~6 GB = robust on free T4.
  Tradeoff: the share URL is ephemeral (72 h / per-session) — acceptable cost of free.

## Status

- [x] Project scaffold + architecture
- [x] Half A: free GPU notebook server (`backend/notebook_server.py`, Hunyuan3D-2)
- [x] Half A: local client (`backend/generate.py`) — drives the notebook's gradio URL
- [x] Half A: verified end-to-end (test_blob.png → 411k-vert .glb via free Colab T4, 2026-06-22)
- [x] Half B: `repair.py` — watertight repair + scale-to-mm (verified: blob → watertight, 150 mm)
- [x] Half B: `docs/PRINT_FAILURES.md` — failure taxonomy → mitigation → stage map
- [x] Half B: `color.py` — quantize to AMS palette → per-face filament (no-color path verified)
- [x] Half B: `orient.py` — orientation optimizer (verified, score 0.36 on real mesh)
- [x] Half B: `split.py` — multi-parameter split + dowel joints (verified; fits-plate early-return safe)
- [x] Half B: `analyze.py` — failure risk engine (verified: overhang/support/footprint/thin/islands)
- [x] Half B: `assemble3mf.py` — lib3mf → `.3mf` (verified: valid 3mf, 1 obj/34998 tris re-opens)
- [x] Half B: `slicer.py` — Bambu CLI wrapper + presets (built; DEFERRED/unsafe — see freeze note)
- [x] Half B: `pipeline.py` — B2B orchestrator, verified end-to-end mesh→.3mf (7.6s, 0 errors)
- [ ] Half B v2: embed Bambu presets + paint_color into the .3mf (fleshed-out project); needs textured Half-A run for color
- [ ] Half B v2: safe single-shot slice validation (do_slice off by default)
- [ ] Half C: app shell (UI over the pipeline)

## ⚠️ Slicer freeze post-mortem (2026-06-22) — DO NOT REPEAT

The verify phase froze the machine. Cause: the workflow ran `slicer.slice_project`
from MULTIPLE agents concurrently (build smoke + verify), each launching the full
`bambu-studio.exe` GUI app via Start-Process, each generating 100s of MB of temp
files — concurrently with an 8.7 GB `manifold3d` boolean process. The pipeline itself
is fine; the swarm of GUI slicer launches was the problem.

Rules now: (1) `do_slice=False` by default. (2) NEVER run the slicer from parallel
agents or a workflow. (3) Only ever slice ONCE, standalone, in an isolated temp dir,
with a timeout, while watching it. (4) The `.3mf` project is the deliverable — user
slices in the Bambu GUI manually. (5) Big-mesh `manifold` booleans: decimate first.

## Half B module map (backend/printprep/)

```
pipeline.py     orchestrates: generate → repair → color → orient → split → analyze → assemble3mf → slicer
repair.py   [x] watertight repair + scale to mm
color.py        texture/vertex-color → quantize to N filaments → per-face index (paint_color)
orient.py       search orientations; minimize support-contact (scarring)+overhang+footprint
split.py        multi-param cut scoring (see PRINT_FAILURES.md) + dowel/socket joints
analyze.py      thickness, disconnected islands, overhang map, warp/adhesion risk report
assemble3mf.py  lib3mf: parts as objects, paint_color per face (UPPERCASE hex), build items
slicer.py       drive C:\Program Files\Bambu Studio\bambu-studio.exe (silent on Win; file-driven)
presets/        machine (P1S) + process + filament (PLA x4 AMS) JSON, carrying our settings
```

Deliverable contract (B2B): `pipeline.run(image_or_mesh, palette, size) -> project.3mf`
that opens in Bambu Studio fully arranged, colored, split, supported, and sliceable.

## Open prior art in this repo

- `projects/model-grabber/export_model.py` — extracts meshes from a web 3D
  viewer (Three.js); useful test meshes.
- `Harbinger-of-Steel-v2.glb` — real test model (the one with the scarring problem).
