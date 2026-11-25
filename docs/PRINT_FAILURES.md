# Print-failure taxonomy → mitigation → pipeline stage

The point of meshforge Half B is to *design out* print failures before slicing, so the
emitted `.3mf` prints clean the first time. This is the master list of FDM failure
modes, their root causes, and **which pipeline stage owns the fix**. Sources: Bambu Lab
wiki, Obico troubleshooting guides, Prusa/Orca docs, FDM troubleshooting literature.

Key principle from the research: **spaghetti is always a *secondary* failure** — the
part detached or a support collapsed first. So fixing adhesion + supports + orientation
eliminates spaghetti for free. We target root causes, not symptoms.

Stages: `repair` · `color` · `orient` · `split` · `analyze` · `assemble3mf`/presets.

| # | Failure | Root cause | Mitigation | Stage |
|---|---------|-----------|------------|-------|
| 1 | **Support scarring** (your original problem) | Supports touch visible/curved surfaces; large contact area | Orient to push support contact onto flat/hidden faces; minimize total support-contact area; prefer "support on plate only" where possible; tune interface gap | `orient`, presets |
| 2 | **Warping / corner lift** | Differential cooling shrink, esp. large flat footprints | Orient to reduce footprint thermal stress; recommend brim/raft; chamfer/`elephant-foot` compensation in preset; enclosure note for ABS (N/A for PLA) | `orient`, presets, `analyze` |
| 3 | **Poor first-layer adhesion** | Bed too far/cold, small contact patch, dirty plate | Orient for a stable flat base; auto-add brim when footprint small/tall; first-layer temp/flow in preset | `orient`, presets |
| 4 | **Spaghetti** (secondary) | Detachment from #2/#3 or support collapse #6 | Fixed indirectly by 1–3, 6; `analyze` flags high-risk tall/narrow parts | (indirect) `analyze` |
| 5 | **Overhang sag / rough underside** | Surfaces beyond ~45° from vertical unsupported | Orient to minimize steep-overhang area; add supports only where overhang angle exceeds threshold; split to convert overhangs into separate flat-printable faces | `orient`, `split` |
| 6 | **Support failure (buckle/peel)** | Tall thin supports, weak base | Minimize support height via orientation; split tall models so each part needs short supports; tree-vs-normal choice in preset | `orient`, `split`, presets |
| 7 | **Bridging droop** | Long unsupported horizontal spans | Orient so spans shorten/align; bridge-flow + bridge-speed in preset; supports if span > limit | `orient`, presets |
| 8 | **Stringing / oozing** | Travel over gaps, temp high / retraction low | Mostly preset (retraction, temp, travel); `split` reduces cross-part travel; wipe/Z-hop in preset | presets, `split` |
| 9 | **Layer shifting** | Nozzle catches warped/curled layer; mechanical | Reduce warp (#2); `analyze` flags tall slender features; speed/accel in preset | `orient`, presets, `analyze` |
| 10 | **Doesn't fit build plate** | Model larger than 256³ mm | `split` into plate-fitting parts; tighter cuts where geometry allows | `split` |
| 11 | **Thin walls / unprintable detail** | Features below nozzle/line width (~0.4 mm) | `analyze` detects sub-threshold thickness; flag or auto-thicken; warn user | `analyze`, `repair` |
| 12 | **Non-manifold / non-watertight** | AI mesh holes, flipped normals, dup verts | `repair`: merge verts, fix winding/normals, fill holes; heavier remesh fallback | `repair` |
| 13 | **Floating islands / disconnected bits** | Generation artifacts | `analyze` finds disconnected components; drop tiny ones or connect | `repair`, `analyze` |
| 14 | **Color bleed / wrong filament regions** | Bad face→filament mapping, tiny color islands | `color`: quantize to AMS palette, clean small islands, snap boundaries to seams | `color` |
| 15 | **Weak seams between split parts** | Glue-only joints, no registration | `split`: dowel/peg + socket joints with print-clearance; align cuts to low-stress, hidden regions | `split` |
| 16 | **Excessive support waste / purge** | Poor orientation, many color changes | `orient` (less support), `color` (fewer transitions / group same-color faces), `split` to isolate color regions per part | `orient`, `color`, `split` |

## Multi-parameter split scoring (drives `split.py`)

A cut plane / part decomposition is scored on a weighted sum (lower = better):

- **Plate-fit**: every resulting part ≤ build volume (hard constraint).
- **Overhang reduction**: parts orientable so steep-overhang area drops.
- **Support reduction**: total predicted support volume across parts.
- **Cut quality**: cross-section area small, planar, in a low-curvature / hidden region.
- **Color coherence**: cut along filament-region boundaries → fewer transitions per part.
- **Stress/strength**: avoid cutting thin necks; joint placed where load is low.
- **Seam concealment**: cut hidden from primary viewing angle.
- **Part count**: penalize over-fragmentation (assembly burden).
- **Joint feasibility**: room for a dowel of min diameter at the cut.

## Notes / open issues flagged during research

- **Bambu CLI is GUI-subsystem on Windows** → no console stdout; drive it by output
  files + exit code, not by parsing stdout. (Verified.)
- **Slicer CLI needs a Bambu-flavored `.3mf` with embedded presets** to produce a
  fleshed-out result; bare STL falls back to defaults. So `assemble3mf` must embed
  presets, not just geometry.
- **`paint_color` hex must be UPPERCASE A–F** for Bambu's parser.
- **Watertight ≠ printable**: still need thickness + island checks (#11, #13).
- **Color requires Half A `--texture`** (Hunyuan3D-Paint); shape-only has no color.
