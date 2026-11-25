"""
meshforge :: Half B — mesh repair + scale to real millimeters.

First stage of print-prep. AI meshes (Hunyuan3D etc.) come out:
  - non-watertight (holes, non-manifold edges, flipped normals)
  - normalized to a ~[-1,1] unit box (no real-world size)
Slicers need a watertight, correctly-scaled solid or supports/infill misbehave.

This module loads a .glb/.obj/.stl, repairs it as far as trimesh allows, scales
it to a target real size, and reports watertightness before/after so the pipeline
can decide whether a heavier remesh is warranted.

Vertex colors (if present, e.g. from a textured Half-A run) are preserved through
load + scale for the downstream color.py stage.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import trimesh

# Bambu P1S usable build volume (mm). Leave margin for brim/skirt + part spacing.
P1S_PLATE_MM = 256.0


@dataclass
class RepairReport:
    source: str
    watertight_before: bool
    watertight_after: bool
    verts: int
    faces: int
    holes_filled: bool
    extents_mm: tuple[float, float, float]
    scaled_by: float
    exceeds_plate: bool        # any axis > plate -> splitting will be required
    has_vertex_color: bool

    def as_dict(self) -> dict:
        return asdict(self)


def load_solid(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file into a single Trimesh, concatenating scene parts."""
    obj = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(obj, trimesh.Scene):
        obj = obj.to_geometry()
    if not isinstance(obj, trimesh.Trimesh):
        raise ValueError(f"Could not load a mesh from {path!r}")
    return obj


def _fan_fill(m: trimesh.Trimesh) -> None:
    """Close ragged/pinched hole loops that trimesh.fill_holes skips: walk the
    directed boundary edges into simple loops, cap each with a centroid fan."""
    f = m.faces
    ed = {(int(a), int(b)) for a, b in np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])}
    out: dict[int, list[int]] = {}
    for a, b in ed:
        if (b, a) not in ed:
            out.setdefault(a, []).append(b)
    loops = []
    while out:
        a = next(iter(out)); path = [a]
        while True:
            nxt = out[path[-1]].pop()
            if not out[path[-1]]:
                del out[path[-1]]
            if nxt in path:  # pinch: peel off the simple sub-loop
                i = path.index(nxt); loops.append(path[i:]); path = path[:i + 1]
                if len(path) == 1:
                    break
            else:
                path.append(nxt)
            if path[-1] not in out:
                break
    verts, new = [m.vertices], []
    n = len(m.vertices)
    for lp in loops:
        if len(lp) < 3:
            continue
        verts.append(m.vertices[lp].mean(axis=0, keepdims=True))
        for i, a in enumerate(lp):
            new.append([lp[(i + 1) % len(lp)], a, n])
        n += 1
    if new:
        m.vertices = np.vstack(verts)
        m.faces = np.vstack([m.faces, new])


def repair(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, bool]:
    """Best-effort watertight repair. Returns (mesh, holes_were_filled)."""
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()

    # consistent outward winding + normals
    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_inversion(m)
    trimesh.repair.fix_normals(m)

    holes_filled = False
    if not m.is_watertight:
        # trimesh.fill_holes can create 3/4-face edges on ragged holes; keep its
        # result only if it closes the mesh, else cap the loops with a fan.
        trial = m.copy()
        trimesh.repair.fill_holes(trial)
        if not trial.is_watertight:
            trial = m.copy()
            _fan_fill(trial)
            trial.merge_vertices()
            trimesh.repair.fix_normals(trial)
        m = trial
        holes_filled = bool(m.is_watertight)
    return m, holes_filled


def scale_to_size(mesh: trimesh.Trimesh, *, target_max_mm: float | None = None,
                  target_height_mm: float | None = None) -> float:
    """Scale mesh in place to a real size. Returns the scale factor applied.

    target_height_mm wins if given (Z is print height); else target_max_mm sets the
    largest dimension. Default 150 mm longest-axis if neither supplied.
    """
    extents = mesh.extents  # (x, y, z) current size
    if target_height_mm is not None:
        factor = target_height_mm / extents[2]
    else:
        factor = (target_max_mm or 150.0) / float(extents.max())
    mesh.apply_scale(factor)
    return factor


def process(path: str | Path, *, target_max_mm: float | None = None,
            target_height_mm: float | None = None,
            out_path: str | Path | None = None) -> tuple[trimesh.Trimesh, RepairReport]:
    raw = load_solid(path)
    wt_before = bool(raw.is_watertight)
    has_color = (raw.visual is not None
                 and getattr(raw.visual, "kind", None) == "vertex")

    fixed, holes_filled = repair(raw)
    factor = scale_to_size(fixed, target_max_mm=target_max_mm,
                           target_height_mm=target_height_mm)
    ext = tuple(round(float(v), 2) for v in fixed.extents)

    report = RepairReport(
        source=str(path),
        watertight_before=wt_before,
        watertight_after=bool(fixed.is_watertight),
        verts=len(fixed.vertices),
        faces=len(fixed.faces),
        holes_filled=holes_filled,
        extents_mm=ext,  # type: ignore[arg-type]
        scaled_by=round(factor, 6),
        exceeds_plate=any(e > P1S_PLATE_MM for e in ext),
        has_vertex_color=has_color,
    )
    if out_path:
        fixed.export(str(out_path))
    return fixed, report


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="meshforge repair + scale")
    ap.add_argument("mesh")
    ap.add_argument("--max-mm", type=float, help="target longest-axis size (mm)")
    ap.add_argument("--height-mm", type=float, help="target Z height (mm)")
    ap.add_argument("--out", help="output mesh path (e.g. repaired.stl)")
    args = ap.parse_args()

    _, rep = process(args.mesh, target_max_mm=args.max_mm,
                     target_height_mm=args.height_mm, out_path=args.out)
    print(json.dumps(rep.as_dict(), indent=2))
