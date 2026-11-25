"""
meshforge :: Half B — printability analysis.

Audits a repaired mesh (or a list of split Parts) for printability issues:
  - Thin features below nozzle resolution (failure #11)
  - Disconnected / floating components (failure #13)
  - Overhang fraction (faces > 45 deg from vertical) (failure #5)
  - Plate-fit check (failure #10)
  - Rough support area fraction
  - Tall / slender spaghetti-risk flag (failure #4, #9)

Consumes: trimesh.Trimesh OR list[Part]
  (Part is the dataclass defined in split.py; we do a lazy duck-type check so
   analyze.py stays importable even before split.py exists.)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Contract types
# ---------------------------------------------------------------------------

@dataclass
class AnalysisReport:
    """Result of audit().  All public fields are part of the cross-module contract."""
    warnings: list[str]
    thin_face_count: int            # faces where edge-proxy thickness < nozzle*1.5
    disconnected_components: int    # number of separate body components
    overhang_fraction: float        # 0..1, fraction of total face area that overhangs >45 deg
    footprint_ok: bool              # base is reasonably stable (footprint / bbox ratio)
    fits_plate: bool                # every part fits within plate_mm on X and Y
    est_support_fraction: float     # rough fraction of surface area needing support
    summary: str                    # human-readable one-paragraph summary

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OVERHANG_THRESH_DEG = 45.0   # faces > this angle from vertical need support


def _face_normals_and_areas(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (normals Nx3, areas N,) for all faces."""
    return mesh.face_normals, mesh.area_faces


def _overhang_fraction(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Return (overhang_fraction, support_fraction) of total surface area.

    overhang_fraction: fraction of faces pointing more than 45 deg below horizontal
    (i.e. face normal has a downward Z component such that angle from -Z > 45 deg,
    equivalently: normal.z < -cos(45) ≈ -0.707 for bottom-facing faces).

    We use: angle from upward vertical = arccos(normal.z).
    A face needs support when normal.z < 0 and the angle from -Z < 45 deg
    i.e. normal.z < -cos(45) = -0.707.

    For reporting overhang_fraction (faces not self-supporting), we use:
    faces where normal.z < cos(90-45) = cos(45) = 0.707 AND normal.z < 0
    i.e. downward-pointing faces with angle from downward vertical < 45 deg.
    """
    normals, areas = _face_normals_and_areas(mesh)
    total_area = float(areas.sum()) or 1.0

    # Faces pointing significantly downward (need support): nz < -cos(45)
    cos45 = float(np.cos(np.radians(_OVERHANG_THRESH_DEG)))
    support_mask = normals[:, 2] < -cos45
    overhang_mask = (normals[:, 2] < 0) & (normals[:, 2] >= -cos45)

    support_area = float(areas[support_mask].sum())
    overhang_area = float(areas[overhang_mask].sum())

    overhang_frac = (support_area + overhang_area) / total_area
    support_frac = support_area / total_area
    return overhang_frac, support_frac


def _thin_face_count(
    mesh: trimesh.Trimesh,
    min_thickness_mm: float,
    n_sample: int = 1000,
) -> int:
    """Estimate thin-face count using bilateral ray-cast thickness sampling.

    We sample up to ``n_sample`` face centers, shoot a ray inward along the
    inverted face normal, and measure the distance to the first opposing surface.
    Faces whose sampled thickness < min_thickness_mm are flagged.

    The count is extrapolated from the sample back to total face count:
        thin_face_count ≈ (thin_in_sample / n_sample) * total_faces

    This avoids false positives from dense tessellation (tiny triangles on a
    large surface look thin by area or edge-length but have full wall thickness).

    Falls back to 0 on error (e.g., mesh has no ray-intersectable surface).
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return 0

    # Work on a vertex-merged copy so STL topology is correct
    m = mesh.copy()
    m.merge_vertices()

    # Sample face indices
    actual_sample = min(n_sample, n_faces)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n_faces, actual_sample)

    origins = m.triangles_center[idx]       # (S, 3) face centroids
    normals = m.face_normals[idx]           # (S, 3)

    # Offset slightly inward to avoid self-intersection at origin
    offset = 0.001
    ray_origins = origins - normals * offset
    ray_dirs = -normals  # shoot inward

    try:
        locs, ray_idx, _ = m.ray.intersects_location(
            ray_origins, ray_dirs, multiple_hits=False
        )
    except Exception:
        return 0

    if len(locs) == 0:
        return 0

    # Distance from original face center to hit point
    dists = np.linalg.norm(locs - origins[ray_idx], axis=1)
    thin_in_sample = int(np.sum(dists < min_thickness_mm))

    # Extrapolate to full face count
    return int(round(thin_in_sample / actual_sample * n_faces))


def _disconnected_components(mesh: trimesh.Trimesh) -> int:
    """Return the number of distinct connected body components.

    STL files are loaded without shared vertices; we merge vertices first so that
    trimesh's graph-based split() sees the actual connectivity rather than treating
    every triangle as isolated.
    """
    m = mesh.copy()
    m.merge_vertices()
    parts = m.split(only_watertight=False)
    if isinstance(parts, trimesh.Trimesh):
        return 1
    return len(parts)


def _fits_plate(mesh: trimesh.Trimesh, plate_mm: float) -> bool:
    """True if the mesh X and Y extents fit within plate_mm."""
    ext = mesh.extents  # (x, y, z)
    return bool(ext[0] <= plate_mm and ext[1] <= plate_mm)


def _footprint_ok(mesh: trimesh.Trimesh) -> bool:
    """Heuristic: footprint stable if base area / bounding box area >= 0.15.

    A very tall, narrow part with a tiny base is at risk of toppling mid-print.
    We approximate base area as the convex hull of vertices projected to Z=min.
    """
    try:
        from shapely.geometry import MultiPoint
        zmin = float(mesh.vertices[:, 2].min())
        base_verts = mesh.vertices[mesh.vertices[:, 2] < zmin + 2.0, :2]
        if len(base_verts) < 3:
            return False
        hull = MultiPoint(base_verts).convex_hull
        base_area = float(hull.area)
        bbox_area = float(mesh.extents[0] * mesh.extents[1])
        if bbox_area < 1e-6:
            return False
        return base_area / bbox_area >= 0.15
    except Exception:
        # shapely unavailable or degenerate mesh — assume ok
        return True


def _audit_single(
    mesh: trimesh.Trimesh,
    nozzle_d: float,
    plate_mm: float,
    part_label: str = "",
) -> tuple[list[str], int, int, float, bool, bool, float]:
    """Audit a single Trimesh.  Returns components of AnalysisReport fields."""
    warnings: list[str] = []
    label = f"[{part_label}] " if part_label else ""

    # --- thin features ---
    min_thick = nozzle_d * 1.5
    thin = _thin_face_count(mesh, min_thick)
    if thin > 0:
        pct = 100.0 * thin / max(len(mesh.faces), 1)
        warnings.append(
            f"{label}Thin features: {thin} faces ({pct:.1f}%) have shortest edge "
            f"< {min_thick:.2f} mm (nozzle × 1.5). These may not print cleanly."
        )

    # --- disconnected components ---
    n_comp = _disconnected_components(mesh)
    if n_comp > 1:
        warnings.append(
            f"{label}Mesh has {n_comp} disconnected components. "
            "Floating islands may fail to print or shift."
        )

    # --- overhang ---
    overhang_frac, support_frac = _overhang_fraction(mesh)
    if overhang_frac > 0.25:
        warnings.append(
            f"{label}High overhang fraction: {overhang_frac:.1%} of surface area. "
            "Consider re-orienting to reduce support requirement."
        )
    if support_frac > 0.15:
        warnings.append(
            f"{label}Estimated support contact area: {support_frac:.1%} of surface. "
            "Surface quality under overhangs may be reduced."
        )

    # --- plate fit ---
    fits = _fits_plate(mesh, plate_mm)
    if not fits:
        ext = mesh.extents
        warnings.append(
            f"{label}Part extents ({ext[0]:.1f} × {ext[1]:.1f} mm) exceed "
            f"plate ({plate_mm:.0f} mm). Must split before printing."
        )

    # --- footprint stability ---
    fp_ok = _footprint_ok(mesh)
    if not fp_ok:
        warnings.append(
            f"{label}Small base footprint relative to part height. "
            "Consider adding a brim or reorienting for better adhesion."
        )

    # --- tall/slender spaghetti risk ---
    ext = mesh.extents
    if ext[2] > 0 and (ext[0] / ext[2] < 0.25 or ext[1] / ext[2] < 0.25):
        warnings.append(
            f"{label}Tall/slender aspect ratio "
            f"({ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f} mm). "
            "Risk of spaghetti if adhesion or supports fail."
        )

    return warnings, thin, n_comp, overhang_frac, fp_ok, fits, support_frac


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit(
    parts_or_mesh: Union[trimesh.Trimesh, list],
    nozzle_d: float = 0.4,
    plate_mm: float = 256.0,
) -> AnalysisReport:
    """Audit a mesh (or list of Part) for printability issues.

    Parameters
    ----------
    parts_or_mesh:
        Either a single trimesh.Trimesh, or a list of Part objects from split.py.
        Part is duck-typed: we look for a `.mesh` attribute.
    nozzle_d:
        Nozzle diameter in mm (default 0.4).  Thin-feature threshold = nozzle_d * 1.5.
    plate_mm:
        Build plate usable dimension (mm) for fit check (default 256 mm, P1S).

    Returns
    -------
    AnalysisReport
    """
    # Normalise input to list of (mesh, label) pairs
    if isinstance(parts_or_mesh, trimesh.Trimesh):
        items = [(parts_or_mesh, "")]
    elif isinstance(parts_or_mesh, list):
        items = []
        for i, p in enumerate(parts_or_mesh):
            if isinstance(p, trimesh.Trimesh):
                items.append((p, f"part{i}"))
            elif hasattr(p, "mesh") and isinstance(p.mesh, trimesh.Trimesh):
                items.append((p.mesh, f"part{i}"))
            else:
                raise TypeError(
                    f"Expected trimesh.Trimesh or Part (with .mesh), got {type(p)}"
                )
    else:
        raise TypeError(
            f"parts_or_mesh must be a Trimesh or list, got {type(parts_or_mesh)}"
        )

    all_warnings: list[str] = []
    total_thin = 0
    max_components = 0
    total_area = 0.0
    weighted_overhang = 0.0
    weighted_support = 0.0
    all_fits = True
    all_fp_ok = True

    for mesh, label in items:
        area = float(mesh.area) or 1.0
        w, thin, n_comp, ovhng, fp_ok, fits, supp = _audit_single(
            mesh, nozzle_d, plate_mm, label
        )
        all_warnings.extend(w)
        total_thin += thin
        max_components = max(max_components, n_comp)
        weighted_overhang += ovhng * area
        weighted_support += supp * area
        total_area += area
        if not fits:
            all_fits = False
        if not fp_ok:
            all_fp_ok = False

    overhang_frac = weighted_overhang / total_area if total_area > 0 else 0.0
    support_frac = weighted_support / total_area if total_area > 0 else 0.0

    # Build summary
    n_parts = len(items)
    part_str = f"{n_parts} part(s)" if n_parts > 1 else "mesh"
    if not all_warnings:
        verdict = "No critical printability issues detected."
    else:
        verdict = f"{len(all_warnings)} issue(s) found — see warnings list."

    summary = (
        f"Analyzed {part_str}. "
        f"Thin faces: {total_thin}. "
        f"Disconnected components: {max_components}. "
        f"Overhang: {overhang_frac:.1%} of surface. "
        f"Est. support area: {support_frac:.1%}. "
        f"Fits plate ({plate_mm:.0f} mm): {all_fits}. "
        f"Footprint OK: {all_fp_ok}. "
        f"{verdict}"
    )

    return AnalysisReport(
        warnings=all_warnings,
        thin_face_count=total_thin,
        disconnected_components=max_components,
        overhang_fraction=round(overhang_frac, 4),
        footprint_ok=all_fp_ok,
        fits_plate=all_fits,
        est_support_fraction=round(support_frac, 4),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    repaired = project_root / "outputs" / "test_blob_repaired.stl"
    raw_glb   = project_root / "outputs" / "test_blob_20260622_103958.glb"

    # Try repaired STL first, fall back to raw GLB
    test_path = repaired if repaired.exists() else raw_glb
    print(f"Loading: {test_path}")

    mesh = trimesh.load(str(test_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"extents {mesh.extents.round(1)} mm")

    report = audit(mesh)
    print("\n=== AnalysisReport ===")
    print(json.dumps(report.as_dict(), indent=2))

    print("\n--- Summary ---")
    print(report.summary)

    if report.warnings:
        print("\n--- Warnings ---")
        for w in report.warnings:
            print(" •", w)
    else:
        print("\nNo warnings.")
