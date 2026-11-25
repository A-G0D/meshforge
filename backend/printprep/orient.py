"""
meshforge :: Half B — print-orientation optimizer.

Third stage of print-prep. A repaired, watertight mesh (units = mm) can be set
on the plate in infinitely many poses; the pose decides where supports land,
how big the footprint is, and how much steep-overhang area sags. This module
searches a set of candidate orientations and picks the one that minimizes the
print pathologies that orientation actually controls.

Why this matters (see docs/PRINT_FAILURES.md):
  - #1 Support scarring — the user's original problem. Supports touch downward /
    steep-overhang faces; orienting those faces away from -Z shrinks the scarred
    contact area.
  - #2 Warping / #3 adhesion — a stable, reasonably large flat footprint resists
    corner-lift and first-layer peel.
  - #5 Overhang sag — faces beyond ~45deg from vertical print rough; we count and
    minimize their area.

Strategy: generate candidate "down" directions from the convex-hull face normals
(laying each hull facet flat on the plate — the classic set of stable resting
poses) plus the 6 axis-aligned poses, score each, and return the best applied to
a translated copy whose base sits at z = 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

# Faces whose outward normal points more than this far below horizontal need
# support (overhang steeper than 45deg from vertical). Cos of angle-from-down.
# A face needs support when its normal has a downward component and the surface
# is steeper than 45deg from vertical, i.e. normal_z < -cos(45deg).
_OVERHANG_COS = float(np.cos(np.deg2rad(45.0)))  # 0.7071...

# Scoring weights (lower score = better). support contact is the dominant term
# because support scarring is the headline failure we are fighting.
_W_SUPPORT = 1.0       # area of faces needing support (the scar source)
_W_OVERHANG = 0.5      # steep-overhang area (rough underside, may sag)
_W_FOOTPRINT = 0.3     # stability term: penalize a too-small base footprint
# Print-height penalty. A pose can have minimal support area yet stand the part
# up tall-and-narrow, which per docs/PRINT_FAILURES.md is the dominant cause of
# warping (#2), poor adhesion (#3), support buckling (#6) and layer shifting (#9)
# — and the slicer outright rejects a part that pushes past the build volume.
# Without this term the optimizer happily picks a 231mm-tall pose over a stable
# 97mm-flat one. Weighted to break ties toward compact poses while still letting
# a big support-area win override it.
_W_HEIGHT = 0.5        # penalize tall poses (max_z normalized by plate height)
_DEFAULT_PLATE_MM = 256.0  # P1S build-volume edge; height is normalized by this


@dataclass
class OrientResult:
    transform: np.ndarray              # 4x4 homogeneous transform applied to input
    mesh: trimesh.Trimesh              # oriented copy, min z == 0 (base on plate)
    metrics: dict                      # support_contact_area, overhang_area, footprint_area, max_z, score
    candidates: list[dict] = field(default_factory=list)  # all scored poses


def _rotation_to_minus_z(normal: np.ndarray) -> np.ndarray:
    """4x4 rotation that maps `normal` onto -Z (so that face rests on the plate)."""
    n = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.eye(4)
    n = n / norm
    target = np.array([0.0, 0.0, -1.0])
    # trimesh handles the parallel / anti-parallel edge cases internally.
    return trimesh.geometry.align_vectors(n, target)


def _settle_shift(verts_rot: np.ndarray) -> np.ndarray:
    """Translation matrix that drops rotated verts so their min z == 0."""
    dz = -float(verts_rot[:, 2].min())
    return trimesh.transformations.translation_matrix([0.0, 0.0, dz])


def _evaluate(verts_rot: np.ndarray, normals_rot: np.ndarray,
              areas: np.ndarray) -> dict:
    """Score a candidate pose from already-rotated vertex/normal arrays.

    Works on raw numpy (no Trimesh copy) so the 24-candidate search stays cheap on
    dense meshes. `areas` are rotation-invariant so the input mesh's are reused.

    support_contact_area : area of faces needing support (steep downward overhang)
    overhang_area        : same set, reported separately for clarity / weighting
    footprint_area       : convex-hull area of vertices projected to the plate (XY)
    max_z                : print height after settling to z = 0
    """
    nz = normals_rot[:, 2]
    # Faces needing support: outward normal points downward, steeper than 45deg
    # from vertical. nz < -cos(45deg) means the face leans past the support cutoff.
    overhang_mask = nz < -_OVERHANG_COS
    overhang_area = float(areas[overhang_mask].sum())

    # Footprint: convex-hull area of all vertices projected to XY. A larger
    # footprint = more stable adhesion (we *reward* it, so it lowers the score).
    # The 2D convex hull is the right stability proxy (it bounds the base extent)
    # and is far cheaper than a 2D triangle union on a dense mesh.
    pts2d = verts_rot[:, :2]
    try:
        from scipy.spatial import ConvexHull

        footprint_area = float(ConvexHull(pts2d).volume)  # 2D "volume" == area
    except Exception:
        ext = pts2d.max(axis=0) - pts2d.min(axis=0)
        footprint_area = float(ext[0] * ext[1])

    zspan = verts_rot[:, 2]
    return {
        "support_contact_area": overhang_area,
        "overhang_area": overhang_area,
        "footprint_area": footprint_area,
        "max_z": float(zspan.max() - zspan.min()),
    }


def _score(metrics: dict, total_area: float, plate_mm: float = _DEFAULT_PLATE_MM) -> float:
    """Weighted score, lower = better. Areas normalized by total surface area so
    the weights are scale-independent. Footprint is rewarded (subtracted) but
    bounded so a giant flat base can't dominate. Print height is penalized
    (normalized by plate height) so a low-support pose can't win by standing the
    part up tall-and-narrow."""
    if total_area <= 0:
        total_area = 1.0
    if plate_mm <= 0:
        plate_mm = _DEFAULT_PLATE_MM
    support = metrics["support_contact_area"] / total_area
    overhang = metrics["overhang_area"] / total_area
    # Footprint reward: fraction of total area; clamp to [0,1] so it acts as a
    # gentle stability bonus rather than overwhelming the support term.
    foot = min(metrics["footprint_area"] / total_area, 1.0)
    # Height penalty: a part standing taller than the plate is unprintable, so
    # this rises sharply past 1.0; below that it is a gentle compactness pressure.
    height = metrics["max_z"] / plate_mm
    return (_W_SUPPORT * support + _W_OVERHANG * overhang
            - _W_FOOTPRINT * foot + _W_HEIGHT * height)


def _candidate_normals(mesh: trimesh.Trimesh, n_candidates: int) -> np.ndarray:
    """Build the set of candidate 'down' directions: the 6 axis poses plus the
    largest convex-hull facet normals (each lays a stable resting facet flat)."""
    axes = np.array([
        [0, 0, 1], [0, 0, -1],
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0],
    ], dtype=float)

    try:
        hull = mesh.convex_hull
        hn = hull.face_normals
        ha = hull.area_faces
        # Largest hull facets = most stable resting poses; take the biggest ones.
        order = np.argsort(ha)[::-1]
        hn = hn[order]
    except Exception:
        hn = np.empty((0, 3))

    cand = np.vstack([axes, hn]) if len(hn) else axes

    # Deduplicate near-identical directions (hull facets often cluster).
    kept: list[np.ndarray] = []
    for v in cand:
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            continue
        u = v / nv
        if all(np.dot(u, k) < 0.999 for k in kept):
            kept.append(u)
        if len(kept) >= n_candidates:
            break
    return np.array(kept)


def optimize(mesh: trimesh.Trimesh, n_candidates: int = 24, *,
             plate_mm: float = _DEFAULT_PLATE_MM) -> OrientResult:
    """Search orientations and return the lowest-scoring (best) pose applied.

    The returned mesh is a copy rotated into the chosen pose and translated so its
    base sits at z = 0. `transform` is the full 4x4 that takes the input mesh into
    that pose (rotation then settle-to-plate translation).
    """
    total_area = float(mesh.area) or 1.0
    normals = _candidate_normals(mesh, n_candidates)

    # Read raw arrays once; score every candidate on rotated copies of these
    # (numpy only) instead of copying the whole Trimesh per candidate — keeps
    # the search cheap and memory-flat on dense meshes (800k+ faces).
    verts = np.asarray(mesh.vertices, dtype=float)
    face_normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)

    candidates: list[dict] = []
    best = None  # (score, full_transform, metrics)

    for nrm in normals:
        rot = _rotation_to_minus_z(nrm)
        R = rot[:3, :3]
        verts_rot = verts @ R.T
        normals_rot = face_normals @ R.T
        shift = _settle_shift(verts_rot)
        full = shift @ rot
        verts_rot = verts_rot + shift[:3, 3]  # apply the z drop for metrics

        metrics = _evaluate(verts_rot, normals_rot, areas)
        s = _score(metrics, total_area, plate_mm)
        metrics["score"] = s
        candidates.append({
            "down_normal": [round(float(x), 4) for x in nrm],
            "support_contact_area": round(metrics["support_contact_area"], 3),
            "overhang_area": round(metrics["overhang_area"], 3),
            "footprint_area": round(metrics["footprint_area"], 3),
            "max_z": round(metrics["max_z"], 3),
            "score": round(s, 6),
        })
        if best is None or s < best[0]:
            best = (s, full, metrics)

    assert best is not None  # _candidate_normals always yields the 6 axes
    score, full, metrics = best
    metrics["score"] = score
    candidates.sort(key=lambda c: c["score"])

    # Build the oriented mesh ONCE, only for the winning pose.
    oriented = mesh.copy()
    oriented.apply_transform(full)
    # Defensively re-zero the base against float drift from the matrix product.
    oriented.apply_transform(
        trimesh.transformations.translation_matrix([0.0, 0.0, -float(oriented.bounds[0][2])])
    )

    return OrientResult(
        transform=full,
        mesh=oriented,
        metrics=metrics,
        candidates=candidates,
    )


if __name__ == "__main__":
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    candidates_paths = [
        root / "outputs" / "test_blob_repaired.stl",
        root / "outputs" / "test_blob_20260622_103958.glb",
    ]
    mesh_path = next((p for p in candidates_paths if p.exists()), None)
    if mesh_path is None:
        raise SystemExit("No test mesh found in outputs/")

    print(f"loading {mesh_path.name} ...")
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    print(f"  faces={len(m.faces)}  extents_mm={tuple(round(float(e),1) for e in m.extents)}  area={m.area:.1f}")

    res = optimize(m, n_candidates=24)

    print(f"\nsearched {len(res.candidates)} candidate orientations")
    print("best pose metrics:")
    print(json.dumps({k: round(float(v), 3) for k, v in res.metrics.items()}, indent=2))

    # Sanity: base settled on plate.
    base_z = float(res.mesh.bounds[0][2])
    print(f"\noriented base min-z = {base_z:.6f} mm (should be ~0)")
    print(f"oriented extents_mm = {tuple(round(float(e),1) for e in res.mesh.extents)}")

    # Show how much support contact the best pose saved vs. the worst candidate.
    worst = res.candidates[-1]
    best = res.candidates[0]
    print(f"\nsupport_contact_area  best={best['support_contact_area']}  "
          f"worst={worst['support_contact_area']}  "
          f"(reduction {100*(1 - best['support_contact_area']/max(worst['support_contact_area'],1e-9)):.1f}%)")
    print("\ntop 3 candidates:")
    for c in res.candidates[:3]:
        print(f"  down={c['down_normal']}  score={c['score']}  "
              f"support={c['support_contact_area']}  footprint={c['footprint_area']}")
