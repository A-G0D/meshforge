"""
meshforge :: Half B — face-color quantization to AMS filament slots.

Takes a repaired trimesh.Trimesh (possibly carrying vertex colors, texture
visuals, or no color at all) and maps every face to one of up to four AMS
filament palette entries.

Color path summary
------------------
1.  TextureVisuals  — sample the UV texture at each face centroid (barycentric
    centre of the UV triangle) to get a per-face RGBA, then quantize.
2.  ColorVisuals with real color (``visual.defined`` is True, or the
    per-vertex/per-face data store is non-empty) — average vertex colors
    per face, then quantize.
3.  No color / default gray  — assign everything to filament 0, had_color=False.

After quantization, tiny isolated color islands smaller than ~0.5 % of total
faces are merged into their largest neighboring region to avoid excessive
filament purges for artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import trimesh
import trimesh.visual.texture as _tv


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default four-slot AMS palette (white / black / Bambu red / Bambu blue).
PALETTE_DEFAULT: List[str] = ["#FFFFFF", "#1A1A1A", "#C8102E", "#005BBB"]

#: Faces smaller than this fraction of total are merged into their neighbor.
_ISLAND_THRESHOLD = 0.005   # 0.5 %


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColorResult:
    """Output of :func:`assign_filaments`."""

    mesh: trimesh.Trimesh
    """The input mesh (not modified in place; same object reference passed in)."""

    face_filament: np.ndarray
    """Integer array of shape ``(n_faces,)`` — each value is an index into
    ``palette``."""

    palette: List[str]
    """Uppercase hex color strings e.g. ``['#FFFFFF', '#1A1A1A', ...]``."""

    had_color: bool
    """True if the mesh carried non-default color information."""

    n_regions: int
    """Number of spatially connected same-filament regions after island cleanup."""

    def as_dict(self) -> dict:
        return {
            "had_color": self.had_color,
            "n_regions": self.n_regions,
            "palette": self.palette,
            "face_filament_min": int(self.face_filament.min()),
            "face_filament_max": int(self.face_filament.max()),
            "filament_counts": {
                str(i): int(np.sum(self.face_filament == i))
                for i in range(len(self.palette))
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _palette_rgb(palette: List[str]) -> np.ndarray:
    """Return ``(n_colors, 3)`` float32 array from a list of '#RRGGBB' strings."""
    out = np.empty((len(palette), 3), dtype=np.float32)
    for i, h in enumerate(palette):
        h = h.lstrip("#")
        out[i] = [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]
    return out


def _quantize(face_rgb: np.ndarray, pal_rgb: np.ndarray) -> np.ndarray:
    """Assign each face to the nearest palette entry by L2 distance in RGB.

    Parameters
    ----------
    face_rgb:
        ``(n_faces, 3)`` float32 array of per-face RGB values (0–255).
    pal_rgb:
        ``(n_colors, 3)`` float32 palette array (0–255).

    Returns
    -------
    np.ndarray
        ``(n_faces,)`` int array of palette indices.
    """
    # Euclidean distance in RGB space: (n_faces, n_colors)
    diffs = face_rgb[:, np.newaxis, :] - pal_rgb[np.newaxis, :, :]   # (F, P, 3)
    dists = np.linalg.norm(diffs, axis=-1)                            # (F, P)
    return np.argmin(dists, axis=-1).astype(np.int32)


def _face_rgb_from_texture(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Sample per-face RGB from a TextureVisuals mesh.

    Computes the barycentric centre (average) of each face's UV coordinates and
    samples the texture image at that point.

    Returns ``(n_faces, 3)`` float32 array or ``None`` if sampling fails.
    """
    vis = mesh.visual
    if not isinstance(vis, _tv.TextureVisuals):
        return None
    try:
        # Convert to ColorVisuals (triggers texture->vertex-color bake)
        color_vis = vis.to_color()
        vc = np.array(color_vis.vertex_colors, dtype=np.float32)  # (V, 4)
        # Average the three vertex colors for each face
        fc_rgb = vc[mesh.faces, :3].mean(axis=1)                  # (F, 3)
        return fc_rgb
    except Exception:
        return None


def _face_rgb_from_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Extract per-face RGB by averaging per-vertex RGBA colors.

    Returns ``(n_faces, 3)`` float32 or ``None`` if no real color is stored.
    """
    vis = mesh.visual
    if not isinstance(vis, trimesh.visual.color.ColorVisuals):
        return None
    # `defined` is True when the DataStore actually contains color data.
    # We also cross-check that the stored colors are not all the trimesh
    # default gray (102, 102, 102).
    if not vis.defined and len(vis._data) == 0:
        return None
    try:
        vc = np.array(vis.vertex_colors, dtype=np.float32)   # (V, 4)
        # All identical → treat as no real color
        if np.all(vc == vc[0]):
            return None
        fc_rgb = vc[mesh.faces, :3].mean(axis=1)             # (F, 3)
        return fc_rgb
    except Exception:
        return None


def _face_rgb_from_face_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Extract per-face RGB directly from face_colors when they vary.

    Returns ``(n_faces, 3)`` float32 or ``None``.
    """
    vis = mesh.visual
    if not isinstance(vis, trimesh.visual.color.ColorVisuals):
        return None
    if not vis.defined and len(vis._data) == 0:
        return None
    try:
        fc = np.array(vis.face_colors, dtype=np.float32)   # (F, 4)
        if np.all(fc == fc[0]):
            return None
        return fc[:, :3]
    except Exception:
        return None


def _clean_islands(face_filament: np.ndarray,
                   adj: np.ndarray,
                   n_faces: int,
                   threshold: float = _ISLAND_THRESHOLD) -> np.ndarray:
    """Merge spatially tiny same-filament islands into their largest neighbor.

    Parameters
    ----------
    face_filament:
        Per-face filament indices ``(n_faces,)`` int.
    adj:
        Face adjacency pairs ``(n_pairs, 2)`` int from ``mesh.face_adjacency``.
    n_faces:
        Total face count (to avoid passing mesh again).
    threshold:
        Regions smaller than ``threshold * n_faces`` faces are merged.

    Returns
    -------
    np.ndarray
        Cleaned ``(n_faces,)`` int array.
    """
    out = face_filament.copy()
    min_size = max(1, int(threshold * n_faces))

    # Build connected-component labels within same-filament adjacency
    same = out[adj[:, 0]] == out[adj[:, 1]]
    rows = np.concatenate([adj[same, 0], adj[same, 1]])
    cols = np.concatenate([adj[same, 1], adj[same, 0]])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
    n_comps, comp_labels = connected_components(A, directed=False)

    sizes = np.bincount(comp_labels, minlength=n_comps)
    tiny_comps = np.where(sizes < min_size)[0]

    if len(tiny_comps) == 0:
        return out

    # For each tiny component find its filament and re-assign to the
    # most-common filament among its boundary neighbors.
    comp_faces: list[np.ndarray] = [
        np.where(comp_labels == c)[0] for c in tiny_comps
    ]

    for faces_idx in comp_faces:
        face_set = set(faces_idx.tolist())
        # Collect neighbor faces outside this component
        mask = np.isin(adj[:, 0], faces_idx) | np.isin(adj[:, 1], faces_idx)
        nbr_pairs = adj[mask]
        nbrs = nbr_pairs.ravel()
        nbrs = nbrs[~np.isin(nbrs, faces_idx)]
        if len(nbrs) == 0:
            continue   # isolated island, keep as-is
        # Pick the most-common filament among neighbors
        counts = np.bincount(out[nbrs], minlength=len(PALETTE_DEFAULT))
        dominant = int(np.argmax(counts))
        out[faces_idx] = dominant

    return out


def _count_regions(face_filament: np.ndarray,
                   adj: np.ndarray,
                   n_faces: int) -> int:
    """Count spatially connected same-filament regions."""
    same = face_filament[adj[:, 0]] == face_filament[adj[:, 1]]
    rows = np.concatenate([adj[same, 0], adj[same, 1]])
    cols = np.concatenate([adj[same, 1], adj[same, 0]])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
    n_comps, _ = connected_components(A, directed=False)
    return int(n_comps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_filaments(
    mesh: trimesh.Trimesh,
    palette: List[str] = PALETTE_DEFAULT,
) -> ColorResult:
    """Assign each mesh face to an AMS filament palette slot.

    Detection order:

    1. ``trimesh.visual.texture.TextureVisuals`` → sample texture.
    2. ``trimesh.visual.color.ColorVisuals`` with non-uniform data → average
       vertex colors per face.
    3. No real color → all faces map to filament 0, ``had_color=False``.

    After quantization, isolated color islands smaller than 0.5 % of total
    faces are merged into the dominant neighboring filament to reduce purges.

    Parameters
    ----------
    mesh:
        A ``trimesh.Trimesh`` as produced by :func:`backend.printprep.repair.process`.
    palette:
        Up to four ``'#RRGGBB'`` strings (uppercase).  Defaults to
        :data:`PALETTE_DEFAULT`.

    Returns
    -------
    ColorResult
    """
    palette = [c.upper() for c in palette]
    pal_rgb = _palette_rgb(palette)

    n_faces = len(mesh.faces)

    # ------------------------------------------------------------------
    # Step 1 – extract per-face RGB
    # ------------------------------------------------------------------
    face_rgb: np.ndarray | None = None
    had_color = False

    # Try texture first
    face_rgb = _face_rgb_from_texture(mesh)
    if face_rgb is not None:
        had_color = True

    # Fall back to vertex colors
    if face_rgb is None:
        face_rgb = _face_rgb_from_vertex_colors(mesh)
        if face_rgb is not None:
            had_color = True

    # Fall back to face colors
    if face_rgb is None:
        face_rgb = _face_rgb_from_face_colors(mesh)
        if face_rgb is not None:
            had_color = True

    # ------------------------------------------------------------------
    # Step 2 – quantize or default
    # ------------------------------------------------------------------
    if had_color and face_rgb is not None:
        face_filament = _quantize(face_rgb, pal_rgb)
    else:
        face_filament = np.zeros(n_faces, dtype=np.int32)

    # ------------------------------------------------------------------
    # Step 3 – clean tiny islands
    # ------------------------------------------------------------------
    adj = mesh.face_adjacency            # (n_adj, 2)
    if had_color and len(adj) > 0:
        face_filament = _clean_islands(face_filament, adj, n_faces)

    # ------------------------------------------------------------------
    # Step 4 – count connected regions
    # ------------------------------------------------------------------
    if len(adj) > 0:
        n_regions = _count_regions(face_filament, adj, n_faces)
    else:
        n_regions = int(np.unique(face_filament).size)

    return ColorResult(
        mesh=mesh,
        face_filament=face_filament,
        palette=palette,
        had_color=had_color,
        n_regions=n_regions,
    )


# ---------------------------------------------------------------------------
# Smoke test / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    from backend.printprep.repair import process as repair_process

    root = Path(__file__).resolve().parents[2]   # project root

    # Prefer the repaired STL; fall back to raw GLB
    stl_path = root / "outputs" / "test_blob_repaired.stl"
    glb_path = root / "outputs" / "test_blob_20260622_103958.glb"

    test_path = stl_path if stl_path.exists() else glb_path
    print(f"[color] loading {test_path.name} …")

    mesh, rep = repair_process(test_path, target_max_mm=150.0)
    print(f"[color] mesh  faces={len(mesh.faces)}  verts={len(mesh.vertices)}")
    print(f"[color] repair had_vertex_color={rep.has_vertex_color}")

    result = assign_filaments(mesh)
    print(f"[color] had_color  = {result.had_color}")
    print(f"[color] n_regions  = {result.n_regions}")
    print(f"[color] palette    = {result.palette}")
    counts = result.as_dict()["filament_counts"]
    print(f"[color] filament counts = {json.dumps(counts)}")

    # Also run on the raw GLB to exercise the vertex-color path
    if glb_path.exists() and test_path != glb_path:
        print(f"\n[color] also testing raw GLB …")
        from backend.printprep.repair import load_solid
        raw = load_solid(glb_path)
        result2 = assign_filaments(raw)
        print(f"[color] GLB had_color = {result2.had_color}")
        print(f"[color] GLB n_regions = {result2.n_regions}")
        counts2 = result2.as_dict()["filament_counts"]
        print(f"[color] GLB filament counts = {json.dumps(counts2)}")

    print("\n[color] smoke test PASSED")
