"""
meshforge :: Half B — assemble parts into a Bambu-flavored .3mf.

Final geometry stage before slicing. Takes the (possibly split) parts coming out
of split.py, lays them out on the P1S plate, and writes a single .3mf that a
Bambu/Orca slicer can open and per-filament-paint.

Color strategy
--------------
Bambu/Orca paint color is *normally* carried in a proprietary
`Metadata/model_settings.config` + `mmu_segmentation` blob that lib3mf does not
author. What lib3mf *does* author cleanly is the open 3MF Materials extension:
a ``ColorGroup`` of palette colors plus per-triangle ``SetTriangleProperties``
mapping every face to a palette index. We write exactly that:

  * one ColorGroup holding the (uppercase-hex) palette,
  * each Part.face_filament index -> the matching ColorGroup property id,
  * applied per triangle via SetTriangleProperties.

This makes the color regions visible/selectable in the slicer. Bambu reads the
materials-extension color groups and surfaces them for filament assignment, but
it will NOT auto-bind them to AMS slots the way its private paint_color blob
does. So we ALSO keep every Part as a *separate mesh object* — that is the
robust fallback: a multi-object .3mf lets the user assign a filament per object
in one click even if the per-triangle group is ignored. Both are emitted; see
the ``notes`` on the returned dataclass / the smoke test for what was written.

UPPERCASE hex is preserved in object names / metadata so downstream tooling and
humans can read the intended palette slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import lib3mf


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    """'#RRGGBB' (or '#RRGGBBAA') -> (r, g, b, a) bytes. Alpha defaults to 255."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        h = h + "FF"
    if len(h) != 8:
        raise ValueError(f"bad hex color {hex_color!r}")
    r, g, b, a = (int(h[i:i + 2], 16) for i in (0, 2, 4, 6))
    return r, g, b, a


def _identity_transform(wrapper: lib3mf.Wrapper, dx: float, dy: float, dz: float) -> lib3mf.Transform:
    """A 3x4 (basis + translation) transform that is identity rotation plus a shift.

    lib3mf.Transform.Fields is a [4][3] float array: four columns, the first
    three are the basis vectors (identity) and the fourth is the translation.
    """
    t = wrapper.GetIdentityTransform()
    t.Fields[3][0] = float(dx)
    t.Fields[3][1] = float(dy)
    t.Fields[3][2] = float(dz)
    return t


@dataclass
class AssembleResult:
    """What build_3mf produced (smoke-test / pipeline introspection aid)."""
    out_path: str
    n_objects: int
    n_triangles: int
    palette: list[str]
    colored: bool          # True if per-triangle color groups were written
    verified: bool         # re-opened and counts matched
    notes: str = ""
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _grid_layout(n: int, sizes: list[np.ndarray], spacing: float = 10.0) -> list[tuple[float, float]]:
    """Place n parts in a near-square row/grid; return (dx, dy) centroid offsets.

    Each cell is sized to the largest part footprint so nothing collides; parts
    are centered in their cell. Returns plate-space XY offsets to add to a part
    whose own footprint is centered on origin.
    """
    if n == 0:
        return []
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell_w = max((float(s[0]) for s in sizes), default=0.0) + spacing
    cell_h = max((float(s[1]) for s in sizes), default=0.0) + spacing

    total_w = cols * cell_w
    total_h = rows * cell_h
    offsets: list[tuple[float, float]] = []
    for i in range(n):
        c = i % cols
        r = i // cols
        # cell center, then recenter whole grid on origin
        cx = (c + 0.5) * cell_w - total_w / 2.0
        cy = (r + 0.5) * cell_h - total_h / 2.0
        offsets.append((cx, cy))
    return offsets


def build_3mf(parts, palette, out_path, *, arrange: bool = True,
              plate_mm: float = 256.0) -> str:
    """Assemble Part list into a Bambu-flavored .3mf. Returns out_path.

    parts:   list of split.Part (each has .mesh trimesh, .face_filament int array
             or None, .joints). A bare Trimesh or list of Trimesh is also accepted
             for convenience / standalone use.
    palette: list of UPPERCASE hex strings (AMS filament slots).
    out_path: destination .3mf path.

    Each part becomes its own mesh object (so per-object filament assignment is
    always possible). Per-triangle color is written via a ColorGroup when the
    part carries face_filament. Parts are arranged on the plate via build-item
    transforms when arrange=True.
    """
    out_path = str(out_path)
    palette = [str(c).upper() for c in palette]

    # Normalize input to a list of (mesh, face_filament) tuples.
    norm: list[tuple[object, np.ndarray | None]] = []
    if hasattr(parts, "vertices"):                      # a single Trimesh
        norm.append((parts, None))
    else:
        for p in parts:
            if hasattr(p, "mesh"):                      # a Part dataclass
                norm.append((p.mesh, getattr(p, "face_filament", None)))
            else:                                       # a bare Trimesh
                norm.append((p, None))

    wrapper = lib3mf.Wrapper()
    model = wrapper.CreateModel()
    model.SetUnit(lib3mf.ModelUnit.MilliMeter)

    # One shared ColorGroup for the whole palette; remember each color's PropertyID.
    color_group = model.AddColorGroup()
    prop_ids: list[int] = []
    for hex_c in palette:
        r, g, b, a = _hex_to_rgba(hex_c)
        prop_ids.append(color_group.AddColor(wrapper.RGBAToColor(r, g, b, a)))
    cg_resource_id = color_group.GetResourceID()

    mesh_objects: list[object] = []
    footprints: list[np.ndarray] = []
    any_colored = False
    total_tris = 0

    for idx, (tm, face_fil) in enumerate(norm):
        verts = np.asarray(tm.vertices, dtype=np.float64)
        faces = np.asarray(tm.faces, dtype=np.uint32)

        mo = model.AddMeshObject()
        mo.SetName(f"meshforge_part_{idx}")

        positions = []
        for v in verts:
            p = lib3mf.Position()
            p.Coordinates[0] = float(v[0])
            p.Coordinates[1] = float(v[1])
            p.Coordinates[2] = float(v[2])
            positions.append(p)
        triangles = []
        for f in faces:
            t = lib3mf.Triangle()
            t.Indices[0] = int(f[0])
            t.Indices[1] = int(f[1])
            t.Indices[2] = int(f[2])
            triangles.append(t)
        mo.SetGeometry(positions, triangles)

        # Per-triangle color via the shared ColorGroup.
        if face_fil is not None and len(face_fil) == len(faces) and len(palette) > 0:
            ff = np.asarray(face_fil, dtype=int)
            # object-level default to the most common filament (clean fallback in slicer)
            default_slot = int(np.bincount(np.clip(ff, 0, len(palette) - 1)).argmax())
            mo.SetObjectLevelProperty(cg_resource_id, prop_ids[default_slot])
            for ti in range(len(faces)):
                slot = int(ff[ti])
                slot = slot if 0 <= slot < len(palette) else 0
                pid = prop_ids[slot]
                tp = lib3mf.TriangleProperties()
                tp.ResourceID = cg_resource_id
                tp.PropertyIDs[0] = pid
                tp.PropertyIDs[1] = pid
                tp.PropertyIDs[2] = pid
                mo.SetTriangleProperties(ti, tp)
            any_colored = True

        mesh_objects.append(mo)
        footprints.append(np.asarray(tm.extents, dtype=np.float64))
        total_tris += len(faces)

    # Build items: place each object on the plate.
    if arrange and mesh_objects:
        offsets = _grid_layout(len(mesh_objects), footprints)
        for i, (mo, (dx, dy)) in enumerate(zip(mesh_objects, offsets)):
            # part centroid (XY) -> origin, then to its grid cell; lift base to z=0.
            tmi = norm[i][0]
            bmin = np.asarray(tmi.bounds[0], dtype=np.float64)
            bmax = np.asarray(tmi.bounds[1], dtype=np.float64)
            cx = (bmin[0] + bmax[0]) / 2.0
            cy = (bmin[1] + bmax[1]) / 2.0
            tx = dx - cx
            ty = dy - cy
            tz = -bmin[2]
            model.AddBuildItem(mo, _identity_transform(wrapper, tx, ty, tz))
    else:
        for mo in mesh_objects:
            model.AddBuildItem(mo, wrapper.GetIdentityTransform())

    # Stash intended palette in model metadata for humans / downstream tooling.
    meta = model.GetMetaDataGroup()
    meta.AddMetaData("https://meshforge", "palette", ",".join(palette),
                     "xs:string", False)

    writer = model.QueryWriter("3mf")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer.WriteToFile(out_path)

    notes = (
        f"Wrote {len(mesh_objects)} object(s), {total_tris} triangle(s). "
        f"Color via 3MF materials-extension ColorGroup (per-triangle "
        f"{'on' if any_colored else 'off — no face_filament supplied'}). "
        "Parts kept as separate objects for per-object filament assignment "
        "(lib3mf does not author Bambu's private paint_color blob)."
    )
    result = AssembleResult(
        out_path=out_path,
        n_objects=len(mesh_objects),
        n_triangles=total_tris,
        palette=palette,
        colored=any_colored,
        verified=False,
        notes=notes,
    )

    # Re-open to verify counts round-tripped.
    try:
        vmodel = lib3mf.Wrapper().CreateModel()
        reader = vmodel.QueryReader("3mf")
        reader.ReadFromFile(out_path)
        it = vmodel.GetMeshObjects()
        ro, rt = 0, 0
        while it.MoveNext():
            ro += 1
            rt += it.GetCurrentMeshObject().GetTriangleCount()
        result.verified = (ro == len(mesh_objects) and rt == total_tris)
        result.extras = {"reopened_objects": ro, "reopened_triangles": rt}
    except Exception as exc:  # pragma: no cover - defensive
        result.extras = {"verify_error": repr(exc)}

    # The contract returns the path; attach the rich result for callers/tests.
    build_3mf.last_result = result  # type: ignore[attr-defined]
    return out_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    import trimesh

    ROOT = Path(__file__).resolve().parents[2]
    src = ROOT / "outputs" / "test_blob_repaired.stl"
    mesh = trimesh.load(str(src), force="mesh", process=False)

    palette = ["#FFFFFF", "#1A1A1A", "#C8102E", "#005BBB"]

    # Build a tiny fake multi-part scenario WITH color to exercise the full path:
    # split the blob's faces into palette regions by face-centroid Z bands.
    class _Part:
        def __init__(self, m, ff):
            self.mesh = m
            self.face_filament = ff
            self.joints = []

    z = mesh.triangles_center[:, 2]
    bands = np.linspace(z.min(), z.max(), len(palette) + 1)
    ff = np.clip(np.digitize(z, bands[1:-1]), 0, len(palette) - 1).astype(int)

    # also make a second, plain (no-color) copy to test mixed object handling
    half = trimesh.load(str(src), force="mesh", process=False)

    parts = [_Part(mesh, ff), half]  # one colored Part + one bare Trimesh

    out = ROOT / "outputs" / "test_assemble.3mf"
    path = build_3mf(parts, palette, out)
    res = build_3mf.last_result  # type: ignore[attr-defined]

    print(json.dumps({
        "out_path": path,
        "n_objects": res.n_objects,
        "n_triangles": res.n_triangles,
        "palette": res.palette,
        "colored": res.colored,
        "verified": res.verified,
        "extras": res.extras,
        "file_bytes": Path(path).stat().st_size,
        "notes": res.notes,
    }, indent=2))
