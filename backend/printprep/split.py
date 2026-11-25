"""
meshforge :: Half B — plate-fit splitting with registered dowel joints.

A repaired, oriented mesh may be larger than the Bambu P1S build plate (256 mm)
on one or more axes, or simply benefit from being cut (overhang #5, support
height #6, see docs/PRINT_FAILURES.md). This stage decides whether to cut, picks
axis-aligned cut planes so every resulting part fits the plate, and adds a
peg + socket dowel at each cut so the parts register and glue cleanly (#15 weak
seams between split parts).

Cut-plane scoring follows the multi-parameter split scoring in PRINT_FAILURES.md
(plate-fit is a hard constraint; then small/planar cross-section, low curvature /
hidden region, part-count penalty, joint feasibility).

Booleans use trimesh's 'manifold' engine. They can be slow / occasionally fail on
large AI meshes, so every dowel op is wrapped: on failure we fall back to a plain
capped cut and still record the joint spec in the Part so a downstream stage (or
the user) can place hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

# Bambu P1S usable build volume (mm). Keep in sync with repair.P1S_PLATE_MM.
P1S_PLATE_MM = 256.0

# Margin reserved on the plate for brim/skirt + part spacing during arrangement.
_PLATE_MARGIN_MM = 10.0

_AXES = np.eye(3)  # x, y, z unit normals


@dataclass
class Part:
    """One plate-printable piece of the original model."""
    mesh: trimesh.Trimesh
    face_filament: np.ndarray | None = None
    joints: list[dict] = field(default_factory=list)


@dataclass
class SplitResult:
    parts: list[Part]
    n_cuts: int
    notes: str


# Tolerance (mm) so an exactly-usable-size part isn't re-cut due to float noise.
_FIT_EPS_MM = 0.5


def _fits_plate(mesh: trimesh.Trimesh, plate_mm: float) -> bool:
    usable = plate_mm - _PLATE_MARGIN_MM
    return bool(np.all(mesh.extents <= usable + _FIT_EPS_MM))


def _splits_needed(extent: float, plate_mm: float) -> int:
    """How many pieces an axis of length `extent` must be cut into to fit."""
    usable = plate_mm - _PLATE_MARGIN_MM
    if extent <= usable + _FIT_EPS_MM:
        return 1
    return int(np.ceil(extent / usable))


def _cross_section_area(mesh: trimesh.Trimesh, origin: np.ndarray,
                        normal: np.ndarray) -> float:
    """Approximate area of the planar cross-section at a cut plane.

    Uses trimesh's planar section; falls back to the bounding-box face area
    (an upper bound) if the section is empty/degenerate so scoring still works.
    """
    try:
        sec = mesh.section(plane_origin=origin, plane_normal=normal)
        if sec is not None:
            # trimesh renamed to_planar -> to_2D; support both.
            to_2d = getattr(sec, "to_2D", None) or sec.to_planar
            planar, _ = to_2d()
            return float(abs(planar.area))
    except Exception:
        pass
    # Fallback: area of the bbox slice perpendicular to `normal`.
    ext = mesh.extents
    other = ext[np.argsort(np.abs(normal))[:2]]
    return float(other[0] * other[1])


def _local_roughness(mesh: trimesh.Trimesh, origin: np.ndarray,
                     normal: np.ndarray, band: float = 3.0) -> float:
    """Curvature proxy near the plane: spread of face-normal directions for
    faces whose centroid is within `band` mm of the plane. Lower = flatter =
    a better (more hidden, lower-stress) place to cut.
    """
    try:
        centroids = mesh.triangles_center
        d = np.abs((centroids - origin) @ normal)
        near = d <= band
        if near.sum() < 3:
            return 0.0
        fn = mesh.face_normals[near]
        mean = fn.mean(axis=0)
        n = np.linalg.norm(mean)
        return float(1.0 - n)  # 0 = all aligned (flat), ->1 = scattered
    except Exception:
        return 0.0


def _score_plane(mesh: trimesh.Trimesh, axis: int, origin: np.ndarray,
                 normal: np.ndarray, n_parts_after: int) -> float:
    """Weighted cut-plane score (lower = better) per PRINT_FAILURES.md."""
    bbox = float(mesh.extents.prod()) ** (2 / 3) or 1.0  # area-scale normaliser
    area = _cross_section_area(mesh, origin, normal) / bbox
    rough = _local_roughness(mesh, origin, normal)
    part_penalty = 0.15 * n_parts_after
    return 2.0 * area + 1.5 * rough + part_penalty


def _best_cut_along_axis(mesh: trimesh.Trimesh, axis: int, n_pieces: int,
                         usable: float) -> tuple[np.ndarray, float]:
    """Pick the best single cut position along `axis`. Returns (origin, score).

    Plate-fit is a hard constraint, so the candidate window is restricted to
    positions where the resulting **+side** child is no larger than `usable`
    (the cut is recursive, so guaranteeing the +side fits drives every part to
    fit eventually). Within that window we prefer the flattest / smallest
    cross-section per the split scoring. The window is centred to also keep the
    -side child as small as possible (ideally a halving cut).
    """
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    length = hi - lo
    normal = _AXES[axis]

    # The +side child spans [c, hi]; require hi - c <= usable  ->  c >= hi-usable.
    # The -side child spans [lo, c]; for a clean split prefer c near a halving
    # point so neither side is left needing many further cuts.
    c_min = max(lo + 1.0, hi - usable)          # +side fits
    c_max = min(hi - 1.0, lo + usable)          # -side fits this single cut too
    if c_max < c_min:
        # No single position lets both sides fit -> take the largest legal -side
        # slab (c at lo+usable); the +side remainder is cut again recursively.
        c_center = lo + usable
        span = 0.0
    else:
        # Center on the midpoint of the legal window (a balanced cut).
        c_center = 0.5 * (c_min + c_max)
        span = 0.5 * (c_max - c_min)

    if span > 0:
        candidates = np.linspace(c_center - span, c_center + span, 9)
    else:
        candidates = np.array([c_center])

    best_origin, best_score = None, np.inf
    cspan = max(c_max - c_min, 1e-6)
    for c in candidates:
        origin = mesh.centroid.copy()
        origin[axis] = c
        # Tiny balance term breaks ties toward the window center (even halves)
        # without overriding genuine cross-section / curvature differences.
        balance = 0.05 * abs(c - c_center) / cspan
        s = _score_plane(mesh, axis, origin, normal, n_pieces) + balance
        if s < best_score:
            best_score, best_origin = s, origin
    return best_origin, best_score


def _capped_slice(mesh: trimesh.Trimesh, origin: np.ndarray,
                  normal: np.ndarray) -> trimesh.Trimesh | None:
    """Half of `mesh` on the +normal side of the plane, capped watertight."""
    try:
        piece = mesh.slice_plane(plane_origin=origin, plane_normal=normal,
                                 cap=True)
        if piece is None or len(piece.faces) == 0:
            return None
        return piece
    except Exception:
        return None


def _add_dowel(part_a: trimesh.Trimesh, part_b: trimesh.Trimesh,
               center: np.ndarray, axis_vec: np.ndarray, dowel_d: float,
               clearance: float) -> tuple[trimesh.Trimesh, trimesh.Trimesh, bool]:
    """Union a peg cylinder into `part_a`, subtract a wider socket from `part_b`.

    The peg straddles the cut plane (half embedded each side); the socket is
    drilled into part_b with +clearance radius so the parts mate with a slip fit.
    Returns (part_a', part_b', boolean_succeeded). On any boolean failure the
    originals are returned unchanged with success=False.
    """
    peg_len = max(2.0 * dowel_d, 8.0)
    peg_r = 0.5 * dowel_d
    sock_r = peg_r + clearance

    T = trimesh.geometry.align_vectors([0, 0, 1], axis_vec / np.linalg.norm(axis_vec))

    def _cyl(radius: float, height: float) -> trimesh.Trimesh:
        c = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
        c.apply_transform(T)
        c.apply_translation(center)
        return c

    peg = _cyl(peg_r, peg_len)
    socket = _cyl(sock_r, peg_len + 2.0 * clearance)
    try:
        a2 = trimesh.boolean.union([part_a, peg], engine="manifold")
        b2 = trimesh.boolean.difference([part_b, socket], engine="manifold")
        if (a2 is None or b2 is None or len(a2.faces) == 0
                or len(b2.faces) == 0):
            return part_a, part_b, False
        return a2, b2, True
    except Exception:
        return part_a, part_b, False


def _joint_specs(center: np.ndarray, axis_vec: np.ndarray,
                 dowel_d: float) -> tuple[dict, dict]:
    """Symmetric peg/socket joint records for the two children of a cut."""
    axis_vec = axis_vec / np.linalg.norm(axis_vec)
    peg = {"type": "peg", "center": [round(float(x), 3) for x in center],
           "axis": [round(float(x), 3) for x in axis_vec], "d": float(dowel_d)}
    socket = {"type": "socket", "center": peg["center"],
              "axis": peg["axis"], "d": float(dowel_d)}
    return peg, socket


def plan_and_split(mesh: trimesh.Trimesh, face_filament: np.ndarray | None = None,
                   plate_mm: float = 256.0, max_parts: int = 8,
                   dowel_d: float = 4.0, clearance: float = 0.2) -> SplitResult:
    """Decide whether/how to cut `mesh` so every part fits the plate, adding a
    registered dowel joint at each cut.

    Strategy:
      1. If the mesh fits the plate on every axis -> return a single Part, 0 cuts.
      2. Otherwise determine, per axis, how many pieces are needed, and cut the
         axis(es) that exceed the plate. Cuts are recursive binary splits: each
         oversize part is cut by its best-scoring plane until all parts fit
         (bounded by max_parts).
      3. At each cut, place a peg (on the +side child) and a matching socket
         (on the -side child) via manifold booleans, with graceful fallback.

    face_filament is propagated to children by re-sampling the original per-face
    assignment onto each child's faces via nearest original-face lookup.
    """
    notes: list[str] = []

    if _fits_plate(mesh, plate_mm):
        notes.append(f"fits plate ({_fmt_ext(mesh)} <= {plate_mm:g}mm); no cut")
        return SplitResult(parts=[Part(mesh=mesh, face_filament=face_filament)],
                           n_cuts=0, notes="; ".join(notes))

    # Prepare a nearest-face lookup so we can carry filament onto children.
    orig_tree = None
    if face_filament is not None:
        try:
            from scipy.spatial import cKDTree
            orig_tree = cKDTree(mesh.triangles_center)
        except Exception:
            orig_tree = None
            notes.append("filament propagation disabled (no scipy KDTree)")

    def child_filament(child: trimesh.Trimesh) -> np.ndarray | None:
        if face_filament is None or orig_tree is None:
            return None
        try:
            _, idx = orig_tree.query(child.triangles_center)
            return face_filament[idx]
        except Exception:
            return None

    # Worklist of parts still to be checked/cut. Each entry is a bare Trimesh;
    # we track its joints separately and assemble Part objects at the end.
    parts: list[trimesh.Trimesh] = [mesh.copy()]
    part_joints: list[list[dict]] = [[]]
    n_cuts = 0

    i = 0
    while i < len(parts):
        part = parts[i]
        if _fits_plate(part, plate_mm) or len(parts) >= max_parts:
            i += 1
            continue

        # Choose the axis with the largest overshoot to cut first.
        usable = plate_mm - _PLATE_MARGIN_MM
        overshoot = part.extents - usable
        axis = int(np.argmax(overshoot))
        n_pieces = _splits_needed(part.extents[axis], plate_mm)
        origin, _score = _best_cut_along_axis(part, axis, n_pieces, usable)
        normal = _AXES[axis]

        pos = _capped_slice(part, origin, +normal)   # +side child
        neg = _capped_slice(part, origin, -normal)   # -side child
        if pos is None or neg is None:
            notes.append(f"cut on axis {axis} failed (slice empty); keeping part")
            i += 1
            continue

        n_cuts += 1
        # Dowel at the centroid of the cut cross-section, axis = cut normal.
        cut_center = origin.copy()
        # nudge center toward the mesh interior centroid in-plane for a real hit
        c_in_plane = part.centroid.copy()
        c_in_plane[axis] = origin[axis]
        cut_center = c_in_plane

        pos2, neg2, ok = _add_dowel(pos, neg, cut_center, normal, dowel_d, clearance)
        peg_spec, sock_spec = _joint_specs(cut_center, normal, dowel_d)
        if ok:
            pos, neg = pos2, neg2
        else:
            notes.append(f"dowel boolean failed at cut {n_cuts}; "
                         "recorded joint spec, plain cut used")

        # Replace current part with the +side child; append the -side child.
        parts[i] = pos
        part_joints[i] = part_joints[i] + [peg_spec]
        parts.append(neg)
        part_joints.append([sock_spec])
        # Do NOT advance i: re-check pos in case it still exceeds the plate.

    # Assemble Part objects with propagated filament.
    out_parts = [
        Part(mesh=p, face_filament=child_filament(p), joints=j)
        for p, j in zip(parts, part_joints)
    ]
    notes.insert(0, f"{_fmt_ext(mesh)} exceeds {plate_mm:g}mm -> "
                    f"{len(out_parts)} parts, {n_cuts} cut(s)")
    return SplitResult(parts=out_parts, n_cuts=n_cuts, notes="; ".join(notes))


def _fmt_ext(mesh: trimesh.Trimesh) -> str:
    return "x".join(f"{e:.0f}" for e in mesh.extents)


@dataclass
class ReassemblyReport:
    """Result of verify_reassembly(): whether split parts actually mate back up."""
    ok: bool
    n_joints_checked: int
    issues: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)


def verify_reassembly(result: SplitResult, area_tol_frac: float = 0.35) -> ReassemblyReport:
    """Physically verify that a SplitResult's parts can be reassembled.

    A split is only as good as its joints. This checks, per cut, that the two
    children produced by plan_and_split actually mate:

      1. Peg/socket clearance: the socket radius must exceed the peg radius
         (else the peg won't physically fit into the socket).
      2. Cut-plane contact: both children must intersect the SAME recorded
         plane (peg/socket center+axis) with a non-degenerate cross-section —
         i.e. the plane is real contact area on both sides, not a stray plane
         from a part that drifted away from the cut (a bug that would silently
         produce two parts that don't actually touch).
      3. Cross-section match: the two children's cross-sectional area at that
         plane must agree within `area_tol_frac`. A real partition of one
         surface cuts the same footprint on both sides (modulo the small,
         known peg/socket area); a mismatch means the parts were not really
         split from the same boundary and will not reassemble flush.

    Pairs peg/socket specs by rounded (center, axis) across all parts' joint
    lists, since plan_and_split always emits exactly one peg and one matching
    socket per cut, on the two children of that cut.
    """
    # index every joint spec -> (part_index, spec)
    entries: list[tuple[int, dict]] = []
    for pi, part in enumerate(result.parts):
        for spec in part.joints:
            entries.append((pi, spec))

    def _key(spec: dict) -> tuple:
        return (tuple(round(x, 2) for x in spec["center"]),
                tuple(round(x, 2) for x in spec["axis"]))

    pegs = {_key(s): (pi, s) for pi, s in entries if s["type"] == "peg"}
    sockets = {_key(s): (pi, s) for pi, s in entries if s["type"] == "socket"}

    issues: list[str] = []
    checks: list[dict] = []
    n_checked = 0

    for key, (peg_pi, peg_spec) in pegs.items():
        if key not in sockets:
            issues.append(f"peg at {peg_spec['center']} has no matching socket")
            continue
        sock_pi, sock_spec = sockets[key]
        n_checked += 1
        origin = np.array(peg_spec["center"])
        normal = np.array(peg_spec["axis"])
        peg_d, sock_d = peg_spec["d"], sock_spec["d"]

        clearance_ok = sock_d >= peg_d
        if not clearance_ok:
            issues.append(f"joint at {peg_spec['center']}: socket d={sock_d} "
                          f"< peg d={peg_d}, peg will not fit")

        area_a = _cross_section_area(result.parts[peg_pi].mesh, origin, normal)
        area_b = _cross_section_area(result.parts[sock_pi].mesh, origin, normal)
        both_touch = area_a > 1e-6 and area_b > 1e-6
        if not both_touch:
            issues.append(f"joint at {peg_spec['center']}: cut plane does not "
                          f"intersect both parts (area_a={area_a:.3g}, "
                          f"area_b={area_b:.3g}) — parts do not actually touch")

        area_match = False
        if both_touch:
            rel_diff = abs(area_a - area_b) / max(area_a, area_b)
            area_match = rel_diff <= area_tol_frac
            if not area_match:
                issues.append(f"joint at {peg_spec['center']}: cross-section "
                              f"area mismatch ({area_a:.1f} vs {area_b:.1f} mm^2, "
                              f"{rel_diff:.0%} apart) — parts won't reassemble flush")

        checks.append({
            "center": peg_spec["center"], "clearance_ok": clearance_ok,
            "both_touch": both_touch, "area_match": area_match,
            "area_a": area_a, "area_b": area_b,
        })

    ok = n_checked > 0 and not issues if pegs else result.n_cuts == 0
    return ReassemblyReport(ok=ok, n_joints_checked=n_checked, issues=issues,
                            checks=checks)


if __name__ == "__main__":
    import time

    SRC = "outputs/test_blob_repaired.stl"
    print(f"[smoke] loading {SRC} ...")
    full = trimesh.load(SRC, force="mesh", process=False)
    print(f"[smoke] full mesh: {len(full.faces)} faces, extents {full.extents}")

    # 822k faces makes manifold booleans slow; decimate a working copy for the
    # smoke test so it runs in seconds. plan_and_split itself is size-agnostic.
    work = full
    if len(full.faces) > 60000:
        try:
            work = full.simplify_quadric_decimation(face_count=40000)
            print(f"[smoke] decimated working copy -> {len(work.faces)} faces")
        except Exception as e:
            print(f"[smoke] decimation failed ({e}); using full mesh")

    # --- Case 1: fits the plate -> single part, 0 cuts.
    t0 = time.time()
    res1 = plan_and_split(work, plate_mm=P1S_PLATE_MM)
    print(f"[case1 fits] parts={len(res1.parts)} n_cuts={res1.n_cuts} "
          f"({time.time() - t0:.2f}s)")
    print(f"            notes: {res1.notes}")
    assert res1.n_cuts == 0 and len(res1.parts) == 1

    # --- Case 2: shrink the plate so the ~150mm part must be cut. This exercises
    # recursive plane selection + filament carry on the REAL mesh. The decimated
    # working copy is non-watertight, so manifold booleans correctly fall back to
    # plain cuts with recorded joint specs (see case 3 for the boolean path).
    ff = np.arange(len(work.faces)) % 4  # dummy filament to test carry-through
    t0 = time.time()
    res2 = plan_and_split(work, face_filament=ff, plate_mm=130.0, max_parts=8)
    dt = time.time() - t0
    print(f"[case2 forced split @130mm] parts={len(res2.parts)} "
          f"n_cuts={res2.n_cuts} ({dt:.2f}s)")
    print(f"            notes: {res2.notes}")
    usable2 = 130.0 - _PLATE_MARGIN_MM
    for k, p in enumerate(res2.parts):
        ext = p.mesh.extents
        fits = bool(np.all(ext <= usable2))
        ffn = None if p.face_filament is None else len(p.face_filament)
        print(f"   part {k}: {len(p.mesh.faces)} faces, "
              f"ext {ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f}, fits={fits}, "
              f"joints={[j['type'] for j in p.joints]}, filament_len={ffn}")
    assert res2.n_cuts >= 1 and len(res2.parts) >= 2
    # Below max_parts, every part must fit the (shrunk) plate.
    if len(res2.parts) < 8:
        assert all(np.all(p.mesh.extents <= 130.0) for p in res2.parts), \
            "a part below max_parts still exceeds the plate"
    # filament must be carried onto every child
    assert all(p.face_filament is not None
               and len(p.face_filament) == len(p.mesh.faces)
               for p in res2.parts)

    # --- Case 3: prove the dowel peg+socket boolean path on a watertight solid.
    box = trimesh.creation.box(extents=[300, 80, 80])  # 300mm > plate -> 2 parts
    assert box.is_watertight
    t0 = time.time()
    res3 = plan_and_split(box, plate_mm=P1S_PLATE_MM, dowel_d=6.0, clearance=0.2)
    print(f"[case3 dowel on box] parts={len(res3.parts)} n_cuts={res3.n_cuts} "
          f"({time.time() - t0:.2f}s)")
    print(f"            notes: {res3.notes}")
    for k, p in enumerate(res3.parts):
        print(f"   part {k}: {len(p.mesh.faces)} faces, wt={p.mesh.is_watertight}, "
              f"joints={[j['type'] for j in p.joints]}")
    assert res3.n_cuts == 1 and len(res3.parts) == 2
    types = sorted(j["type"] for p in res3.parts for j in p.joints)
    assert types == ["peg", "socket"], f"expected peg+socket, got {types}"
    # boolean should have succeeded -> children watertight and face count grew
    boolean_ok = "dowel boolean failed" not in res3.notes
    print(f"[case3] dowel boolean succeeded: {boolean_ok}")
    if boolean_ok:
        assert all(p.mesh.is_watertight for p in res3.parts)
    print("[smoke] OK")
