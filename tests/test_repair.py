"""
meshforge :: tests for printprep.repair.

Repair (watertight fix + scale-to-mm) is the first pipeline stage; every
downstream stage (color/orient/split/analyze/assemble3mf) assumes its output
is watertight and correctly scaled. Uses synthetic trimesh primitives so the
math is exact and deterministic, no GPU/notebook/mesh assets needed.
"""
from __future__ import annotations

import trimesh

from backend.printprep.repair import (
    RepairReport,
    P1S_PLATE_MM,
    process,
    repair,
    scale_to_size,
)


def test_scale_to_size_targets_longest_axis_by_default():
    box = trimesh.creation.box(extents=[10, 20, 40])
    factor = scale_to_size(box, target_max_mm=150.0)

    # longest axis (40) should now be 150mm; factor = 150/40 = 3.75
    assert abs(factor - 3.75) < 1e-9
    assert abs(box.extents.max() - 150.0) < 1e-6


def test_scale_to_size_targets_height_when_given():
    box = trimesh.creation.box(extents=[10, 20, 40])
    factor = scale_to_size(box, target_height_mm=80.0)

    # Z extent (40) should now be 80mm regardless of the other axes.
    assert abs(factor - 2.0) < 1e-9
    assert abs(box.extents[2] - 80.0) < 1e-6


def test_repair_keeps_a_clean_watertight_box_watertight():
    box = trimesh.creation.box(extents=[50, 50, 50])
    assert box.is_watertight

    fixed, holes_filled = repair(box)
    assert fixed.is_watertight
    assert holes_filled is False  # nothing needed fixing


def test_process_end_to_end_reports_exceeds_plate(tmp_path):
    # 300mm > P1S_PLATE_MM (256) on the longest axis, and target_max_mm keeps
    # it there, so this box must split (exceeds_plate=True downstream trigger).
    box = trimesh.creation.box(extents=[300, 40, 40])
    src = tmp_path / "big_box.stl"
    box.export(str(src))

    fixed, report = process(str(src), target_max_mm=300.0)

    assert isinstance(report, RepairReport)
    # STL round-trip can report watertight_before=False for a clean box (no
    # winding info in plain STL); repair() must still recover watertightness.
    assert report.watertight_after is True
    assert report.exceeds_plate is True
    assert max(report.extents_mm) > P1S_PLATE_MM
    assert report.verts > 0 and report.faces > 0


def test_process_end_to_end_small_box_fits_plate(tmp_path):
    box = trimesh.creation.box(extents=[10, 10, 10])
    src = tmp_path / "small_box.stl"
    box.export(str(src))

    fixed, report = process(str(src), target_max_mm=100.0)

    assert report.exceeds_plate is False
    assert abs(max(report.extents_mm) - 100.0) < 1e-3


def test_repair_closes_ragged_holes():
    # 10% of faces removed leaves pinched loops that trimesh.fill_holes alone
    # turns into non-manifold edges; the fan fallback must still close it.
    import numpy as np
    rng = np.random.default_rng(0)
    s = trimesh.creation.icosphere(3)
    bad = trimesh.Trimesh(s.vertices, s.faces[rng.random(len(s.faces)) > 0.10], process=False)
    fixed, _ = repair(bad)
    assert fixed.is_watertight and fixed.is_winding_consistent
