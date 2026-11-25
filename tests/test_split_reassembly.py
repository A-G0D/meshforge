"""
meshforge :: tests for split.verify_reassembly.

Uses a synthetic watertight box (like split.py's own case3 smoke test) so the
joint math is exact and deterministic, no GPU/notebook/mesh assets needed.
"""
from __future__ import annotations

import copy

import trimesh

from backend.printprep.split import ReassemblyReport, plan_and_split, verify_reassembly


def test_reassembly_ok_on_clean_split():
    box = trimesh.creation.box(extents=[300, 80, 80])  # > 256mm plate -> 1 cut
    result = plan_and_split(box, plate_mm=256.0, dowel_d=6.0, clearance=0.2)

    assert result.n_cuts == 1
    assert len(result.parts) == 2

    report = verify_reassembly(result)
    assert isinstance(report, ReassemblyReport)
    assert report.n_joints_checked == 1
    assert report.issues == []
    assert report.ok is True
    check = report.checks[0]
    assert check["clearance_ok"] is True
    assert check["both_touch"] is True
    assert check["area_match"] is True
    # box cross-section at the cut is exactly 80x80 = 6400mm^2 on both sides
    assert abs(check["area_a"] - 6400.0) < 50.0
    assert abs(check["area_b"] - 6400.0) < 50.0


def test_reassembly_flags_undersized_socket():
    box = trimesh.creation.box(extents=[300, 80, 80])
    result = plan_and_split(box, plate_mm=256.0, dowel_d=6.0, clearance=0.2)

    # Corrupt the recorded socket diameter to simulate a joint that would not
    # physically receive its peg (e.g. a future regression in _joint_specs).
    broken = copy.deepcopy(result)
    for part in broken.parts:
        for spec in part.joints:
            if spec["type"] == "socket":
                spec["d"] = 1.0  # smaller than the 6.0mm peg

    report = verify_reassembly(broken)
    assert report.ok is False
    assert report.n_joints_checked == 1
    assert any("will not fit" in msg for msg in report.issues)


def test_reassembly_flags_missing_socket():
    box = trimesh.creation.box(extents=[300, 80, 80])
    result = plan_and_split(box, plate_mm=256.0, dowel_d=6.0, clearance=0.2)

    # Drop the socket spec entirely to simulate a part that lost its joint.
    broken = copy.deepcopy(result)
    for part in broken.parts:
        part.joints = [s for s in part.joints if s["type"] != "socket"]

    report = verify_reassembly(broken)
    assert report.ok is False
    assert any("no matching socket" in msg for msg in report.issues)


def test_reassembly_trivial_when_no_cuts():
    box = trimesh.creation.box(extents=[100, 80, 80])  # fits plate as-is
    result = plan_and_split(box, plate_mm=256.0)
    assert result.n_cuts == 0

    report = verify_reassembly(result)
    assert report.ok is True
    assert report.n_joints_checked == 0
