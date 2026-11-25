"""
Tests for the slicer concurrency lock (backend/printprep/slicer.py).

Post-mortem (ARCHITECTURE.md, 2026-06-22): two bambu-studio.exe GUI processes
launched at once froze the machine. Before this fix that was only a "don't do
that" rule in docs; these tests verify the filesystem lock now makes it an
automated refusal instead. None of these tests ever launch the real
bambu-studio.exe — `_find_bambu_exe` and `_launch` are monkeypatched.
"""
from __future__ import annotations

import os
import time

from backend.printprep import slicer


def test_acquire_lock_blocks_second_holder(tmp_path):
    lock = tmp_path / ".slicer.lock"

    ok1, holder1 = slicer._acquire_lock(lock)
    assert ok1 is True
    assert holder1 is None
    assert lock.exists()

    ok2, holder2 = slicer._acquire_lock(lock)
    assert ok2 is False
    assert holder2 == str(os.getpid())

    slicer._release_lock(lock)
    assert not lock.exists()


def test_acquire_lock_reclaims_stale_lock(tmp_path):
    lock = tmp_path / ".slicer.lock"
    lock.write_text("99999999", encoding="utf-8")
    stale_time = time.time() - (slicer.LOCK_STALE_S + 10)
    os.utime(lock, (stale_time, stale_time))

    ok, holder = slicer._acquire_lock(lock)
    assert ok is True
    assert holder is None
    slicer._release_lock(lock)


def test_slice_project_refuses_when_locked(tmp_path, monkeypatch):
    lock = tmp_path / ".slicer.lock"
    lock.write_text("12345", encoding="utf-8")

    in_3mf = tmp_path / "in.3mf"
    in_3mf.write_bytes(b"fake")
    out_3mf = tmp_path / "out.gcode.3mf"

    monkeypatch.setattr(slicer, "_find_bambu_exe", lambda explicit=None: "fake-bambu-studio.exe")

    def _boom(*_a, **_k):
        raise AssertionError("must not launch the slicer while the lock is held")

    monkeypatch.setattr(slicer, "_launch", _boom)

    result = slicer.slice_project(in_3mf, out_3mf, slice=True, lock_path=lock)

    assert result.ok is False
    assert "already in progress" in result.notes
    assert not out_3mf.exists()
    assert lock.exists()  # the other holder's lock is left untouched


def test_slice_project_acquires_and_releases_lock_on_success(tmp_path, monkeypatch):
    lock = tmp_path / ".slicer.lock"
    in_3mf = tmp_path / "in.3mf"
    in_3mf.write_bytes(b"fake")
    out_3mf = tmp_path / "out.gcode.3mf"

    monkeypatch.setattr(slicer, "_find_bambu_exe", lambda explicit=None: "fake-bambu-studio.exe")

    def _fake_launch(_cmd, _cwd, timeout=600):
        out_3mf.write_bytes(b"gcode")
        return 0

    monkeypatch.setattr(slicer, "_launch", _fake_launch)

    result = slicer.slice_project(in_3mf, out_3mf, slice=True, lock_path=lock)

    assert result.ok is True
    assert not lock.exists()  # released after the slice completes


def test_slice_project_releases_lock_on_launch_failure(tmp_path, monkeypatch):
    lock = tmp_path / ".slicer.lock"
    in_3mf = tmp_path / "in.3mf"
    in_3mf.write_bytes(b"fake")
    out_3mf = tmp_path / "out.gcode.3mf"

    monkeypatch.setattr(slicer, "_find_bambu_exe", lambda explicit=None: "fake-bambu-studio.exe")
    monkeypatch.setattr(slicer, "_launch", lambda *_a, **_k: 1)  # nonzero exit, no output

    result = slicer.slice_project(in_3mf, out_3mf, slice=True, lock_path=lock)

    assert result.ok is False
    assert not lock.exists()  # released even though the slice failed
