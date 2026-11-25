"""
Tests for the optional model-grabber -> print-prep bridge
(backend/grabber_import.py, and its --from-grab wiring in backend/cli.py).

These build a manifest.json shaped exactly like model-grabber's
capture.py:CaptureSession.write_manifest produces, without importing
model-grabber itself — the two projects share a file-format contract, not
a Python dependency.
"""
from __future__ import annotations

import json

import trimesh

from backend import cli
from backend.grabber_import import resolve_grabbed_mesh


def _write_grab_session(session_dir, assets_spec) -> None:
    """assets_spec: list of (filename, bytes_written | None). None -> no file
    written (mirrors a manifest entry whose body was url-only/unavailable)."""
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_assets = []
    for name, payload in assets_spec:
        if payload is not None:
            (session_dir / name).write_bytes(payload)
            manifest_assets.append({
                "url": f"https://example.com/{name}",
                "content_type": "model/gltf-binary",
                "reason": "ext:." + name.rsplit(".", 1)[-1],
                "saved": name,
                "bytes": len(payload),
                "status": 200,
            })
        else:
            manifest_assets.append({
                "url": f"https://example.com/{name}",
                "content_type": "model/gltf-binary",
                "reason": "ext:." + name.rsplit(".", 1)[-1],
                "saved": None,
                "bytes": None,
                "note": "body unavailable",
            })
    (session_dir / "manifest.json").write_text(json.dumps({
        "session": session_dir.name,
        "captured_at": "2026-08-04T00:00:00",
        "start_url": "https://example.com/viewer",
        "count": len(manifest_assets),
        "assets": manifest_assets,
    }, indent=2), encoding="utf-8")


def _box_glb_bytes() -> bytes:
    import io
    buf = io.BytesIO()
    trimesh.creation.box(extents=[20.0, 20.0, 20.0]).export(buf, file_type="glb")
    return buf.getvalue()


def test_resolve_picks_glb_over_texture_and_buffer(tmp_path):
    session = tmp_path / "20260804_010101"
    glb = _box_glb_bytes()
    _write_grab_session(session, [
        ("scene.bin", b"\x00" * 64),           # geometry buffer, not usable alone
        ("diffuse.ktx2", b"\x00" * 128),        # texture fragment
        ("model.glb", glb),                     # the actual asset
    ])

    resolved = resolve_grabbed_mesh(session)
    assert resolved.endswith("model.glb")
    assert trimesh.load(resolved) is not None  # loads without error


def test_resolve_accepts_manifest_path_directly(tmp_path):
    session = tmp_path / "sess"
    _write_grab_session(session, [("thing.stl", b"solid x\nendsolid x\n")])

    resolved = resolve_grabbed_mesh(session / "manifest.json")
    assert resolved.endswith("thing.stl")


def test_resolve_raises_on_missing_manifest(tmp_path):
    try:
        resolve_grabbed_mesh(tmp_path / "does_not_exist")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_resolve_raises_when_no_usable_asset(tmp_path):
    session = tmp_path / "empty_session"
    _write_grab_session(session, [("scene.bin", b"\x00" * 10), ("tex.ktx2", None)])

    try:
        resolve_grabbed_mesh(session)
        assert False, "expected ValueError: no usable model asset"
    except ValueError:
        pass


def test_cli_from_grab_runs_pipeline_end_to_end(tmp_path, capsys):
    session = tmp_path / "grab_session"
    _write_grab_session(session, [("model.glb", _box_glb_bytes())])

    rc = cli.main([
        "--from-grab", str(session),
        "--size-mm", "50",
        "--out-dir", str(tmp_path / "out"),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "resolved model-grabber capture" in out
    assert "ok=True" in out
    produced = list((tmp_path / "out").glob("run_*/model.3mf"))
    assert len(produced) == 1 and produced[0].stat().st_size > 0


def test_cli_rejects_mesh_and_from_grab_together(tmp_path):
    try:
        cli.main(["--mesh", "a.glb", "--from-grab", str(tmp_path)])
        assert False, "expected SystemExit: mutually exclusive"
    except SystemExit as e:
        assert e.code != 0
