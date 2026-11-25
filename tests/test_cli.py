"""
Tests for the unified CLI entry point (backend/cli.py).

Before this, there was no single command tying Half A and Half B together —
the documented run command was "run generate.py by hand, then call
pipeline.run() from a Python shell." These tests exercise the CLI's argument
handling and confirm a clean invocation actually runs the pipeline end to end
and returns the right exit code.
"""
from __future__ import annotations

import trimesh

from backend import cli


def _write_box_mesh(path) -> None:
    trimesh.creation.box(extents=[20.0, 20.0, 20.0]).export(str(path))


def test_cli_runs_pipeline_end_to_end(tmp_path, capsys):
    mesh_path = tmp_path / "box.stl"
    _write_box_mesh(mesh_path)

    rc = cli.main([
        "--mesh", str(mesh_path),
        "--size-mm", "50",
        "--out-dir", str(tmp_path / "out"),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "ok=True" in out
    assert (tmp_path / "out").exists()
    produced = list((tmp_path / "out").glob("run_*/model.3mf"))
    assert len(produced) == 1 and produced[0].stat().st_size > 0


def test_cli_requires_mesh_or_image_plus_server():
    try:
        cli.main([])
        assert False, "expected SystemExit from argparse.error()"
    except SystemExit as e:
        assert e.code != 0


def test_cli_rejects_image_without_server(tmp_path):
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        cli.main(["--image", str(img)])
        assert False, "expected SystemExit: --image without --server"
    except SystemExit as e:
        assert e.code != 0
