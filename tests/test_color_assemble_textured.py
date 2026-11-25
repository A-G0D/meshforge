"""
Regression test for the textured-mesh color path (ARCHITECTURE.md blocker:
"embed Bambu presets + paint_color into the .3mf ... needs textured Half-A
run for color").

A real textured Half-A run needs the free GPU notebook (network + a live
gradio.live URL), which is not available in this environment, so the full
generation-to-.3mf path is not verified here. This test instead builds a
synthetic textured mesh locally and drives it through the same two stages
that would consume Half A's texture output — color.assign_filaments and
assemble3mf.build_3mf — so a regression in either (e.g. texture sampling
silently falling back to had_color=False, or per-triangle color failing to
round-trip through the .3mf) is caught without needing the notebook.
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.texture import TextureVisuals

from backend.printprep import assemble3mf, color


def _textured_box() -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    rng = np.random.default_rng(0)
    uv = rng.random((len(mesh.vertices), 2))
    img = Image.new("RGB", (4, 4))
    px = img.load()
    for x in range(4):
        for y in range(4):
            px[x, y] = (255, 0, 0) if (x + y) % 2 == 0 else (0, 0, 255)
    mesh.visual = TextureVisuals(uv=uv, image=img)
    return mesh


def test_texture_quantizes_to_palette():
    mesh = _textured_box()
    result = color.assign_filaments(mesh, palette=color.PALETTE_DEFAULT)

    assert result.had_color is True
    assert result.face_filament.shape == (len(mesh.faces),)
    # every assigned slot must be a valid palette index
    assert result.face_filament.min() >= 0
    assert result.face_filament.max() < len(color.PALETTE_DEFAULT)


def test_colored_mesh_round_trips_through_3mf(tmp_path):
    mesh = _textured_box()
    color_result = color.assign_filaments(mesh, palette=color.PALETTE_DEFAULT)

    class _Part:
        def __init__(self, m, ff):
            self.mesh = m
            self.face_filament = ff
            self.joints: list = []

    out_path = tmp_path / "textured.3mf"
    assemble3mf.build_3mf([_Part(mesh, color_result.face_filament)],
                          color_result.palette, out_path)

    result = assemble3mf.build_3mf.last_result
    assert result.colored is True
    assert result.verified is True
    assert result.n_triangles == len(mesh.faces)
    assert out_path.exists() and out_path.stat().st_size > 0
