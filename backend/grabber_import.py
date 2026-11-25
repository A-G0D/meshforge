"""
meshforge :: optional bridge from model-grabber's capture output to Half B.

model-grabber (a sibling personal project, not a dependency of this one)
watches a browser's network traffic and saves 3D assets it recognizes to a
`downloads/<session>/` directory alongside a `manifest.json` describing what
was captured (see its `capture.py:CaptureSession.write_manifest`). This module
reads that manifest — a plain JSON file, not a Python import — and picks the
single asset most likely to be the actual model geometry, so it can be handed
straight to `backend.printprep.pipeline.run(mesh_path=...)`.

There is no hard dependency on model-grabber: nothing here imports it, and
this module is only touched when the caller explicitly points at a grabbed
manifest (`--from-grab` on the CLI, or `resolve_grabbed_mesh()` directly).
Standalone meshforge usage (`--mesh`, `--image`+`--server`) never runs this
code and is unaffected if model-grabber's directory doesn't exist at all.
"""
from __future__ import annotations

import json
from pathlib import Path

# Extensions that are plausible standalone/primary model geometry, ranked
# best-first. Mirrors model-grabber's capture.py MODEL_EXTS but narrowed to
# things worth feeding to a print-prep pipeline as-is — texture/geometry-buffer
# fragments (.bin, .ktx2, .dds, ...) are deliberately excluded since they are
# not a loadable mesh on their own.
PREFERRED_EXTS = ("glb", "gltf", "obj", "fbx", "stl", "ply", "dae", "3mf")


def _load_manifest(manifest_path: Path) -> dict:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"no model-grabber manifest at {manifest_path}"
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed model-grabber manifest at {manifest_path}: {e}") from e
    if "assets" not in data or not isinstance(data["assets"], list):
        raise ValueError(f"{manifest_path} does not look like a model-grabber manifest "
                          "(missing 'assets' list)")
    return data


def _pick_best_asset(assets: list[dict]) -> dict | None:
    """Pick the asset most likely to be the primary model, not a texture or
    geometry-buffer fragment. Ranks by extension preference, then by size."""
    candidates = []
    for a in assets:
        saved = a.get("saved")
        if not saved:
            continue
        ext = saved.rsplit(".", 1)[-1].lower() if "." in saved else ""
        if ext not in PREFERRED_EXTS:
            continue
        candidates.append((PREFERRED_EXTS.index(ext), -(a.get("bytes") or 0), a))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2]


def resolve_grabbed_mesh(path: str | Path) -> str:
    """
    Resolve a model-grabber capture into a mesh file path Half B can consume.

    `path` may be:
      - a manifest.json file directly,
      - a session directory (e.g. `downloads/20260622_101500/`) containing one,
      - or an existing mesh file, passed straight through.

    Raises FileNotFoundError / ValueError with a clear message if the manifest
    is missing, malformed, or has no usable model asset. Never silently
    returns a wrong path.
    """
    p = Path(path)

    if p.is_file() and p.suffix.lower() != ".json":
        return str(p)

    manifest_path = p / "manifest.json" if p.is_dir() else p
    manifest = _load_manifest(manifest_path)

    best = _pick_best_asset(manifest["assets"])
    if best is None:
        raise ValueError(
            f"{manifest_path} has no captured asset with a usable model extension "
            f"({', '.join(PREFERRED_EXTS)}) — nothing to feed to print-prep."
        )

    mesh_path = manifest_path.parent / best["saved"]
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"manifest lists '{best['saved']}' but {mesh_path} does not exist on disk"
        )
    return str(mesh_path)
