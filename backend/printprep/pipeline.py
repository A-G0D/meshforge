"""
meshforge :: Half B — end-to-end print-prep pipeline orchestrator.

Wires all stages defensively:
  repair -> color -> orient -> split -> analyze -> assemble3mf -> slicer

Each stage is wrapped in a try/except; failures are recorded in the manifest
and the pipeline continues where sensible (e.g. a color failure falls back to
no-color, an orient failure falls back to the unoriented mesh, etc.).

A manifest.json is written to out_dir at the end containing stage results,
paths, and any errors.

Typical usage:
    from backend.printprep.pipeline import run
    result = run(mesh_path="outputs/test_blob_20260622_103958.glb", do_slice=False)
"""
from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import trimesh

# ---------------------------------------------------------------------------
# Project root / path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclass/ndarray objects to JSON-serialisable form."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    ok: bool
    mesh_path: str
    out_dir: str
    manifest_path: str
    stages: dict[str, Any]
    errors: dict[str, str]
    final_3mf: str | None
    slice_result: Any | None


# ---------------------------------------------------------------------------
# Individual stage wrappers (return (result, error_str | None))
# ---------------------------------------------------------------------------

def _stage_repair(mesh_path: str, size_mm: float, out_dir: Path):
    from backend.printprep import repair as repair_mod  # noqa: PLC0415

    out_stl = out_dir / "repaired.stl"
    mesh, report = repair_mod.process(
        mesh_path,
        target_max_mm=size_mm,
        out_path=str(out_stl),
    )
    return mesh, report, str(out_stl)


def _stage_color(mesh: trimesh.Trimesh, palette: list[str] | None):
    from backend.printprep import color as color_mod  # noqa: PLC0415

    kw: dict[str, Any] = {}
    if palette is not None:
        kw["palette"] = palette
    result = color_mod.assign_filaments(mesh, **kw)
    return result


def _stage_orient(mesh: trimesh.Trimesh):
    from backend.printprep import orient as orient_mod  # noqa: PLC0415

    result = orient_mod.optimize(mesh)
    return result


def _stage_split(mesh: trimesh.Trimesh, face_filament, plate_mm: float):
    from backend.printprep import split as split_mod  # noqa: PLC0415

    result = split_mod.plan_and_split(mesh, face_filament=face_filament,
                                      plate_mm=plate_mm)
    return result


def _stage_analyze(parts_or_mesh, plate_mm: float):
    from backend.printprep import analyze as analyze_mod  # noqa: PLC0415

    result = analyze_mod.audit(parts_or_mesh, plate_mm=plate_mm)
    return result


def _stage_assemble3mf(parts, palette: list[str], out_dir: Path):
    from backend.printprep import assemble3mf as assemble_mod  # noqa: PLC0415

    out_path = str(out_dir / "model.3mf")
    result_path = assemble_mod.build_3mf(parts, palette, out_path)
    return result_path


def _stage_slice(in_3mf: str, out_dir: Path):
    from backend.printprep import slicer as slicer_mod  # noqa: PLC0415

    out_3mf = str(out_dir / "sliced.gcode.3mf")
    result = slicer_mod.slice_project(in_3mf, out_3mf)
    return result


# ---------------------------------------------------------------------------
# Optional generation step
# ---------------------------------------------------------------------------

def _run_generation(image_path: str, server_url: str) -> str:
    """
    Call backend.generate to produce a .glb from an image.
    Returns the path to the generated mesh file.
    """
    from pathlib import Path as _Path  # noqa: PLC0415
    from backend.generate import connect, generate  # noqa: PLC0415

    client = connect(server_url, hf_token=None)
    img = _Path(image_path)
    out = generate(client, img, steps=30, seed=0, api_name="/generate")
    return str(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    *,
    image_path: str | None = None,
    mesh_path: str | None = None,
    server_url: str | None = None,
    palette: list[str] | None = None,
    size_mm: float = 150.0,
    out_dir: str = "outputs",
    do_slice: bool = False,
) -> dict:
    """
    Orchestrate the full Half B print-prep pipeline.

    Args:
        image_path:  Input image (requires server_url for generation).
        mesh_path:   Pre-existing mesh (skip generation). Takes precedence over
                     image_path for tests / direct use.
        server_url:  Gradio share URL for the generation server (Half A).
        palette:     Up to 4 AMS filament colours as UPPERCASE hex strings.
                     Defaults to color.PALETTE_DEFAULT.
        size_mm:     Target longest-axis size in mm (passed to repair.process).
        out_dir:     Output directory (created if absent). Relative paths are
                     resolved from the project root.
        do_slice:    Whether to call the Bambu Studio CLI slicer (default False).
                     OFF by default on purpose: the .3mf project is the deliverable
                     (open it in Bambu Studio and click Slice). Only enable for a
                     deliberate, standalone validation slice — never from parallel
                     agents (concurrent GUI launches froze the machine, 2026-06-22).

    Returns:
        A dict that mirrors the PipelineResult dataclass (JSON-serialisable).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve output directory
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_dir
    run_dir = out_path / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    stages: dict[str, Any] = {}
    errors: dict[str, str] = {}

    # -----------------------------------------------------------------------
    # 0. Optional generation
    # -----------------------------------------------------------------------
    source_mesh: str | None = mesh_path
    if source_mesh is None:
        if image_path and server_url:
            try:
                print("[pipeline] stage=generate ...")
                source_mesh = _run_generation(image_path, server_url)
                stages["generate"] = {"mesh_path": source_mesh}
                print(f"[pipeline] generate -> {source_mesh}")
            except Exception:
                tb = traceback.format_exc()
                errors["generate"] = tb
                print(f"[pipeline] generate FAILED:\n{tb}")
                # Cannot continue without a mesh
                return _finalise({}, errors, None, None, str(run_dir))
        else:
            msg = "No mesh_path and no (image_path + server_url) — cannot proceed."
            errors["generate"] = msg
            print(f"[pipeline] {msg}")
            return _finalise({}, errors, None, None, str(run_dir))

    # -----------------------------------------------------------------------
    # 1. Repair
    # -----------------------------------------------------------------------
    mesh: trimesh.Trimesh | None = None
    repair_report = None
    try:
        print("[pipeline] stage=repair ...")
        mesh, repair_report, repaired_path = _stage_repair(
            source_mesh, size_mm, run_dir
        )
        stages["repair"] = {
            "report": _to_dict(repair_report),
            "repaired_path": repaired_path,
        }
        print(f"[pipeline] repair ok  verts={repair_report.verts}  faces={repair_report.faces}")
    except Exception:
        tb = traceback.format_exc()
        errors["repair"] = tb
        print(f"[pipeline] repair FAILED:\n{tb}")
        # Attempt bare load so downstream can still run
        try:
            from backend.printprep.repair import load_solid  # noqa: PLC0415
            mesh = load_solid(source_mesh)
        except Exception:
            pass

    if mesh is None:
        print("[pipeline] could not obtain a mesh — aborting.")
        return _finalise(stages, errors, None, None, str(run_dir))

    # -----------------------------------------------------------------------
    # 2. Color
    # -----------------------------------------------------------------------
    color_result = None
    face_filament = None
    effective_palette: list[str] = palette or []
    try:
        print("[pipeline] stage=color ...")
        color_result = _stage_color(mesh, palette)
        face_filament = color_result.face_filament
        effective_palette = color_result.palette
        stages["color"] = {
            "had_color": color_result.had_color,
            "n_regions": color_result.n_regions,
            "palette": effective_palette,
        }
        print(f"[pipeline] color ok  had_color={color_result.had_color}  "
              f"n_regions={color_result.n_regions}")
        # From here on use the color-result mesh (same geometry, visuals may differ)
        mesh = color_result.mesh
    except Exception:
        tb = traceback.format_exc()
        errors["color"] = tb
        print(f"[pipeline] color FAILED (continuing without color):\n{tb}")
        # Provide minimal palette so downstream stages can continue
        if not effective_palette:
            try:
                from backend.printprep.color import PALETTE_DEFAULT  # noqa: PLC0415
                effective_palette = PALETTE_DEFAULT
            except Exception:
                effective_palette = ["#FFFFFF", "#1A1A1A", "#C8102E", "#005BBB"]

    # -----------------------------------------------------------------------
    # 3. Orient
    # -----------------------------------------------------------------------
    orient_result = None
    try:
        print("[pipeline] stage=orient ...")
        orient_result = _stage_orient(mesh)
        mesh = orient_result.mesh  # use oriented mesh going forward
        stages["orient"] = {
            "metrics": _to_dict(orient_result.metrics),
            "n_candidates": len(orient_result.candidates),
        }
        print(f"[pipeline] orient ok  score={orient_result.metrics.get('score', '?')}")
    except Exception:
        tb = traceback.format_exc()
        errors["orient"] = tb
        print(f"[pipeline] orient FAILED (continuing with unoriented mesh):\n{tb}")

    # -----------------------------------------------------------------------
    # 4. Split
    # -----------------------------------------------------------------------
    split_result = None
    parts = None
    try:
        print("[pipeline] stage=split ...")
        split_result = _stage_split(mesh, face_filament, plate_mm=256.0)
        parts = split_result.parts
        stages["split"] = {
            "n_parts": len(parts),
            "n_cuts": split_result.n_cuts,
            "notes": split_result.notes,
        }
        print(f"[pipeline] split ok  n_parts={len(parts)}  n_cuts={split_result.n_cuts}")
    except Exception:
        tb = traceback.format_exc()
        errors["split"] = tb
        print(f"[pipeline] split FAILED (treating mesh as single part):\n{tb}")
        # Fallback: wrap the mesh as a single Part-like object
        parts = _make_fallback_part(mesh, face_filament)

    # -----------------------------------------------------------------------
    # 5. Analyze
    # -----------------------------------------------------------------------
    try:
        print("[pipeline] stage=analyze ...")
        analyze_result = _stage_analyze(parts if parts else mesh, plate_mm=256.0)
        stages["analyze"] = {
            "warnings": analyze_result.warnings,
            "thin_face_count": analyze_result.thin_face_count,
            "disconnected_components": analyze_result.disconnected_components,
            "overhang_fraction": analyze_result.overhang_fraction,
            "footprint_ok": analyze_result.footprint_ok,
            "fits_plate": analyze_result.fits_plate,
            "est_support_fraction": analyze_result.est_support_fraction,
            "summary": analyze_result.summary,
        }
        print(f"[pipeline] analyze ok  summary={analyze_result.summary!r}")
    except Exception:
        tb = traceback.format_exc()
        errors["analyze"] = tb
        print(f"[pipeline] analyze FAILED (non-fatal):\n{tb}")

    # -----------------------------------------------------------------------
    # 6. Assemble .3mf
    # -----------------------------------------------------------------------
    final_3mf: str | None = None
    try:
        print("[pipeline] stage=assemble3mf ...")
        final_3mf = _stage_assemble3mf(parts, effective_palette, run_dir)
        stages["assemble3mf"] = {"out_path": final_3mf}
        print(f"[pipeline] assemble3mf ok  -> {final_3mf}")
    except Exception:
        tb = traceback.format_exc()
        errors["assemble3mf"] = tb
        print(f"[pipeline] assemble3mf FAILED:\n{tb}")

    # -----------------------------------------------------------------------
    # 7. Slice
    # -----------------------------------------------------------------------
    slice_result = None
    if do_slice and final_3mf is not None:
        try:
            print("[pipeline] stage=slice ...")
            slice_result = _stage_slice(final_3mf, run_dir)
            stages["slicer"] = {
                "ok": slice_result.ok,
                "out_path": slice_result.out_path,
                "returncode": slice_result.returncode,
                "notes": slice_result.notes,
            }
            print(f"[pipeline] slice  ok={slice_result.ok}  rc={slice_result.returncode}")
        except Exception:
            tb = traceback.format_exc()
            errors["slicer"] = tb
            print(f"[pipeline] slice FAILED:\n{tb}")
    elif do_slice and final_3mf is None:
        note = "Skipped: no .3mf produced by assemble3mf stage."
        stages["slicer"] = {"ok": False, "notes": note}
        print(f"[pipeline] slice skipped — {note}")
    else:
        stages["slicer"] = {"ok": False, "notes": "do_slice=False"}
        print("[pipeline] slice skipped (do_slice=False)")

    return _finalise(stages, errors, final_3mf, slice_result, str(run_dir))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fallback_part(mesh: trimesh.Trimesh, face_filament) -> list:
    """
    Create a minimal Part-like object from a mesh when split.py is unavailable.
    Returns a list containing one object with the attributes Part is expected to have.
    """
    class _FallbackPart:
        def __init__(self, m, ff):
            self.mesh = m
            self.face_filament = ff
            self.joints: list = []

    return [_FallbackPart(mesh, face_filament)]


def _finalise(
    stages: dict,
    errors: dict,
    final_3mf: str | None,
    slice_result: Any,
    run_dir: str,
) -> dict:
    """Serialise results into a dict and write manifest.json."""
    ok = len(errors) == 0

    manifest_path = Path(run_dir) / "manifest.json"
    data: dict[str, Any] = {
        "ok": ok,
        "run_dir": run_dir,
        "manifest_path": str(manifest_path),
        "final_3mf": final_3mf,
        "slice_result": _to_dict(slice_result) if slice_result else None,
        "stages": stages,
        "errors": errors,
    }
    try:
        manifest_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"[pipeline] manifest written -> {manifest_path}")
    except Exception as exc:
        print(f"[pipeline] WARNING: could not write manifest: {exc}")
    return data


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Prefer the raw .glb for a more thorough test
    raw_glb = PROJECT_ROOT / "outputs" / "test_blob_20260622_103958.glb"
    repaired_stl = PROJECT_ROOT / "outputs" / "test_blob_repaired.stl"

    if raw_glb.exists():
        test_mesh = str(raw_glb)
    elif repaired_stl.exists():
        test_mesh = str(repaired_stl)
    else:
        print("No test mesh found under outputs/. Aborting.")
        sys.exit(1)

    print(f"[smoke] Running pipeline on: {test_mesh}")
    print(f"[smoke] do_slice=False  size_mm=150.0")
    print("=" * 70)

    result = run(
        mesh_path=test_mesh,
        size_mm=150.0,
        out_dir="outputs",
        do_slice=False,
    )

    print("=" * 70)
    print("[smoke] Pipeline complete.")
    print(f"[smoke] ok={result['ok']}")
    print(f"[smoke] run_dir={result['run_dir']}")
    print(f"[smoke] final_3mf={result['final_3mf']}")
    print(f"[smoke] stages completed: {list(result['stages'].keys())}")
    if result["errors"]:
        print(f"[smoke] stage errors: {list(result['errors'].keys())}")
    print()
    print(json.dumps(result, indent=2, default=str))
