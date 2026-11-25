"""
meshforge :: unified CLI entry point.

Wires Half A (generation, via a notebook's gradio share URL) and Half B
(local print-prep) into one command, so there is a single documented run
command instead of "run generate.py by hand, then call pipeline.run() from
a Python shell."

Usage:
    # Half A + Half B, starting from an image (needs a running notebook):
    python -m backend.cli --image inputs/test_blob.png --server https://xxxx.gradio.live

    # Half B only, starting from an existing mesh:
    python -m backend.cli --mesh outputs/test_blob_20260622_103958.glb

    # Half B, from a model-grabber capture (session dir or manifest.json) —
    # optional bridge, no hard dependency on model-grabber; see
    # backend/grabber_import.py:
    python -m backend.cli --from-grab ../model-grabber/downloads/20260622_101500

do_slice stays off unless --slice is passed explicitly: running the Bambu
Studio slicer concurrently from more than one process froze the machine once
(see ARCHITECTURE.md post-mortem, 2026-06-22). slicer.slice_project now also
refuses a second concurrent launch on its own via a filesystem lock, but the
default here still stays conservative.
"""
from __future__ import annotations

import argparse
import sys

from backend.printprep.pipeline import run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="meshforge: image or mesh -> print-ready Bambu .3mf"
    )
    ap.add_argument("--image", help="input image path (requires --server)")
    ap.add_argument("--mesh", help="existing mesh file (.glb/.stl/...) — skips generation")
    ap.add_argument("--from-grab", metavar="PATH",
                    help="model-grabber capture to use as the mesh — a session dir "
                         "(e.g. downloads/<timestamp>/) or manifest.json path; picks "
                         "the best model asset from it. Mutually exclusive with --mesh.")
    ap.add_argument("--server", help="gradio share URL from the notebook (Half A)")
    ap.add_argument("--palette", nargs="+", metavar="HEX",
                    help="up to 4 '#RRGGBB' AMS filament colors (default: color.PALETTE_DEFAULT)")
    ap.add_argument("--size-mm", type=float, default=150.0,
                    help="target longest-axis size in mm (default: 150.0)")
    ap.add_argument("--out-dir", default="outputs", help="output directory (default: outputs)")
    ap.add_argument("--slice", action="store_true",
                    help="also run the Bambu Studio slicer (default off; only ever run "
                         "this standalone and watched — never from parallel invocations)")
    args = ap.parse_args(argv)

    if args.mesh and args.from_grab:
        ap.error("--mesh and --from-grab are mutually exclusive")

    if not args.mesh and not args.from_grab and not (args.image and args.server):
        ap.error("provide --mesh, --from-grab, or --image together with --server")

    mesh_path = args.mesh
    if args.from_grab:
        from backend.grabber_import import resolve_grabbed_mesh  # noqa: PLC0415
        try:
            mesh_path = resolve_grabbed_mesh(args.from_grab)
        except (FileNotFoundError, ValueError) as e:
            ap.error(f"--from-grab: {e}")
        print(f"[cli] resolved model-grabber capture -> {mesh_path}")

    result = run(
        image_path=args.image,
        mesh_path=mesh_path,
        server_url=args.server,
        palette=args.palette,
        size_mm=args.size_mm,
        out_dir=args.out_dir,
        do_slice=args.slice,
    )

    print(f"ok={result['ok']}  final_3mf={result['final_3mf']}  manifest={result['manifest_path']}")
    if result["errors"]:
        print(f"stage errors: {list(result['errors'].keys())}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
