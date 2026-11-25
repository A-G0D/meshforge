"""
meshforge :: Half A — generation backbone client (free path).

Drives the image->3D model that you run on a FREE cloud GPU notebook
(Kaggle/Colab) via backend/notebook_server.py. That notebook prints a public
gradio share URL (https://xxxx.gradio.live); you pass it here as --server.

No Pro, no paid API, no local GPU: the GPU is the free notebook's T4/P100,
borrowed only while the notebook is running.

Usage:
  1. Start the notebook (see NOTEBOOK.md). Copy its https://xxxx.gradio.live URL.
  2. (sanity) python generate.py --server https://xxxx.gradio.live --inspect
  3. python generate.py --server https://xxxx.gradio.live --image ../inputs/cat.png

The server exposes one endpoint by contract:  /generate(image, steps, seed) -> .glb
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    from gradio_client import Client, handle_file
except ImportError:
    sys.exit("Missing dep: pip install gradio_client  (see requirements.txt)")

PROJECT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT / "outputs"

DEFAULT_ENDPOINT = "/generate"  # matches notebook_server.py's api_name


def connect(server: str, hf_token: str | None) -> Client:
    print(f"[meshforge] connecting to: {server}")
    return Client(server, token=hf_token)


def inspect(client: Client) -> None:
    print("=" * 70)
    print(client.view_api(return_format="str"))
    print("=" * 70)


def generate(client: Client, image_path: Path, *, steps: int, seed: int,
             api_name: str) -> Path:
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")
    print(f"[meshforge] calling {api_name!r} (steps={steps}, seed={seed}) ...")
    result = client.predict(
        handle_file(str(image_path)), steps, seed, api_name=api_name
    )
    return _save_result(result, image_path)


def _save_result(result, image_path: Path) -> Path:
    src = _find_mesh(result)
    if src is None:
        sys.exit(f"No mesh file found in result: {result!r}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = OUTPUTS / f"{image_path.stem}_{stamp}{Path(src).suffix}"
    shutil.copy(src, dst)
    print(f"[meshforge] saved mesh -> {dst}")
    return dst


def _find_mesh(obj) -> str | None:
    exts = (".glb", ".obj", ".ply", ".gltf")
    if isinstance(obj, str) and obj.lower().endswith(exts):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if (found := _find_mesh(v)):
                return found
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if (found := _find_mesh(v)):
                return found
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="meshforge generation backbone client")
    ap.add_argument("--server", required=True,
                    help="gradio share URL from the notebook, e.g. https://xxxx.gradio.live")
    ap.add_argument("--image", help="path to input image")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api-name", default=DEFAULT_ENDPOINT,
                    help="endpoint name (default: %(default)s)")
    ap.add_argument("--hf-token", help="HF token (optional)")
    ap.add_argument("--inspect", action="store_true", help="print the API and exit")
    args = ap.parse_args()

    client = connect(args.server, args.hf_token)
    if args.inspect:
        inspect(client)
        return
    if not args.image:
        sys.exit("Provide --image PATH (or --inspect).")
    generate(client, Path(args.image), steps=args.steps, seed=args.seed,
             api_name=args.api_name)


if __name__ == "__main__":
    main()
