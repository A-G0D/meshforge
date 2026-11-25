"""
meshforge :: Half A — free GPU notebook server.

Runs on a FREE cloud GPU notebook (Kaggle T4/P100 16GB, or Google Colab T4 16GB)
and exposes an image->3D endpoint over a public gradio share URL. The local app
(backend/generate.py) then drives it via gradio_client — no Pro, no paid API,
no local GPU.

Model: Tencent Hunyuan3D-2.
  - shape-only (default): ~6GB VRAM, very robust on a free T4. Untextured mesh —
    which is what 3D printing actually wants (color = filament + slicer, not PBR).
  - --texture: adds Hunyuan3D-Paint (heavier; needs the custom rasterizer build).

HOW TO RUN (see NOTEBOOK.md for the click-by-click version):
  In a GPU-enabled Kaggle/Colab cell, with Internet ON:

    !git clone https://github.com/Tencent/Hunyuan3D-2 /content/hy3d
    %cd /content/hy3d
    !pip install -q -r requirements.txt && pip install -q -e . gradio
    # upload this file, then:
    !python /content/notebook_server.py            # shape-only
    # or: !python /content/notebook_server.py --texture

It prints a line like:  Running on public URL: https://xxxx.gradio.live
Copy that URL — that's what you pass to generate.py as --space.
"""
from __future__ import annotations

import argparse
import sys

import gradio as gr


def build_pipelines(use_texture: bool):
    """Import + load Hunyuan3D-2. Imported lazily so --help works without GPU deps."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    print("[meshforge-gpu] loading shape pipeline (tencent/Hunyuan3D-2)...")
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
    rembg = BackgroundRemover()

    paint = None
    if use_texture:
        print("[meshforge-gpu] loading paint pipeline (textures)...")
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")

    return shape, rembg, paint


def make_generate_fn(shape, rembg, paint):
    def generate(image, steps: int, seed: int):
        """image (PIL) -> path to a .glb file."""
        if image is None:
            raise gr.Error("No image provided.")
        img = rembg(image)  # remove background -> better reconstruction
        mesh = shape(
            image=img,
            num_inference_steps=int(steps),
            generator=_seeded(seed),
        )[0]
        if paint is not None:
            mesh = paint(mesh, image=img)
        out = "/tmp/meshforge_out.glb"
        mesh.export(out)
        print(f"[meshforge-gpu] exported {out}  (verts~{len(mesh.vertices)})")
        return out

    return generate


def _seeded(seed: int):
    try:
        import torch
        return torch.Generator(device="cuda").manual_seed(int(seed))
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="meshforge free-GPU notebook server")
    ap.add_argument("--texture", action="store_true",
                    help="also generate PBR texture (heavier; needs rasterizer build)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    try:
        shape, rembg, paint = build_pipelines(args.texture)
    except ImportError as e:
        sys.exit(f"Hunyuan3D deps not installed in this notebook: {e}\n"
                 f"Run the !git clone + pip install cell first (see NOTEBOOK.md).")

    generate = make_generate_fn(shape, rembg, paint)

    # Blocks (not Interface) so we control the endpoint name -> /generate, which is
    # the contract backend/generate.py calls. Interface would name it /predict.
    with gr.Blocks(title="meshforge generation backbone (Hunyuan3D-2)") as demo:
        gr.Markdown("## meshforge generation backbone (Hunyuan3D-2)")
        with gr.Row():
            inp = gr.Image(type="pil", label="input image")
            out = gr.Model3D(label="mesh")
        steps = gr.Slider(10, 50, value=30, step=1, label="steps")
        seed = gr.Number(value=0, label="seed", precision=0)
        btn = gr.Button("Generate", variant="primary")
        btn.click(generate, inputs=[inp, steps, seed], outputs=out,
                  api_name="generate")
    # share=True -> public *.gradio.live URL the local app connects to.
    demo.launch(share=True, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
