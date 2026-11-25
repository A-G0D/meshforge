# Running the free GPU generation backbone

The heavy model can't run on your laptop (no CUDA GPU). It runs on a **free** cloud
GPU notebook that you launch yourself. The notebook exposes a public URL; your
local app connects to it. No payment, no box, no paid API.

Pick **Kaggle** (more reliable, 30 GPU-hrs/week, needs a phone-verified account for
internet) or **Colab** (instant, but flakier, ~12 h sessions).

## Kaggle (recommended)

1. kaggle.com → create account → **Settings → Phone verify** (required to enable
   notebook internet).
2. **Create → Notebook**. Right panel: **Accelerator = GPU T4 x2** (or P100),
   **Internet = On**.
3. In the first cell, install Hunyuan3D-2:
   ```python
   !git clone https://github.com/Tencent/Hunyuan3D-2 /kaggle/working/hy3d
   %cd /kaggle/working/hy3d
   !pip install -q -r requirements.txt && pip install -q -e . gradio
   ```
4. Upload `backend/notebook_server.py` (Notebook → File → Upload), then run it:
   ```python
   !python /kaggle/working/notebook_server.py          # shape-only (~6GB, robust)
   # !python /kaggle/working/notebook_server.py --texture   # +PBR (heavier)
   ```
5. Watch for:  `Running on public URL: https://xxxx.gradio.live`
   **Copy that URL.** It's valid for 72 h (or until the notebook stops).

## Colab

Same idea: **Runtime → Change runtime type → T4 GPU**, then run the clone/pip cell
(use `/content/...` paths), upload `notebook_server.py`, `!python notebook_server.py`.

## Then, on your laptop

```bash
cd meshforge
python backend/generate.py --server https://xxxx.gradio.live --inspect          # sanity check
python backend/generate.py --server https://xxxx.gradio.live --image inputs/test_blob.png
```

A `.glb` lands in `outputs/`. That's image → 3D on free compute. Half B then takes
that `.glb` and does the split/dowel/scarring print-prep locally.

## Notes / gotchas

- The `*.gradio.live` URL **changes every time** you restart the notebook — paste the
  fresh one. (This is the cost of free; a fixed URL would need a paid host.)
- Shape-only first: it's the most likely to "just work" on a free T4 and is what
  printing needs. Add `--texture` only once shape-only is confirmed.
- If pip `-e .` fails on the custom rasterizer, that only affects `--texture`;
  shape-only still works.
- Kaggle session caps at ~12 h; weekly GPU budget ~30 h. Colab disconnects sooner.
