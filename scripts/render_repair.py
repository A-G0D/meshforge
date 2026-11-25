"""Headless before/after renders (matplotlib Agg, no window). Open edges drawn red.
Run: python scripts/render_repair.py   -> docs/repair_before_after.png"""
import sys
from pathlib import Path
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from backend.printprep.repair import repair
from bench_printprep import damage

def draw(ax, m, title):
    tri = m.vertices[m.faces]
    ax.add_collection3d(Poly3DCollection(tri, facecolor="#9fb4c7", edgecolor="#5c6f80", linewidths=0.1))
    u, inv, c = np.unique(m.edges_sorted, axis=0, return_inverse=True, return_counts=True)
    open_e = u[c != 2]
    for a, b in open_e:
        ax.plot(*zip(m.vertices[a], m.vertices[b]), color="red", lw=1.2)
    b = m.bounds; ctr = b.mean(0); r = (b[1] - b[0]).max() / 2
    ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r); ax.set_zlim(ctr[2]-r, ctr[2]+r)
    ax.set_axis_off(); ax.view_init(25, 35)
    ax.set_title(f"{title}\nwatertight={m.is_watertight}, open edges={len(open_e)}", fontsize=9)

rng = np.random.default_rng(0)
cases = [("icosphere (generated)", trimesh.creation.icosphere(3)),
         ("featuretype.STL (trimesh, MIT)", trimesh.load(ROOT / "data/featuretype.STL", force="mesh"))]
fig = plt.figure(figsize=(10, 9))
for i, (name, src) in enumerate(cases):
    bad = damage(src, "holes10", rng)
    fixed, _ = repair(bad)
    draw(fig.add_subplot(2, 2, 2*i+1, projection="3d"), bad, f"{name}: 10% faces deleted")
    draw(fig.add_subplot(2, 2, 2*i+2, projection="3d"), fixed, "after meshforge repair")
fig.suptitle("meshforge repair.py on synthetic damage (matplotlib render, red = open edges)", fontsize=10)
fig.savefig(ROOT / "docs/repair_before_after.png", dpi=110)
