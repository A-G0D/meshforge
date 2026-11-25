"""Print-prep benchmark: repair rate, time per mesh, orientation support area.

Meshes: 2 files from trimesh/models (MIT, data/) + 4 generated primitives.
Each is damaged with seeded defects, then repaired by (a) naive baseline
(merge + fill_holes only) and (b) backend.printprep.repair.repair.
Run: python scripts/bench_printprep.py [--out results/bench_printprep.json]
"""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.printprep import repair as R, orient as O

def sources():
    out = {}
    for f in ["featuretype.STL", "ballA.off"]:
        m = trimesh.load(ROOT / "data" / f, force="mesh", process=True)
        out[f] = m
    out["icosphere"] = trimesh.creation.icosphere(3)
    out["box"] = trimesh.creation.box((1, 2, 3))
    out["torus"] = trimesh.creation.torus(2, 0.6, 48, 24) if hasattr(trimesh.creation, "torus") else trimesh.creation.annulus(1, 2, 1)
    out["capsule"] = trimesh.creation.capsule(3, 1)
    # shortcut: damage is only meaningful on a clean reference, so drop sources that are not closed volumes
    return {k: m for k, m in out.items() if m.is_volume}

def damage(m, kind, rng):
    v, f = m.vertices.copy(), m.faces.copy()
    n = len(f)
    if kind == "holes2":
        f = f[rng.random(n) > 0.02]
    elif kind == "holes10":
        f = f[rng.random(n) > 0.10]
    elif kind == "flip30":
        k = rng.random(n) < 0.30; f[k] = f[k][:, ::-1]
    elif kind == "dup5":
        f = np.vstack([f, f[rng.random(n) < 0.05]])
    elif kind == "soup":  # every triangle owns its vertices
        v, f = v[f].reshape(-1, 3), np.arange(3 * n).reshape(-1, 3)
    elif kind == "combo":  # 3% holes + 20% flipped + soup
        f = f[rng.random(n) > 0.03]
        k = rng.random(len(f)) < 0.20; f[k] = f[k][:, ::-1]
        v, f = v[f].reshape(-1, 3), np.arange(3 * len(f)).reshape(-1, 3)
    return trimesh.Trimesh(v, f, process=False)

def naive(m):
    m = m.copy(); m.merge_vertices(); trimesh.repair.fill_holes(m); return m

def ok(m, ref_vol):
    return bool(m.is_watertight and m.is_winding_consistent and m.volume > 0), \
        (abs(m.volume - ref_vol) / ref_vol if m.is_watertight else None)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results/bench_printprep.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for name, src in sources().items():
        ref = abs(src.volume)
        for kind in ["holes2", "holes10", "flip30", "dup5", "soup", "combo"]:
            bad = damage(src, kind, rng)
            t = time.perf_counter(); nv = naive(bad); tn = time.perf_counter() - t
            t = time.perf_counter(); mf, _ = R.repair(bad); tm = time.perf_counter() - t
            o_n, e_n = ok(nv, ref); o_m, e_m = ok(mf, ref)
            rows.append(dict(mesh=name, defect=kind, faces=int(len(bad.faces)),
                             naive_ok=o_n, meshforge_ok=o_m, naive_s=tn, meshforge_s=tm,
                             naive_vol_err=e_n, meshforge_vol_err=e_m))
    # orientation: random delivered pose vs optimize()
    orows = []
    for name, src in sources().items():
        for s in range(3):
            T = trimesh.transformations.random_rotation_matrix(rng.random(3))
            m = src.copy(); m.apply_transform(T)
            t = time.perf_counter(); r = O.optimize(m); dt = time.perf_counter() - t
            a0 = O._evaluate(np.asarray(m.vertices), np.asarray(m.face_normals), np.asarray(m.area_faces))
            orows.append(dict(mesh=name, seed=s, total_area=float(m.area),
                              support_before=a0["support_contact_area"],
                              support_after=r.metrics["support_contact_area"], seconds=dt))
    n = len(rows)
    summ = dict(
        repair_cases=n,
        naive_rate=sum(r["naive_ok"] for r in rows) / n,
        meshforge_rate=sum(r["meshforge_ok"] for r in rows) / n,
        median_s_meshforge=float(np.median([r["meshforge_s"] for r in rows])),
        median_faces=float(np.median([r["faces"] for r in rows])),
        orient_cases=len(orows),
        support_frac_before=float(np.mean([r["support_before"] / r["total_area"] for r in orows])),
        support_frac_after=float(np.mean([r["support_after"] / r["total_area"] for r in orows])),
        median_orient_s=float(np.median([r["seconds"] for r in orows])),
        machine=f"{platform.machine()} {platform.system()} py{platform.python_version()} trimesh {trimesh.__version__}")
    Path(a.out).write_text(json.dumps(dict(summary=summ, repair=rows, orient=orows), indent=1))
    print(json.dumps(summ, indent=1))
    for r in rows:
        if not r["meshforge_ok"]: print("FAIL", r["mesh"], r["defect"], r["faces"])

if __name__ == "__main__": main()
