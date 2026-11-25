"""
meshforge :: Half B — drive Bambu Studio CLI to slice a print-ready .3mf.

Final stage of print-prep. Takes the Bambu-flavored .3mf emitted by assemble3mf
(geometry + per-triangle filament paint) and slices it to gcode using embedded
P1S / PLA presets, producing a `<name>.gcode.3mf` the printer can consume.

Two hard facts about the toolchain (verified 2026-06-22), both baked into how this
module judges success:

  1. ``bambu-studio.exe`` on Windows is a GUI-subsystem binary: it has NO console
     stdout/stderr even when launched headless with ``--slice``. So we judge success
     by (a) process exit code == 0 AND (b) the output .3mf actually appearing on disk
     — never by parsing stdout (there is none). Empirically the binary ALSO refuses
     to run (exits -100 / 4294967196) when launched by Python's ``subprocess`` with
     console handles attached, regardless of creationflags. It runs cleanly only when
     started via ``Start-Process -NoNewWindow -Wait``, so on Windows we shell out
     through PowerShell to launch it and propagate the exit code.

  2. The CLI resolves a preset's ``inherits`` chain against its own bundled
     ``resources/profiles`` tree. So the presets shipped under ``presets/`` are the
     real BBL system JSONs (P1S 0.4 machine, 0.20mm Standard process, PLA Basic
     filament) copied verbatim — they keep their ``inherits`` keys and the binary
     fills in the parents. The one edit made was widening the process profile's
     ``compatible_printers`` to include the P1S machine name, otherwise the CLI
     rejects the machine/process pairing with exit -17.

CLI shape (Orca/Bambu identical):
    bambu-studio --slice 1
        --load-settings "machine.json;process.json"
        --load-filaments "filament.json"
        --export-3mf out.gcode.3mf
        in.3mf
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Default install location of the Bambu Studio CLI on Windows.
BAMBU_EXE_DEFAULT = r"C:\Program Files\Bambu Studio\bambu-studio.exe"

# Filesystem lock guarding against the concurrent-launch freeze (see module
# docstring / ARCHITECTURE.md post-mortem, 2026-06-22): two bambu-studio.exe
# GUI processes launched at once previously froze the machine. This lock makes
# that an automated refusal instead of a manual "don't do that" rule.
LOCK_PATH_DEFAULT = Path(__file__).resolve().parents[2] / "outputs" / ".slicer.lock"
# Longer than _launch's own 600s timeout, so a lock can only look stale after
# the process that held it must already be dead.
LOCK_STALE_S = 900


def _acquire_lock(lock_path: Path) -> tuple[bool, str | None]:
    """Try to create ``lock_path`` exclusively. Returns (acquired, holder_pid_or_None).

    A lock older than ``LOCK_STALE_S`` is treated as abandoned by a crashed
    process and removed before the fresh attempt.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age > LOCK_STALE_S:
            try:
                lock_path.unlink()
            except OSError:
                pass
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        return True, None
    except FileExistsError:
        try:
            holder = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            holder = "unknown"
        return False, holder


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except OSError:
        pass

# Presets that ship alongside this module (real BBL system JSONs, trimmed names).
_PRESETS_DIR_DEFAULT = Path(__file__).resolve().parent / "presets"
MACHINE_JSON = "p1s_machine.json"
PROCESS_JSON = "pla_process.json"
FILAMENT_JSON = "pla_filament.json"

# Bambu Studio returns 0 on a clean slice; anything else is a failure.
_OK_RETURNCODE = 0


@dataclass
class SliceResult:
    ok: bool
    out_path: str
    returncode: int
    notes: str

    def as_dict(self) -> dict:
        return asdict(self)


def _find_bambu_exe(explicit: str | Path | None = None) -> str | None:
    """Locate the Bambu Studio CLI. Returns an absolute path or None."""
    if explicit:
        p = Path(explicit)
        return str(p) if p.exists() else None
    p = Path(BAMBU_EXE_DEFAULT)
    if p.exists():
        return str(p)
    # Fall back to PATH (covers OrcaSlicer or a relocated install).
    for name in ("bambu-studio", "bambu-studio.exe", "orca-slicer", "orca-slicer.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _resolve_presets(presets_dir: str | Path | None) -> tuple[Path, Path, Path]:
    """Return (machine, process, filament) preset paths, raising if any is missing."""
    base = Path(presets_dir) if presets_dir else _PRESETS_DIR_DEFAULT
    machine = base / MACHINE_JSON
    process = base / PROCESS_JSON
    filament = base / FILAMENT_JSON
    missing = [str(p) for p in (machine, process, filament) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing preset(s): " + ", ".join(missing))
    return machine, process, filament


def build_command(in_3mf: str | Path, out_3mf: str | Path,
                  machine: Path, process: Path, filament: Path,
                  exe: str, *, do_slice: bool = True) -> list[str]:
    """Assemble the bambu-studio CLI argv. Kept pure for logging / dry-runs."""
    # load-settings takes machine + process joined by ';'.
    settings = f"{machine};{process}"
    cmd: list[str] = [exe]
    if do_slice:
        cmd += ["--slice", "1"]
    cmd += [
        "--load-settings", settings,
        "--load-filaments", str(filament),
        "--export-3mf", str(out_3mf),
        str(in_3mf),
    ]
    return cmd


def _ps_quote(s: str) -> str:
    """Single-quote a string for a PowerShell argument list."""
    return "'" + s.replace("'", "''") + "'"


def _launch(cmd: list[str], cwd: Path, timeout: int = 600) -> int:
    """Launch the slicer and return its exit code.

    On Windows the Bambu GUI binary aborts (-100) under a direct ``subprocess``
    launch, so we drive it via ``Start-Process -NoNewWindow -Wait -PassThru`` and
    propagate ``$p.ExitCode``. Elsewhere (Orca on Linux/macOS) a plain subprocess
    is fine.
    """
    if os.name == "nt":
        exe, argv = cmd[0], cmd[1:]
        arglist = ",".join(_ps_quote(a) for a in argv) if argv else "@()"
        script = (
            f"$ErrorActionPreference='Stop'; "
            f"$p = Start-Process -FilePath {_ps_quote(exe)} "
            f"-ArgumentList {arglist} -NoNewWindow -Wait -PassThru; "
            f"exit $p.ExitCode"
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            timeout=timeout, cwd=str(cwd),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return proc.returncode
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=str(cwd))
    return proc.returncode


def slice_project(in_3mf: str | Path, out_3mf: str | Path, *,
                  presets_dir: str | Path | None = None,
                  slice: bool = True,
                  lock_path: str | Path | None = None) -> SliceResult:
    """Slice ``in_3mf`` to ``out_3mf`` with Bambu Studio, judging success by exit
    code + output-file existence (the binary emits no stdout on Windows).

    If the slice cannot be confirmed (binary missing, presets missing, non-zero
    exit, or no output file), returns ``ok=False`` with the exact command and
    reason in ``notes`` so the pipeline can record it and continue.
    """
    in_3mf = Path(in_3mf)
    out_3mf = Path(out_3mf)
    out_str = str(out_3mf)

    if not in_3mf.exists():
        return SliceResult(False, out_str, -1,
                           f"input .3mf not found: {in_3mf}")

    exe = _find_bambu_exe()
    if exe is None:
        return SliceResult(False, out_str, -1,
                           f"bambu-studio.exe not found at {BAMBU_EXE_DEFAULT} or on PATH")

    try:
        machine, process, filament = _resolve_presets(presets_dir)
    except FileNotFoundError as e:
        return SliceResult(False, out_str, -1, str(e))

    out_3mf.parent.mkdir(parents=True, exist_ok=True)
    # Stale output would otherwise masquerade as a fresh success.
    if out_3mf.exists():
        out_3mf.unlink()

    cmd = build_command(in_3mf, out_3mf, machine, process, filament, exe,
                        do_slice=slice)
    cmd_str = subprocess.list2cmdline(cmd)

    if not slice:
        return SliceResult(False, out_str, 0,
                           "slice=False: command built but not executed\n" + cmd_str)

    # Slicer resolves preset paths relative to its cwd when given relative; we pass
    # absolute paths, so cwd just anchors any relative I/O — use the project root.
    cwd = Path.cwd()

    lp = Path(lock_path) if lock_path else LOCK_PATH_DEFAULT
    acquired, holder = _acquire_lock(lp)
    if not acquired:
        return SliceResult(False, out_str, -1,
                           f"refusing to launch bambu-studio: another slice is already "
                           f"in progress (lock held by pid {holder} at {lp}); concurrent "
                           "slicer launches froze the machine on 2026-06-22 — wait for "
                           "it to finish, or delete the lock file if you are sure no "
                           "slicer process is running")

    try:
        # GUI-subsystem binary: stdout/stderr are empty; we ignore them entirely
        # and rely on the return code + output file. 10 min cap for big meshes.
        rc = _launch(cmd, cwd, timeout=600)
    except subprocess.TimeoutExpired:
        return SliceResult(False, out_str, -1,
                           "bambu-studio timed out (>600s)\n" + cmd_str)
    except OSError as e:
        return SliceResult(False, out_str, -1,
                           f"failed to launch bambu-studio: {e}\n" + cmd_str)
    finally:
        _release_lock(lp)

    produced = out_3mf.exists() and out_3mf.stat().st_size > 0

    if rc == _OK_RETURNCODE and produced:
        size = out_3mf.stat().st_size
        return SliceResult(True, out_str, rc,
                           f"sliced OK -> {out_3mf.name} ({size} bytes)")

    # Failure: surface the exact command so it can be retried / debugged by hand.
    reason = (f"slice failed (returncode={rc}, output_produced={produced}); "
              "no stdout on Windows GUI binary to inspect")
    return SliceResult(False, out_str, rc, reason + "\n" + cmd_str)


def _smoke_make_3mf(stl_path: Path, out_3mf: Path) -> None:
    """Build a minimal Bambu-loadable .3mf from a mesh via lib3mf (mirrors the
    geometry path that assemble3mf produces, for a self-contained smoke test)."""
    import lib3mf
    import trimesh

    # process=True merges duplicate vertices into a clean manifold; an unmerged
    # soup makes Bambu reject the load (exit -100).
    mesh = trimesh.load(str(stl_path), force="mesh", process=True)
    w = lib3mf.Wrapper()
    model = w.CreateModel()
    obj = model.AddMeshObject()
    obj.SetName(stl_path.stem)
    positions = []
    for v in mesh.vertices:
        p = lib3mf.Position()
        p.Coordinates = (float(v[0]), float(v[1]), float(v[2]))
        positions.append(p)
    triangles = []
    for f in mesh.faces:
        t = lib3mf.Triangle()
        t.Indices = (int(f[0]), int(f[1]), int(f[2]))
        triangles.append(t)
    obj.SetGeometry(positions, triangles)
    model.AddBuildItem(obj, w.GetIdentityTransform())
    model.QueryWriter("3mf").WriteToFile(str(out_3mf))


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]  # project root (…/meshforge)
    stl = root / "outputs" / "test_blob_repaired.stl"
    in_3mf = root / "outputs" / "_slicer_smoke_in.3mf"
    out_3mf = root / "outputs" / "_slicer_smoke_out.gcode.3mf"

    print(f"smoke: building input .3mf from {stl.name} ...")
    _smoke_make_3mf(stl, in_3mf)

    print("smoke: slicing via Bambu Studio CLI ...")
    res = slice_project(in_3mf, out_3mf)
    print(json.dumps(res.as_dict(), indent=2))

    # Clean up the transient input on success; keep it on failure for debugging.
    if res.ok:
        try:
            in_3mf.unlink()
        except OSError:
            pass
