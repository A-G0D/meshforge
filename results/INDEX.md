# Results index

Every number in the README comes from one file here.

| README number | File | Command |
|---|---|---|
| Repair success 97.2% (35/36), naive baseline 36.1%, median 0.04 s per mesh | `bench_printprep.json` (`summary`, `repair` rows) | `python scripts/bench_printprep.py` |
| Support-area fraction 0.123 to 0.094, median orient 0.2 s | `bench_printprep.json` (`orient` rows) | same |
| Before/after render | `../docs/repair_before_after.png` | `python scripts/render_repair.py` |

Machine: arm64 macOS, Python 3.9.6, trimesh 4.12.2. Timings are wall clock on a busy laptop and move run to run; the success rates are seeded and repeat.
