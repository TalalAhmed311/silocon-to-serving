"""sgemm_bench.py — run the D1 ladder bench with the peak measured by examples/02_fma_peak.

Run:      uv run python bench/sgemm_bench.py         (from the module folder, after building examples and bench)
Output:   the C++ table (with % of measured all-core peak) and results/sgemm.json.
Hardware: T0.
"""
import json
import subprocess
from pathlib import Path

peak_file = Path("results/peak.json")
if not peak_file.exists():
    subprocess.run(["./build/examples/02_fma_peak"], check=True)
peak = json.loads(peak_file.read_text())["all_cores"]
print(f"using measured peak {peak:.1f} GFLOP/s (all cores)")
subprocess.run(["./build/bench/sgemm_bench", f"{peak}"], check=True)
