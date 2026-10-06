"""Exercise 3 solution."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import specs  # noqa: E402


def pct_of_peak(measured, gpu_name, kind, precision="fp16"):
    g = specs.get(gpu_name)
    peak = g["hbm_gbs"] if kind == "bandwidth" else g.get(f"{precision}_dense_tflops")
    return None if not peak else 100.0 * measured / peak


def annotate(rate, gpu_name, kind, precision="fp16"):
    p = pct_of_peak(rate, gpu_name, kind, precision)
    if p is None:
        return f"{rate:.1f} (no spec)"
    status = specs.get(gpu_name)["status"]
    return f"{rate:.1f} ({p:.0f}% of {'UNVERIFIED ' if status == 'UNVERIFIED' else ''}peak)"
