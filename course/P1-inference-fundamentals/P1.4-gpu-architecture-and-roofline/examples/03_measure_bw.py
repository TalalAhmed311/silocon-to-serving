"""03_measure_bw.py — achieved device-to-device copy bandwidth with PyTorch vs the spec in gpu_specs.yaml (T2).

Run:      uv run --extra torch python course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/examples/03_measure_bw.py [--gpu L4]
Output:   | size | copy GB/s | % of spec | for buffers from 1 MiB to 1 GiB (CUDA events, median of 20).
Hardware: T2 (any NVIDIA GPU). Exits with a message on machines without CUDA.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import specs  # noqa: E402

if not torch.cuda.is_available():
    sys.exit("no CUDA device: this example is T2 (run on a GPU instance, e.g. g6.xlarge)")
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", help="name in gpu_specs.yaml (default: guess from torch.cuda.get_device_name)")
a = ap.parse_args()
name = torch.cuda.get_device_name()
spec = next((g for g in specs.load() if (a.gpu or "") == g["name"] or g["name"].split("-")[0] in name), None)
print(f"device: {name}; spec entry: {spec['name'] if spec else 'none'} ({spec['status'] if spec else ''})\n")
print("| size | copy GB/s | % of spec |\n|---|---|---|")
for mib in (1, 16, 64, 256, 1024):
    n = mib * 2**20 // 4
    src, dst = torch.empty(n, device="cuda"), torch.empty(n, device="cuda")
    for _ in range(5):
        dst.copy_(src)
    times = []
    for _ in range(20):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); dst.copy_(src); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    ms = sorted(times)[10]
    gbs = 2 * n * 4 / (ms * 1e6)  # read + write
    pct = f"{100 * gbs / spec['hbm_gbs']:.0f}%" if spec and spec["hbm_gbs"] else "—"
    print(f"| {mib} MiB | {gbs:.0f} | {pct} |")
print("\nSmall buffers fit in L2 (or are launch-bound) — only the large rows measure DRAM/HBM.")
