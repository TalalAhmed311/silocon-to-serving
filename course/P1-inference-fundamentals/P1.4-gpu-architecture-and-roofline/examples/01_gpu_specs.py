"""01_gpu_specs.py — validate gpu_specs.yaml and print each GPU's ridge point per precision.

Run:      uv run python course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/examples/01_gpu_specs.py
Expected: the ridge table (computed from the YAML) and a warning per UNVERIFIED / missing value.
Hardware: T0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import specs  # noqa: E402

gpus = specs.load()
print("| GPU | GB/s | fp16 dense TF | ridge fp16 | fp8 dense TF | ridge fp8 | status |\n|---|---|---|---|---|---|---|")
for g in gpus:
    r16, r8 = specs.ridge(g, "fp16"), specs.ridge(g, "fp8")
    fmt = lambda v: "—" if v is None else f"{v:.0f}"  # noqa: E731
    print(f"| {g['name']} | {g['hbm_gbs'] or '—'} | {fmt(g['fp16_dense_tflops'])} | {fmt(r16)} | {fmt(g['fp8_dense_tflops'])} | {fmt(r8)} | {g['status']} |")
print()
for g in gpus:
    if g["status"] == "UNVERIFIED":
        print(f"WARNING {g['name']}: UNVERIFIED — check against: {g['source']}")
    if g.get("todo"):
        print(f"TODO    {g['name']}: {g['todo']}")
