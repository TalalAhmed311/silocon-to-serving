"""bench_triton_vs_cuda.py — the P5.9 comparison table: Triton (ours) vs PyTorch on the same shapes as the D4 CUDA benches
(T2). CUDA numbers come from the P5.5/P5.6/P5.8 bench outputs (results/*.jsonl); this script prints the Triton and
torch rows and, if those files exist, the matching CUDA rows next to them.

Run: uv run python course/P5-cuda-deep-dive/P5.9-triton/examples/bench_triton_vs_cuda.py
"""
import json
from pathlib import Path

import torch
import triton

from kernels.triton_kernels.flash_attention import attention
from kernels.triton_kernels.matmul import matmul
from kernels.triton_kernels.rmsnorm import rmsnorm
from kernels.triton_kernels.softmax import softmax


def ms(fn):
    return triton.testing.do_bench(fn, warmup=10, rep=50)


def main():
    dev = "cuda"
    rows = []
    x = torch.randn(65536, 1024, device=dev)
    rows.append(("softmax 65536×1024 fp32", "GB/s", 8 * x.numel() / ms(lambda: softmax(x)) / 1e6,
                 8 * x.numel() / ms(lambda: torch.softmax(x, -1)) / 1e6))
    x, w = torch.randn(16384, 4096, device=dev), torch.randn(4096, device=dev)
    rows.append(("rmsnorm 16384×4096 fp32", "GB/s", 8 * x.numel() / ms(lambda: rmsnorm(x, w)) / 1e6,
                 8 * x.numel() / ms(lambda: torch.nn.functional.rms_norm(x, (4096,), w)) / 1e6))
    for dt in (torch.float32, torch.float16):
        a, b = torch.randn(4096, 4096, device=dev, dtype=dt), torch.randn(4096, 4096, device=dev, dtype=dt)
        f = 2 * 4096**3
        rows.append((f"matmul 4096³ {dt}", "TFLOP/s", f / ms(lambda: matmul(a, b)) / 1e9, f / ms(lambda: a @ b) / 1e9))
    q, k, v = (torch.randn(1, 16, 4096, 64, device=dev, dtype=torch.float16) for _ in range(3))
    f = 4 * 16 * 4096 * 4096 * 64
    rows.append(("FA-2 fwd N=4096 BH=16", "TFLOP/s", f / ms(lambda: attention(q, k, v)) / 1e9,
                 f / ms(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)) / 1e9))
    print("| kernel | unit | Triton (ours) | PyTorch |\n|---|---|---|---|")
    for name, unit, a, b in rows:
        print(f"| {name} | {unit} | {a:.1f} | {b:.1f} |")
    for f in ("softmax", "rmsnorm", "gemm", "hgemm"):
        p = Path(f"results/{f}.jsonl")
        if p.exists():
            print(f"\nCUDA (D4) rows from {p}:")
            for line in p.read_text().splitlines()[-6:]:
                d = json.loads(line)
                print(f"  {d['label']}: {d['rate']:.1f} {d['unit']}")


if __name__ == "__main__":
    main()
