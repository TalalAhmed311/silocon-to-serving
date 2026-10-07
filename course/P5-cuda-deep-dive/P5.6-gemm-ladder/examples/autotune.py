"""autotune.py — sweep rung 7's tile parameters and keep the fastest valid configuration for YOUR GPU (T2).

Rung 7 is a template: sgemm_r7<BM, BN, BK, WM, WN, TM, TN>. This script writes a small .cu that instantiates each
candidate, builds it with nvcc, runs it at N = 4096, and prints a ranked table + the winner as JSON. Candidates that
violate rung 7's static_asserts are filtered here first (same rules), so every build succeeds.

Run on the GPU box: uv run python course/P5-cuda-deep-dive/P5.6-gemm-ladder/examples/autotune.py --arch native
Expected: a few configs within ~5% of each other at the top; the default (128,128,8,64,32,8,8) near the top on most
GPUs; very different winners on T4 vs L4 are a good discussion point (smem size, register file).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TEMPLATE = r"""
#include <cstdio>
#include <d4/gemm.cuh>
#include "s2s_cuda.cuh"
int main() {
  const int n = %(N)d;
  s2s::DeviceBuffer<float> A(s2s::random_vec<float>(size_t(n) * n)), B(s2s::random_vec<float>(size_t(n) * n)), C(size_t(n) * n);
  auto t = s2s::time_gpu([&] {
    d4::sgemm_r7<%(BM)d, %(BN)d, %(BK)d, %(WM)d, %(WN)d, %(TM)d, %(TN)d>
        <<<dim3(n / %(BN)d, n / %(BM)d), (%(BM)d * %(BN)d) / (%(TM)d * %(TN)d)>>>(n, n, n, A.get(), B.get(), C.get());
  }, 3, 20);
  std::printf("%%f\n", 2.0 * n * n * n / (t.median_ms * 1e9));
}
"""


def valid(BM, BN, BK, WM, WN, TM, TN):
    nt = BM * BN // (TM * TN)
    return (nt % 32 == 0 and nt <= 1024 and BM % WM == 0 and BN % WN == 0 and (BM // WM) * (BN // WN) == nt // 32
            and (WM // TM) * (WN // TN) == 32 and BM * BK == 4 * nt and BK * BN == 4 * nt and TN % 4 == 0 and TM % 4 == 0
            and 2 * 4 * BK * (BM + BN) <= 48 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="native")
    ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()
    space = itertools.product([64, 128, 256], [64, 128, 256], [8, 16], [32, 64, 128], [32, 64, 128], [4, 8], [4, 8])
    cands = [c for c in space if valid(*c)]
    print(f"{len(cands)} valid configurations")
    results = []
    with tempfile.TemporaryDirectory() as d:
        for c in cands:
            BM, BN, BK, WM, WN, TM, TN = c
            src = Path(d) / "t.cu"
            src.write_text(TEMPLATE % dict(N=a.n, BM=BM, BN=BN, BK=BK, WM=WM, WN=WN, TM=TM, TN=TN))
            exe = Path(d) / "t"
            cmd = ["nvcc", "-O3", "-std=c++17", f"-arch={a.arch}", "-o", str(exe), str(src),
                   f"-I{ROOT / 'platform/kernels/include'}", f"-I{ROOT / 'laneB-cuda/harness/include'}",
                   f"-I{ROOT / 'course/common/include'}", "-lcublas"]
            if subprocess.run(cmd, capture_output=True).returncode != 0:
                print("build failed:", c)
                continue
            out = subprocess.run([str(exe)], capture_output=True, text=True, env={**os.environ})
            try:
                results.append((float(out.stdout.strip()), c))
            except ValueError:
                print("run failed:", c, out.stderr[-200:])
    results.sort(reverse=True)
    print("| TFLOP/s | BM BN BK WM WN TM TN |\n|---|---|")
    for tf, c in results[:15]:
        print(f"| {tf:.2f} | {' '.join(map(str, c))} |")
    if results:
        best = dict(zip(["BM", "BN", "BK", "WM", "WN", "TM", "TN"], results[0][1]), tflops=results[0][0], n=a.n)
        os.makedirs("results", exist_ok=True)
        Path("results/gemm_autotune.json").write_text(json.dumps(best, indent=2))
        print("best:", best)


if __name__ == "__main__":
    main()
