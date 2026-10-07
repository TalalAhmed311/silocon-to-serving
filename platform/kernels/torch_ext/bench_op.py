"""bench_op.py — #12 microbenchmark: our op (cuda, triton) vs vLLM's own fused_add_rms_norm (if vLLM is installed) vs
eager PyTorch, at serving shapes (num_tokens × hidden). T2.

Run: PYTHONPATH=platform:platform/kernels uv run python platform/kernels/torch_ext/bench_op.py
"""
import os

import torch
import triton

import torch_ext.ops  # noqa: F401  (registers torch.ops.s2s.*)


def eager(x, r, w, eps):
    s = x + r
    r.copy_(s)
    sf = s.float()
    x.copy_((sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w)


def main():
    hidden, dt = 4096, torch.bfloat16
    try:
        from vllm import _custom_ops as vops
    except Exception:  # vLLM not installed or no GPU build
        vops = None
    print("| tokens | impl | µs | GB/s |\n|---|---|---|---|")
    for T in (1, 32, 256, 4096):
        x, r = torch.randn(T, hidden, device="cuda", dtype=dt), torch.randn(T, hidden, device="cuda", dtype=dt)
        w = torch.randn(hidden, device="cuda", dtype=dt)
        nbytes = 4 * T * hidden * x.element_size()                       # read x, r; write r, x
        impls = {"eager": lambda: eager(x, r, w, 1e-6)}
        for name in ("cuda", "triton"):
            def f(name=name):
                os.environ["S2S_RMSNORM_IMPL"] = name
                torch.ops.s2s.fused_add_rms_norm(x, r, w, 1e-6)
            impls[f"ours ({name})"] = f
        if vops is not None:
            impls["vLLM"] = lambda: vops.fused_add_rms_norm(x, r, w, 1e-6)
        for name, fn in impls.items():
            ms = triton.testing.do_bench(fn, warmup=20, rep=100)
            print(f"| {T} | {name} | {ms * 1e3:.1f} | {nbytes / (ms * 1e6):.0f} |")


if __name__ == "__main__":
    main()
