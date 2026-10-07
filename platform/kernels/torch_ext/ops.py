"""ops.py — registers torch.ops.s2s.fused_add_rms_norm (#12).

    import torch_ext.ops  # PYTHONPATH=platform/kernels   (or `from kernels.torch_ext import ops` with PYTHONPATH=platform)
    torch.ops.s2s.fused_add_rms_norm(x, residual, weight, 1e-6)   # updates x and residual in place, returns None

Why a registered custom op (torch.library.custom_op) and not a plain Python function: torch.compile and CUDA-graph
capture (vLLM uses both) treat a registered op as an opaque node with a declared schema (which args it mutates) and a
fake/meta implementation for shape propagation — a plain function would be traced into and could break the graph.

Implementations:
  default (any device) : PyTorch reference — also what opcheck exercises on CPU (T0)
  cuda                 : S2S_RMSNORM_IMPL=cuda (default) → JIT-built csrc/s2s_ext.cu; =triton → triton_kernels.rmsnorm
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ext = None


def _cuda_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        _ext = load(name="s2s_ext", sources=[str(_HERE / "csrc" / "s2s_ext.cu")], extra_cuda_cflags=["-O3"], verbose=False)
    return _ext


@torch.library.custom_op("s2s::fused_add_rms_norm", mutates_args=("x", "residual"))
def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    s = (x.float() + residual.float()).to(residual.dtype)
    residual.copy_(s)
    sf = s.float()
    x.copy_((sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype))


@fused_add_rms_norm.register_kernel("cuda")
def _fused_add_rms_norm_cuda(x, residual, weight, eps):
    impl = os.environ.get("S2S_RMSNORM_IMPL", "cuda")
    if impl == "triton":
        from kernels.triton_kernels.rmsnorm import rmsnorm
        x2, r2 = x.view(-1, x.shape[-1]), residual.view(-1, residual.shape[-1])
        x2.copy_(rmsnorm(x2, weight, eps, residual=r2))
    else:
        _cuda_ext().fused_add_rms_norm(x, residual, weight, eps)


@fused_add_rms_norm.register_fake
def _fused_add_rms_norm_fake(x, residual, weight, eps):
    return None
