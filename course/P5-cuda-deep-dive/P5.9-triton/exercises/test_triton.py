"""P5.9 exercises 1–4: Triton rewrites vs PyTorch references. T0: Triton's interpreter on CPU (set automatically when
no GPU is present). T2: the same tests on a real GPU. S2S_SOLUTIONS=1 tests platform/kernels/triton_kernels."""
import importlib
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")        # must be set before triton is imported
pytest.importorskip("triton")
pytestmark = pytest.mark.torch

HERE = Path(__file__).resolve().parent
if os.environ.get("S2S_SOLUTIONS") == "1":
    PKG = "kernels.triton_kernels"
else:
    sys.path.insert(0, str(HERE))
    PKG = "ours"
softmax = importlib.import_module(f"{PKG}.softmax").softmax
rmsnorm = importlib.import_module(f"{PKG}.rmsnorm").rmsnorm
matmul = importlib.import_module(f"{PKG}.matmul").matmul
attention = importlib.import_module(f"{PKG}.flash_attention").attention

DEV = "cuda" if torch.cuda.is_available() else "cpu"
gen = torch.Generator().manual_seed(0)


def rnd(*shape, dtype=torch.float32, lo=-1.0, hi=1.0):
    return (torch.rand(*shape, generator=gen) * (hi - lo) + lo).to(dtype).to(DEV)


@pytest.mark.parametrize("rows,cols", [(1, 1), (4, 33), (8, 1024), (2, 20000)])
def test_softmax(rows, cols):
    x = rnd(rows, cols, lo=-20, hi=20)
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 1e-2)])
def test_rmsnorm(dtype, tol):
    x, w = rnd(16, 1000, dtype=dtype, lo=-3, hi=3), rnd(1000, dtype=dtype, lo=0.5, hi=1.5)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(dtype)
    torch.testing.assert_close(rmsnorm(x, w), ref, rtol=tol, atol=tol)
    r = rnd(16, 1000, dtype=dtype)
    r_ref = (x.float() + r.float()).to(dtype)
    y = rmsnorm(x, w, residual=r)
    torch.testing.assert_close(r, r_ref, rtol=0, atol=0)              # residual updated in place, exactly
    ref2 = (r_ref.float() * torch.rsqrt(r_ref.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(dtype)
    torch.testing.assert_close(y, ref2, rtol=tol, atol=tol)


@pytest.mark.parametrize("M,N,K", [(32, 32, 16), (65, 47, 70), (128, 96, 33)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matmul(M, N, K, dtype):
    a, b = rnd(M, K, dtype=dtype), rnd(K, N, dtype=dtype)
    ref = a.float() @ b.float()
    tol = 1e-4 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(matmul(a, b, autotune=False), ref, rtol=tol, atol=tol)


@pytest.mark.parametrize("N,causal", [(64, False), (100, False), (96, True), (130, True)])
def test_flash_attention(N, causal):
    q, k, v = (rnd(1, 2, N, 64, dtype=torch.float16) for _ in range(3))
    s = (q.float() @ k.float().transpose(-1, -2)) / 8.0
    if causal:
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=DEV).triu(1), float("-inf"))
    ref = torch.softmax(s, -1) @ v.float()
    torch.testing.assert_close(attention(q, k, v, causal=causal).float(), ref, rtol=2e-3, atol=2e-3)
