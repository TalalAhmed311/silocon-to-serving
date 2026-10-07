"""#12 / P5.10 exercises 1–2: the custom op passes torch.library.opcheck and is torch.compile-safe (T0 on CPU via the
reference implementation; the CUDA tests run when a GPU is present)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # platform/kernels → torch_ext
import torch_ext.ops  # noqa: E402,F401

op = torch.ops.s2s.fused_add_rms_norm


def ref(x, r, w, eps):
    s = (x.float() + r.float()).to(r.dtype)
    sf = s.float()
    return (sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype), s


def make(device, dtype, T=7, H=256):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(T, H, generator=g).to(device, dtype)
    r = torch.randn(T, H, generator=g).to(device, dtype)
    w = (torch.rand(H, generator=g) + 0.5).to(device, dtype)
    return x, r, w


def test_semantics_cpu():
    x, r, w = make("cpu", torch.float32)
    want_x, want_r = ref(x, r, w, 1e-6)
    assert op(x, r, w, 1e-6) is None
    torch.testing.assert_close(r, want_r)
    torch.testing.assert_close(x, want_x)


def test_opcheck_cpu():
    x, r, w = make("cpu", torch.float32)
    torch.library.opcheck(op, (x, r, w, 1e-6))          # schema, fake impl, mutation declarations, aot dispatch


def test_torch_compile_fullgraph_cpu():
    def block(h, res, w):
        torch.ops.s2s.fused_add_rms_norm(h, res, w, 1e-6)
        return h * 2.0
    x, r, w = make("cpu", torch.float32)
    x2, r2 = x.clone(), r.clone()
    eager_out = block(x, r, w)
    compiled = torch.compile(block, fullgraph=True)       # graph break → error, not silent fallback
    torch.testing.assert_close(compiled(x2, r2, w), eager_out)
    torch.testing.assert_close(r2, r)


@pytest.mark.gpu
@pytest.mark.parametrize("impl", ["cuda", "triton"])
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 1e-2), (torch.float16, 2e-3)])
def test_cuda_impls(monkeypatch, impl, dtype, tol):
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    monkeypatch.setenv("S2S_RMSNORM_IMPL", impl)
    x, r, w = make("cuda", dtype, T=33, H=4096)
    want_x, want_r = ref(x, r, w, 1e-6)
    op(x, r, w, 1e-6)
    torch.testing.assert_close(r, want_r, rtol=0, atol=0)      # same fp32 add, same rounding: exact
    torch.testing.assert_close(x, want_x, rtol=tol, atol=tol)
    torch.library.opcheck(op, make("cuda", dtype, T=4, H=128) + (1e-6,))
