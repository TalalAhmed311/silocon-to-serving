"""RMSNorm (+ fused residual add) in Triton: one program per row, fp32 accumulation, any storage dtype."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x_ptr, r_ptr, w_ptr, y_ptr, stride, n_cols, eps, HAS_RESID: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK], tl.float32)
    for off in range(0, n_cols, BLOCK):                       # pass 1: Σx² (and the residual add, stored back)
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESID:
            x = x + tl.load(r_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
            xs = x.to(r_ptr.dtype.element_ty)
            tl.store(r_ptr + row * stride + cols, xs, mask=mask)
            x = xs.to(tl.float32)                             # normalise what was stored (matches the unfused path)
        acc += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)
    for off in range(0, n_cols, BLOCK):                       # pass 2: scale
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        if HAS_RESID:
            x = tl.load(r_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * stride + cols, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, residual: torch.Tensor | None = None) -> torch.Tensor:
    """y = rmsnorm(x [+ residual]) · w. With `residual`, it is updated IN PLACE to x + residual (vLLM's convention)."""
    assert x.is_contiguous() and x.dim() == 2
    rows, cols = x.shape
    y = torch.empty_like(x)
    block = min(triton.next_power_of_2(cols), 4096)
    _rmsnorm[(rows,)](x, residual if residual is not None else x, w, y, x.stride(0), cols, eps,
                      HAS_RESID=residual is not None, BLOCK=block)
    return y
