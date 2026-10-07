"""Row-wise softmax in Triton: one program per row. Short rows: the whole row in one block (1 read, 1 write).
Long rows (> MAX_BLOCK): online max/sum loop (2 reads, 1 write) — the same ladder as P5.5, in ~30 lines."""
import torch
import triton
import triton.language as tl

MAX_BLOCK = 16384


@triton.jit
def _softmax_row(x_ptr, y_ptr, n_cols, x_stride, y_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(y_ptr + row * y_stride + offs, (num / tl.sum(num, axis=0)).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_online(x_ptr, y_ptr, n_cols, x_stride, y_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)     # scalar −inf of the right type
    d = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        x = tl.load(x_ptr + row * x_stride + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        x = tl.load(x_ptr + row * x_stride + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        tl.store(y_ptr + row * y_stride + cols, (tl.exp(x - m) / d).to(y_ptr.dtype.element_ty), mask=cols < n_cols)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.stride(1) == 1
    rows, cols = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    if block <= MAX_BLOCK:
        _softmax_row[(rows,)](x, y, cols, x.stride(0), y.stride(0), BLOCK=block)
    else:
        _softmax_online[(rows,)](x, y, cols, x.stride(0), y.stride(0), BLOCK=4096)
    return y
