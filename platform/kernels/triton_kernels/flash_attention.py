"""FlashAttention-2 forward in Triton: one program per (query block, batch·head); K/V blocks stream through; online
softmax state (m_i, l_i, acc) in registers; tl.dot puts both matmuls on tensor cores. q, k, v: [B, H, N, D] fp16/bf16.
Compare with d4/attention.cuh's CUDA-core FA-2: same algorithm, and the matmuls are the difference."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fa2_fwd(Q, K, V, O, sqz, sqh, sqm, sqd, skz, skh, skn, skd, svz, svh, svn, svd, soz, soh, som, sod,
             H, N, scale, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    z, h = off_hz // H, off_hz % H
    offs_m = start_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + z * sqz + h * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd, mask=offs_m[:, None] < N, other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    if CAUSAL:
        hi = (start_m + 1) * BM          # K/V blocks after this query block are fully masked: never visited
    else:
        hi = N
    for start_n in range(0, hi, BN):
        cols = start_n + offs_n
        k = tl.load(K + z * skz + h * skh + cols[None, :] * skn + offs_d[:, None] * skd, mask=cols[None, :] < N, other=0.0)
        s = tl.dot(q, k) * scale                                         # [BM, BN]
        valid = cols[None, :] < N
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= cols[None, :])
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + z * svz + h * svh + cols[:, None] * svn + offs_d[None, :] * svd, mask=cols[:, None] < N, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + z * soz + h * soh + offs_m[:, None] * som + offs_d[None, :] * sod, acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < N)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, BM: int = 64, BN: int = 32) -> torch.Tensor:
    B, H, N, D = q.shape
    assert D in (16, 32, 64, 128)
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BM), B * H)
    _fa2_fwd[grid](q, k, v, o, *q.stride(), *k.stride(), *v.stride(), *o.stride(), H, N, 1.0 / math.sqrt(D),
                   BM=BM, BN=BN, D=D, CAUSAL=causal)
    return o
