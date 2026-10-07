"""Matmul in Triton: block tiles, tl.dot (tensor cores for fp16/bf16), grouped program ordering for L2 reuse, masks
for any shape, and an autotune list. fp32 inputs use IEEE fp32 dot (no TF32) so comparisons with SGEMM are fair."""
import torch
import triton
import triton.language as tl

CONFIGS = [
    triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=2, num_stages=5),
]


@triton.jit
def _matmul(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, IEEE: tl.constexpr):
    pid = tl.program_id(0)
    num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    per_group = GROUP_M * num_n                          # consecutive programs share A row-panels → L2 hits
    group = pid // per_group
    first_m = group * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % per_group) % size_m
    pid_n = (pid % per_group) // size_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kr = K - k * BK
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < kr), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < kr) & (rn[None, :] < N), other=0.0)
        if IEEE:
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(c_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


_autotuned = triton.autotune(configs=CONFIGS, key=["M", "N", "K"])(_matmul)


def matmul(a: torch.Tensor, b: torch.Tensor, out_dtype=torch.float32, autotune: bool = True) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    ieee = a.dtype == torch.float32
    args = (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    if autotune:
        grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)  # noqa: E731
        _autotuned[grid](*args, IEEE=ieee)
    else:                                                  # fixed config: interpreter-friendly, deterministic tests
        BM = BN = 32
        _matmul[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](*args, BM=BM, BN=BN, BK=16, GROUP_M=4, IEEE=ieee)
    return c
