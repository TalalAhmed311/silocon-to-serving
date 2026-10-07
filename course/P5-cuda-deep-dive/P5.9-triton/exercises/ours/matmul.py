"""Exercise 3: matmul in Triton. API: matmul(a[M,K], b[K,N], out_dtype=torch.float32, autotune=True) → c[M,N].
Requirements: masks for any M, N, K; fp32 inputs must not silently use TF32 (input_precision="ieee")."""
import torch


def matmul(a: torch.Tensor, b: torch.Tensor, out_dtype=torch.float32, autotune: bool = True) -> torch.Tensor:
    raise NotImplementedError("TODO")
