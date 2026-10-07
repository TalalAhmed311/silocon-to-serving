"""Exercise 2: RMSNorm in Triton. API: rmsnorm(x[rows, cols], w[cols], eps=1e-6, residual=None) → y; with `residual`,
update it in place to x + residual and normalise that (see platform/kernels/triton_kernels/rmsnorm.py once done)."""
import torch


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, residual: torch.Tensor | None = None) -> torch.Tensor:
    raise NotImplementedError("TODO")
