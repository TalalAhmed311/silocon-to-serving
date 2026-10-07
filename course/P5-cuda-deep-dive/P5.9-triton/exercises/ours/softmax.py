"""Exercise 1: softmax in Triton. Same API as platform/kernels/triton_kernels/softmax.py: softmax(x[rows, cols])."""
import torch
import triton  # noqa: F401
import triton.language as tl  # noqa: F401


def softmax(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO: one program per row; tl.load with mask/other=-inf; max, exp, sum, store")
