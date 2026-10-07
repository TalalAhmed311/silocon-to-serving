"""Exercise 4: FlashAttention-2 forward in Triton. API: attention(q, k, v, causal=False) with q, k, v [B, H, N, D]."""
import torch


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    raise NotImplementedError("TODO")
