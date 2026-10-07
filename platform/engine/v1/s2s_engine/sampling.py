"""sampling.py — logits → next tokens (P6.5): temperature, top-k, top-p, min-p, seeded. NumPy reference (T0) and a
batched PyTorch version for the GPU that avoids a full vocabulary sort for top-p (threshold binary search).
"""
from __future__ import annotations

import numpy as np

from .sequence import SamplingParams


def _softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def filter_probs_np(logits: np.ndarray, p: SamplingParams) -> np.ndarray:
    """Probability vector after temperature, top-k, top-p, min-p (renormalised). logits: [V]."""
    x = logits.astype(np.float64) / max(p.temperature, 1e-6)
    probs = _softmax(x)
    if p.top_k > 0 and p.top_k < probs.size:
        kth = np.partition(probs, -p.top_k)[-p.top_k]
        probs = np.where(probs >= kth, probs, 0.0)
    if p.min_p > 0.0:
        probs = np.where(probs >= p.min_p * probs.max(), probs, 0.0)
    if p.top_p < 1.0:
        probs = probs / probs.sum()
        tau = top_p_threshold_np(probs, p.top_p)
        probs = np.where(probs >= tau, probs, 0.0)
    return probs / probs.sum()


def top_p_threshold_np(probs: np.ndarray, top_p: float, iters: int = 40) -> float:
    """Largest τ such that the mass of {p_i ≥ τ} is still ≥ top_p — found by bisection, no sort (P6.5 §2)."""
    lo, hi = 0.0, float(probs.max())
    for _ in range(iters):
        mid = (lo + hi) / 2
        if probs[probs >= mid].sum() >= top_p:
            lo = mid                              # mass still enough: try a higher threshold
        else:
            hi = mid
    return lo


def top_p_threshold_sorted(probs: np.ndarray, top_p: float) -> float:
    """Reference via sort: the smallest probability in the minimal set whose cumulative mass reaches top_p."""
    s = np.sort(probs)[::-1]
    c = np.cumsum(s)
    k = int(np.searchsorted(c, top_p - 1e-12)) + 1
    return float(s[min(k, s.size) - 1])


def sample_np(logits: np.ndarray, p: SamplingParams, rng: np.random.Generator) -> int:
    if p.temperature <= 0.0:
        return int(np.argmax(logits))
    probs = filter_probs_np(logits, p)
    return int(rng.choice(probs.size, p=probs))


# ---- PyTorch, batched (GPU) --------------------------------------------------------------------------------------
def sample_torch(logits, temperature, top_k, top_p, min_p, generator=None):
    """logits [B, V] (any float dtype, on any device); per-row parameter tensors [B]. Returns token ids [B].
    Greedy rows (temperature == 0) use argmax. Top-p uses a 30-step threshold bisection: no [B, V] sort."""
    import torch
    x = logits.float()
    greedy = temperature <= 0
    t = torch.where(greedy, torch.ones_like(temperature), temperature).float()
    probs = torch.softmax(x / t[:, None], dim=-1)
    if (top_k > 0).any():
        k = torch.where(top_k > 0, top_k, torch.full_like(top_k, probs.shape[-1])).clamp(max=probs.shape[-1])
        kth = torch.topk(probs, int(k.max()), dim=-1).values.gather(-1, (k - 1)[:, None].long())
        probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
    if (min_p > 0).any():
        probs = torch.where(probs >= (min_p[:, None] * probs.amax(-1, keepdim=True)), probs, torch.zeros_like(probs))
    if (top_p < 1).any():
        probs = probs / probs.sum(-1, keepdim=True)
        lo = torch.zeros_like(top_p, dtype=torch.float32)
        hi = probs.amax(-1)
        for _ in range(30):
            mid = (lo + hi) / 2
            mass = torch.where(probs >= mid[:, None], probs, torch.zeros_like(probs)).sum(-1)
            ok = mass >= top_p
            lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
        keep = (probs >= lo[:, None]) | (top_p[:, None] >= 1)
        probs = torch.where(keep, probs, torch.zeros_like(probs))
    probs = probs / probs.sum(-1, keepdim=True)
    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return torch.where(greedy, x.argmax(-1), sampled)
