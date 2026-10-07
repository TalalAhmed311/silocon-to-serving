"""hf_models.py — wrap Hugging Face causal LMs as specdec Models (T2 for real sizes; T0 with tiny random configs).

probs_many runs the target on prefix + proposals in ONE forward pass and reads the k+1 distributions from the last k+1
positions — the reason speculative decoding is fast: a k+1-token mini-prefill costs ~one decode step when memory-bound.
This prototype recomputes the full prefix each call (no KV cache) for clarity; exercise 3 adds a cache.
"""
from __future__ import annotations

import numpy as np
import torch

from .core import Model


class HFModel(Model):
    def __init__(self, model, temperature: float = 1.0):
        self.m, self.T = model.eval(), temperature

    @torch.no_grad()
    def _logits(self, ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], device=next(self.m.parameters()).device)
        return self.m(x).logits[0].float()

    def _dist(self, logits: torch.Tensor) -> np.ndarray:
        p = torch.softmax(logits / self.T, dim=-1).double().cpu().numpy()
        return p / p.sum(axis=-1, keepdims=True)

    def probs(self, prefix):
        return self._dist(self._logits(prefix)[-1])

    def probs_many(self, prefix, proposals):
        lg = self._logits(prefix + proposals)
        return self._dist(lg[len(prefix) - 1:])          # positions predicting proposals[0..k-1] and the bonus
