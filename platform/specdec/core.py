"""core.py — speculative sampling (Leviathan et al. 2211.17192; Chen et al. 2302.01318), model-agnostic.

A "model" here is anything with `probs(prefix: list[int]) -> np.ndarray[V]` (next-token distribution after `prefix`)
and, for the target, `probs_many(prefix, proposals) -> np.ndarray[k+1, V]` (distributions after prefix, prefix+p1, …,
prefix+p1..pk) — ONE forward pass in a real engine. Default probs_many just loops (correct, not fast).

Guarantee: the emitted tokens are distributed exactly as if sampled from the target alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Model:
    def probs(self, prefix: list[int]) -> np.ndarray:
        raise NotImplementedError

    def probs_many(self, prefix: list[int], proposals: list[int]) -> np.ndarray:
        return np.stack([self.probs(prefix + proposals[:i]) for i in range(len(proposals) + 1)])


@dataclass
class Stats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    target_calls: int = 0
    per_round_accepted: list = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(1, self.proposed)

    @property
    def tokens_per_target_call(self) -> float:
        return self.emitted / max(1, self.target_calls)


def speculative_step(prefix, draft: Model, target: Model, k: int, rng: np.random.Generator, stats: Stats) -> list[int]:
    """One round: draft proposes k tokens, target verifies all of them in one call. Returns the accepted tokens
    plus exactly one more (a resampled correction, or a bonus token if all k were accepted)."""
    proposals, q = [], []
    ctx = list(prefix)
    for _ in range(k):                                   # k cheap draft steps
        qd = draft.probs(ctx)
        t = int(rng.choice(len(qd), p=qd))
        proposals.append(t)
        q.append(qd)
        ctx.append(t)
    p = target.probs_many(list(prefix), proposals)       # [k+1, V] in ONE target call
    stats.target_calls += 1
    stats.rounds += 1
    stats.proposed += k
    out = []
    for i, t in enumerate(proposals):
        if rng.random() < min(1.0, p[i][t] / q[i][t]):   # accept with prob min(1, p/q)
            out.append(t)
            continue
        residual = np.maximum(p[i] - q[i], 0.0)          # reject: resample from norm(max(0, p − q))
        residual /= residual.sum()
        out.append(int(rng.choice(len(residual), p=residual)))
        stats.accepted += i
        stats.per_round_accepted.append(i)
        stats.emitted += len(out)
        return out
    out.append(int(rng.choice(len(p[k]), p=p[k])))       # all accepted: free bonus token from the last position
    stats.accepted += k
    stats.per_round_accepted.append(k)
    stats.emitted += len(out)
    return out


def generate(prefix, draft, target, k, n_tokens, seed=0):
    rng, stats, out = np.random.default_rng(seed), Stats(), list(prefix)
    while len(out) - len(prefix) < n_tokens:
        out += speculative_step(out, draft, target, k, rng, stats)
    return out[: len(prefix) + n_tokens], stats


def expected_tokens_per_round(alpha: float, k: int) -> float:
    """E[tokens emitted per target call] for i.i.d. acceptance with rate alpha: (1 − α^(k+1)) / (1 − α)."""
    return k + 1.0 if alpha >= 1 else (1 - alpha ** (k + 1)) / (1 - alpha)


def expected_speedup(alpha: float, k: int, c: float) -> float:
    """Wall-clock speedup vs plain decoding when one draft step costs c × one target step (Leviathan et al. eq. 2-ish):
    tokens per round / (cost of k draft steps + 1 target step), in units of target steps."""
    return expected_tokens_per_round(alpha, k) / (k * c + 1.0)


class Markov(Model):
    """A toy model for T0 tests: next-token distribution depends only on the last token (a V×V transition matrix)."""

    def __init__(self, T: np.ndarray):
        self.T = T / T.sum(axis=1, keepdims=True)

    def probs(self, prefix):
        return self.T[prefix[-1]]
