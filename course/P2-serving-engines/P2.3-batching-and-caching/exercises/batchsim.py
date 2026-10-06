"""batchsim.py — a discrete-event simulator of an LLM engine's batching policy (P2.3). STARTER: implement
`continuous()` (exercise 1) and its chunked-prefill option (exercise 2). `static()` is given.

Cost model (same shape as platform/mockllm): one engine step costs
    step_ms = base_ms + per_seq_ms * (sequences decoding this step) + prefill_ms_per_token * (prompt tokens this step)
A sequence emits its first token at the end of the step that finishes its prefill, then one token per step.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Req:
    rid: int
    arrival: float      # ms
    prompt: int         # tokens
    output: int         # tokens to generate (>= 1)


@dataclass
class Cost:
    base_ms: float = 8.0
    per_seq_ms: float = 0.25
    prefill_ms_per_token: float = 0.05


@dataclass
class Outcome:
    first: dict = field(default_factory=dict)      # rid -> time of first token
    done: dict = field(default_factory=dict)       # rid -> time of last token
    token_times: dict = field(default_factory=dict)  # rid -> [times]
    steps: list = field(default_factory=list)      # (t_start, t_end, n_decoding, prefill_tokens) for plotting

    def ttft(self, reqs):
        return {r.rid: self.first[r.rid] - r.arrival for r in reqs}

    def e2e(self, reqs):
        return {r.rid: self.done[r.rid] - r.arrival for r in reqs}

    def max_itl(self, rid):
        t = self.token_times[rid]
        return max((b - a for a, b in zip(t, t[1:])), default=0.0)


def static(reqs: list[Req], batch: int, cost: Cost = Cost()) -> Outcome:
    """Request-level batching: take the next `batch` requests in arrival order, wait until the last of them has
    arrived (and the previous batch is done), prefill them all in one step, then decode until ALL have finished."""
    out, t = Outcome(), 0.0
    q = sorted(reqs, key=lambda r: r.arrival)
    for i in range(0, len(q), batch):
        group = q[i:i + batch]
        t = max(t, group[-1].arrival)
        step = cost.base_ms + cost.prefill_ms_per_token * sum(r.prompt for r in group) + cost.per_seq_ms * len(group)
        out.steps.append((t, t + step, len(group), sum(r.prompt for r in group)))
        t += step
        left = {r.rid: r.output for r in group}
        for r in group:
            out.first[r.rid] = t
            out.token_times[r.rid] = [t]
            left[r.rid] -= 1
        while any(v > 0 for v in left.values()):
            running = [rid for rid, v in left.items() if v > 0]
            step = cost.base_ms + cost.per_seq_ms * len(running)   # finished slots sit idle but the batch keeps going
            out.steps.append((t, t + step, len(running), 0))
            t += step
            for rid in running:
                out.token_times[rid].append(t)
                left[rid] -= 1
        for r in group:
            out.done[r.rid] = out.token_times[r.rid][-1]
    return out


def continuous(reqs: list[Req], max_batch: int, cost: Cost = Cost(), chunk_tokens: int | None = None) -> Outcome:
    """Iteration-level batching (exercise 1): every step, admit arrived requests (FIFO) while fewer than max_batch are
    running; run one step; finished sequences leave immediately.
    chunk_tokens (exercise 2): if set, each step processes at most `chunk_tokens` tokens in total — one per decoding
    sequence, the remainder spent on prefill chunks (FIFO among sequences still prefilling)."""
    raise NotImplementedError
