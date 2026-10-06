"""workload.py — request shapes for load tests: prompt/output length distributions and arrival processes.

Everything is seeded so a load test is reproducible: same seed -> same request sequence.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Request:
    t: float              # scheduled send time (s, relative to start)
    prompt_tokens: int
    output_tokens: int
    prefix_id: int | None = None   # requests with the same prefix_id share their first `shared_prefix_tokens`


@dataclass
class Workload:
    rate: float                       # mean arrivals per second
    duration_s: float
    prompt_mean: int = 512
    prompt_sigma: float = 0.6         # lognormal shape: real prompt lengths are heavy-tailed
    output_mean: int = 128
    output_sigma: float = 0.5
    max_prompt: int = 8000
    max_output: int = 1024
    shared_prefix_tokens: int = 0     # >0: a fraction of requests reuse one of `n_prefixes` system prompts
    shared_fraction: float = 0.0
    n_prefixes: int = 4
    arrival: str = "poisson"          # "poisson" (open loop) or "uniform"
    seed: int = 0

    def _lognormal(self, rng: random.Random, mean: float, sigma: float, hi: int) -> int:
        mu = math.log(mean) - sigma**2 / 2  # so that E[X] = mean
        return max(1, min(hi, int(rng.lognormvariate(mu, sigma))))

    def requests(self) -> list[Request]:
        rng = random.Random(self.seed)
        out, t = [], 0.0
        while True:
            t += rng.expovariate(self.rate) if self.arrival == "poisson" else 1.0 / self.rate
            if t >= self.duration_s:
                return out
            pid = rng.randrange(self.n_prefixes) if self.shared_prefix_tokens and rng.random() < self.shared_fraction else None
            p = self._lognormal(rng, self.prompt_mean, self.prompt_sigma, self.max_prompt)
            if pid is not None:
                p = max(p, self.shared_prefix_tokens + 1)
            out.append(Request(t, p, self._lognormal(rng, self.output_mean, self.output_sigma, self.max_output), pid))


def prompt_text(req: Request, shared_prefix_tokens: int) -> str:
    """A prompt of ~req.prompt_tokens words. Shared-prefix requests start with an identical block of words.
    (Word count ≈ token count for the mock; for real tokenizers it is an approximation — loadgen reports both.)"""
    words = []
    if req.prefix_id is not None:
        words += [f"sys{req.prefix_id}w{i}" for i in range(shared_prefix_tokens)]
    words += [f"u{i % 97}" for i in range(req.prompt_tokens - len(words))]
    return " ".join(words)
