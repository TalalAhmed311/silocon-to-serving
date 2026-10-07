"""router.py — choose a backend for a model: weighted random, or least outstanding requests (LOR).

LOR matters for LLM serving: request durations vary 100× (a 10-token vs a 2,000-token answer), so round-robin piles
long requests onto one replica. Counting in-flight requests per backend is a cheap proxy for its queue depth.
"""
from __future__ import annotations

import random
import threading

from .config import Backend


class Router:
    def __init__(self, backends: list[Backend], strategy: str = "least_outstanding", rng: random.Random | None = None):
        self.backends, self.strategy, self.rng = backends, strategy, rng or random.Random(0)
        self.inflight = {b.name: 0 for b in backends}
        self.healthy = {b.name: True for b in backends}
        self._lock = threading.Lock()

    def candidates(self, model: str) -> list[Backend]:
        """All backends serving `model`, best first. The caller tries them in order (fallback)."""
        with self._lock:
            pool = [b for b in self.backends if (model in b.models or "*" in b.models)]
            healthy = [b for b in pool if self.healthy[b.name]] or pool   # if all are marked unhealthy, still try
            if self.strategy == "weighted":
                ordered = []
                rest = list(healthy)
                while rest:
                    pick = self.rng.choices(rest, weights=[b.weight for b in rest])[0]
                    ordered.append(pick)
                    rest.remove(pick)
                return ordered
            return sorted(healthy, key=lambda b: (self.inflight[b.name] / b.weight, self.rng.random()))

    def acquire(self, b: Backend) -> None:
        with self._lock:
            self.inflight[b.name] += 1

    def release(self, b: Backend) -> None:
        with self._lock:
            self.inflight[b.name] -= 1

    def mark(self, b: Backend, ok: bool) -> None:
        with self._lock:
            self.healthy[b.name] = ok
