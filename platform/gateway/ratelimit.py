"""ratelimit.py — token bucket (request rate) and a rolling token budget (LLM tokens per tenant per window).

Both take an injectable `clock` so tests run in virtual time (no sleeps).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable


class TokenBucket:
    """Classic token bucket: capacity `burst`, refilled at `rate` tokens/s. allow(n) consumes n if available."""

    def __init__(self, rate: float, burst: float, clock: Callable[[], float] = time.monotonic):
        self.rate, self.burst, self.clock = rate, burst, clock
        self.tokens, self.t = burst, clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now

    def allow(self, n: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def retry_after(self, n: float = 1.0) -> float:
        """Seconds until n tokens will be available (for the Retry-After header)."""
        with self._lock:
            self._refill()
            return max(0.0, (n - self.tokens) / self.rate)


class TokenBudget:
    """Sliding-window budget of LLM tokens (prompt + completion) per tenant, e.g. 1M tokens per 24 h.
    check() is called before a request with an *estimate*; commit() records actual usage after it completes."""

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window_s, clock
        self.events: deque[tuple[float, int]] = deque()
        self.used = 0
        self._lock = threading.Lock()

    def _expire(self) -> None:
        cutoff = self.clock() - self.window
        while self.events and self.events[0][0] <= cutoff:
            self.used -= self.events.popleft()[1]

    def remaining(self) -> int:
        with self._lock:
            self._expire()
            return self.limit - self.used

    def check(self, estimate: int) -> bool:
        return self.remaining() >= estimate

    def commit(self, actual: int) -> None:
        with self._lock:
            self._expire()
            self.events.append((self.clock(), actual))
            self.used += actual
