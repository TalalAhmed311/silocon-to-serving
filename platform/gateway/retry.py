"""retry.py — retry policy for LLM calls: exponential backoff with full jitter, a per-request attempt cap, and a
global retry budget (retries ≤ budget_ratio × requests) so retries can't amplify an outage into a retry storm.

Rule that matters for LLMs: never retry a *streaming* request after the first byte reached the client — the client
would see duplicated or spliced output. Before the first byte, a retry (or fallback) is invisible to the client.
"""
from __future__ import annotations

import random
import threading

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class RetryBudget:
    def __init__(self, ratio: float = 0.2, min_retries: int = 10):
        self.ratio, self.min = ratio, min_retries
        self.requests = self.retries = 0
        self._lock = threading.Lock()

    def on_request(self) -> None:
        with self._lock:
            self.requests += 1

    def try_spend(self) -> bool:
        with self._lock:
            if self.retries < max(self.min, self.ratio * self.requests):
                self.retries += 1
                return True
            return False


def backoff_s(attempt: int, base: float = 0.1, cap: float = 2.0, rng: random.Random | None = None) -> float:
    """Full jitter (AWS architecture blog's recommendation): uniform(0, min(cap, base * 2^attempt))."""
    return (rng or random).uniform(0, min(cap, base * 2 ** attempt))


def should_retry(status: int | None, exc: BaseException | None, started_streaming: bool, attempt: int, max_attempts: int,
                 budget: RetryBudget) -> bool:
    if started_streaming or attempt + 1 >= max_attempts:
        return False
    retryable = exc is not None or (status in RETRYABLE_STATUS)
    return retryable and budget.try_spend()
