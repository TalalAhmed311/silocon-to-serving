"""Exercise 4 (solution): when to retry, and how long to wait."""
import random


def should_retry(status: int | None, attempt: int, idempotent: bool, max_attempts: int = 3) -> bool:
    if attempt >= max_attempts:
        return False
    if status in (429, 503):
        return True
    if status is None or status in (500, 502, 504):
        return idempotent
    return False


def backoff_bound(attempt: int, base: float = 0.1, cap: float = 5.0) -> float:
    return min(cap, base * 2 ** (attempt - 1))


def backoff_delay(attempt: int, rng: random.Random, base: float = 0.1, cap: float = 5.0) -> float:
    return rng.uniform(0.0, backoff_bound(attempt, base, cap))
