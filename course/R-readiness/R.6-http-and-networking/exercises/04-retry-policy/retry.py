"""Exercise 4: when to retry, and how long to wait."""
import random


def should_retry(status: int | None, attempt: int, idempotent: bool, max_attempts: int = 3) -> bool:
    """Decide whether to send the request again.

    status:   the HTTP status of the failed try, or None if there was no response (timeout, connection reset).
    attempt:  how many tries have been made so far (1 after the first try).
    idempotent: True if sending the request twice has the same effect as once.

    Rules:
      - never retry once attempt >= max_attempts;
      - 429 and 503 mean "I did not take this on": retry, idempotent or not;
      - 500, 502, 504 and None (no response) may have been processed: retry only if idempotent;
      - everything else (2xx, 3xx, other 4xx such as 400/401/404) is never retried.
    """
    # TODO
    raise NotImplementedError


def backoff_bound(attempt: int, base: float = 0.1, cap: float = 5.0) -> float:
    """Upper bound of the wait before the next try: min(cap, base * 2 ** (attempt - 1))."""
    # TODO
    raise NotImplementedError


def backoff_delay(attempt: int, rng: random.Random, base: float = 0.1, cap: float = 5.0) -> float:
    """'Full jitter': a random wait, uniform between 0 and backoff_bound(attempt)."""
    # TODO: use rng.uniform
    raise NotImplementedError
