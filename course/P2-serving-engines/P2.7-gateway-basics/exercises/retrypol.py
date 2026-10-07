"""Exercise 2 starter."""


class RetryBudget:
    def __init__(self, ratio: float = 0.2, min_retries: int = 10):
        raise NotImplementedError

    def on_request(self) -> None:
        raise NotImplementedError

    def try_spend(self) -> bool:
        raise NotImplementedError


def backoff_s(attempt, base=0.1, cap=2.0, rng=None) -> float:
    raise NotImplementedError


def should_retry(status, exc, started_streaming, attempt, max_attempts, budget) -> bool:
    raise NotImplementedError
