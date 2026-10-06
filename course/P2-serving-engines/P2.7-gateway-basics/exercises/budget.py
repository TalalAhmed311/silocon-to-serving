"""Exercise 4 starter."""
import time


class TokenBudget:
    def __init__(self, limit: int, window_s: float, clock=time.monotonic):
        raise NotImplementedError

    def remaining(self) -> int:
        raise NotImplementedError

    def check(self, estimate: int) -> bool:
        raise NotImplementedError

    def commit(self, actual: int) -> None:
        raise NotImplementedError
