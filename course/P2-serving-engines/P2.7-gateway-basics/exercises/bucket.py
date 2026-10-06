"""Exercise 1 starter."""
import time


class TokenBucket:
    def __init__(self, rate: float, burst: float, clock=time.monotonic):
        raise NotImplementedError

    def allow(self, n: float = 1.0) -> bool:
        raise NotImplementedError

    def retry_after(self, n: float = 1.0) -> float:
        raise NotImplementedError
