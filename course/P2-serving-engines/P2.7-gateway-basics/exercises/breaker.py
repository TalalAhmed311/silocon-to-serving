"""Exercise 3 starter."""
import time


class CircuitBreaker:
    def __init__(self, threshold: int = 5, cooldown_s: float = 30.0, clock=time.monotonic):
        raise NotImplementedError

    @property
    def state(self) -> str:
        raise NotImplementedError

    def allow(self) -> bool:
        raise NotImplementedError

    def record(self, ok: bool) -> None:
        raise NotImplementedError
