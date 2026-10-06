"""Exercise 3 solution."""
import time


class CircuitBreaker:
    def __init__(self, threshold=5, cooldown_s=30.0, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown_s, clock
        self.failures, self.opened_at, self._probe_out = 0, None, False

    @property
    def state(self):
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.cooldown else "open"

    def allow(self):
        s = self.state
        if s == "closed":
            return True
        if s == "half_open" and not self._probe_out:
            self._probe_out = True          # exactly one probe in flight
            return True
        return False

    def record(self, ok):
        if ok:
            self.failures, self.opened_at, self._probe_out = 0, None, False
            return
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.threshold:
            self.opened_at, self._probe_out = self.clock(), False   # (re)open and restart the cooldown
