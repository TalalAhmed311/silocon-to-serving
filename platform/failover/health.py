"""health.py — a health signal that doesn't flap (C1 exercise 2).

A raw probe result is noisy: one timeout is not an outage, one success after ten failures is not a recovery.
HealthTracker turns a stream of probe results into a stable UP/DOWN state with hysteresis:
  UP → DOWN   after `fail_threshold` consecutive failures   (detection time ≈ fail_threshold × probe interval)
  DOWN → UP   after `rise_threshold` consecutive successes  (rise_threshold > fail_threshold: slow to trust again)
A probe counts as a failure if it errored OR its latency exceeded `latency_slo_s` (a region that answers in 30 s is
down for users). Route 53 health checks have the same two knobs (failure threshold, request interval); this is the
logic you test locally before trusting theirs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Probe:
    t: float
    ok: bool
    latency_s: float = 0.0


@dataclass
class HealthTracker:
    fail_threshold: int = 3
    rise_threshold: int = 5
    latency_slo_s: float = 10.0
    up: bool = True
    _fails: int = 0
    _oks: int = 0
    transitions: list = field(default_factory=list)       # (t, "down" | "up")

    def observe(self, p: Probe) -> bool:
        good = p.ok and p.latency_s <= self.latency_slo_s
        if good:
            self._oks, self._fails = self._oks + 1, 0
            if not self.up and self._oks >= self.rise_threshold:
                self.up = True
                self.transitions.append((p.t, "up"))
        else:
            self._fails, self._oks = self._fails + 1, 0
            if self.up and self._fails >= self.fail_threshold:
                self.up = False
                self.transitions.append((p.t, "down"))
        return self.up


def flaps(transitions) -> int:
    """Number of state changes — the metric the hysteresis tests bound."""
    return len(transitions)
