"""policy.py — which region serves traffic (C1 exercise 1). A pure function plus a small state machine.

Active-passive: traffic goes to `primary` while it is healthy. When it goes DOWN and the secondary is UP, fail over.
Fail BACK only after the primary has been continuously UP for `failback_after_s` (and never automatically if
`auto_failback` is False — many teams fail back by hand, in business hours). If both are down, stay put: moving
traffic to another dead region only adds a DNS change to the incident. `min_dwell_s` stops ping-pong when health
signals disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    active: str
    reason: str


def decide(active: str, primary: str, secondary: str, up: dict[str, bool], since_up: dict[str, float],
           now: float, last_switch: float, failback_after_s: float = 600.0, min_dwell_s: float = 120.0,
           auto_failback: bool = True) -> Decision:
    """up[r]: region r's (hysteresis-filtered) health; since_up[r]: when r last became UP (inf if down)."""
    other = secondary if active == primary else primary
    if now - last_switch < min_dwell_s:
        return Decision(active, "dwell")
    if not up[active]:
        if up[other]:
            return Decision(other, f"{active} down, {other} up: fail over")
        return Decision(active, "both down: hold")
    if active == secondary and auto_failback and up[primary] and now - since_up[primary] >= failback_after_s:
        return Decision(primary, f"{primary} stable for {failback_after_s:.0f}s: fail back")
    return Decision(active, "steady")


@dataclass
class Controller:
    primary: str
    secondary: str
    failback_after_s: float = 600.0
    min_dwell_s: float = 120.0
    auto_failback: bool = True
    active: str = ""
    last_switch: float = float("-inf")
    since_up: dict = field(default_factory=dict)
    log: list = field(default_factory=list)                 # (t, from, to, reason)

    def __post_init__(self):
        self.active = self.active or self.primary

    def step(self, now: float, up: dict[str, bool]) -> str:
        for r, u in up.items():
            if u and self.since_up.get(r, float("inf")) == float("inf"):
                self.since_up[r] = now
            elif not u:
                self.since_up[r] = float("inf")
        d = decide(self.active, self.primary, self.secondary, up, self.since_up, now, self.last_switch,
                   self.failback_after_s, self.min_dwell_s, self.auto_failback)
        if d.active != self.active:
            self.log.append((now, self.active, d.active, d.reason))
            self.active, self.last_switch = d.active, now
        return self.active
