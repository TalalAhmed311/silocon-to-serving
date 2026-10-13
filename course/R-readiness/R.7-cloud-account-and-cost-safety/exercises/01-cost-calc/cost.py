"""Exercise 1 starter: the cost formulas from R.7. Prices are inputs; never hard-code a real price."""

HOURS_PER_DAY = 24
DAYS_PER_MONTH = 30  # a billing month is modelled as 30 days here


def session_cost(price_per_hour: float, hours: float, sessions: int = 1) -> float:
    """Cost of `sessions` GPU sessions of `hours` each: hours x hourly price, per session."""
    raise NotImplementedError


def idle_cost(ebs_gb: float, price_per_gb_month: float, days: float) -> float:
    """What a disk costs while the instance is stopped: GB x price per GB-month x (days / 30)."""
    raise NotImplementedError


def forgotten_cost(price_per_hour: float, days: float) -> float:
    """What a running GPU instance costs if you forget it for `days` full days (24 h each)."""
    raise NotImplementedError


def budget_ok(planned: float, budget: float, headroom: float = 0.2) -> bool:
    """True if `planned` leaves at least `headroom` (a fraction, 0.2 = 20%) of the budget unspent."""
    raise NotImplementedError
