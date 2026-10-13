"""Exercise 1 solution."""

HOURS_PER_DAY = 24
DAYS_PER_MONTH = 30


def session_cost(price_per_hour: float, hours: float, sessions: int = 1) -> float:
    return price_per_hour * hours * sessions


def idle_cost(ebs_gb: float, price_per_gb_month: float, days: float) -> float:
    return ebs_gb * price_per_gb_month * (days / DAYS_PER_MONTH)


def forgotten_cost(price_per_hour: float, days: float) -> float:
    return price_per_hour * HOURS_PER_DAY * days


def budget_ok(planned: float, budget: float, headroom: float = 0.2) -> bool:
    return planned <= budget * (1 - headroom)
