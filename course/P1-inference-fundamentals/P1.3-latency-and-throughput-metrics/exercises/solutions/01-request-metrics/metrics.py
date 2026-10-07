"""Exercise 1 solution."""
import numpy as np


def request_metrics(arrival: float, token_times: list[float]) -> dict:
    if not token_times:
        raise ValueError("no tokens")
    ttft = token_times[0] - arrival
    e2e = token_times[-1] - arrival
    n = len(token_times)
    itl = [b - a for a, b in zip(token_times, token_times[1:])]
    return {"ttft": ttft, "e2e": e2e, "n": n, "itl": itl, "tpot": (e2e - ttft) / (n - 1) if n > 1 else None}


def percentiles(values, qs=(50, 90, 99)) -> dict:
    v = np.asarray(values, dtype=float)
    return {q: float(np.percentile(v, q)) for q in qs}
