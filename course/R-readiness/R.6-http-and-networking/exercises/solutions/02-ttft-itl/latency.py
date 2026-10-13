"""Exercise 2 (solution): time-to-first-token and inter-token latency from timestamps."""
import numpy as np


def stream_metrics(request_sent: float, arrivals: list[float]) -> dict:
    if not arrivals:
        raise ValueError("no chunks arrived")
    gaps = np.diff(np.asarray(arrivals, dtype=float))
    return {
        "ttft": arrivals[0] - request_sent,
        "itl_mean": float(gaps.mean()) if gaps.size else None,
        "itl_p90": float(np.percentile(gaps, 90)) if gaps.size else None,
        "e2e": arrivals[-1] - request_sent,
    }
