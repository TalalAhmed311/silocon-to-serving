"""analysis.py — summarize load-test results and find the latency knee.

Knee definition used by #4 (documented so it is reproducible): the largest offered rate whose p90 TTFT is within the
SLO **and** whose goodput is ≥ 90% of the offered rate. Above it, the queue grows and goodput falls away.
"""
from __future__ import annotations

import numpy as np


def summarize(results, duration_s: float, ttft_slo: float, tpot_slo: float) -> dict:
    ok = [r for r in results if r.error is None and r.first is not None]
    ttft = np.array([r.ttft for r in ok]) if ok else np.array([np.nan])
    tp = [r.tpot for r in ok if r.tpot is not None]
    tpot = np.array(tp) if tp else np.array([np.nan])
    itl = np.concatenate([r.itls for r in ok if r.itls]) if any(r.itls for r in ok) else np.array([np.nan])
    good = sum(1 for r in ok if r.ttft <= ttft_slo and (r.tpot is None or r.tpot <= tpot_slo))
    out_tokens = sum(len(r.stamps) for r in ok)
    pct = lambda v, q: float(np.nanpercentile(v, q))  # noqa: E731
    return {"sent": len(results), "ok": len(ok), "errors": len(results) - len(ok),
            "offered_rps": len(results) / duration_s, "goodput_rps": good / duration_s,
            "out_tok_s": out_tokens / duration_s,
            "ttft_p50": pct(ttft, 50), "ttft_p90": pct(ttft, 90), "ttft_p99": pct(ttft, 99),
            "tpot_p50": pct(tpot, 50), "tpot_p90": pct(tpot, 90), "itl_p50": pct(itl, 50), "itl_p90": pct(itl, 90), "itl_p99": pct(itl, 99)}


def find_knee(rows: list[dict], ttft_slo: float) -> dict | None:
    """rows: summaries sorted by offered rate. Returns the knee row (or None if even the lowest rate fails)."""
    good = [r for r in rows if r["ttft_p90"] <= ttft_slo and r["goodput_rps"] >= 0.9 * r["offered_rps"]]
    return max(good, key=lambda r: r["offered_rps"]) if good else None
