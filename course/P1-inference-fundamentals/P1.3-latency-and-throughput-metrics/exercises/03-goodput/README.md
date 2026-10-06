# Exercise 3 — Goodput under a two-part SLO (easy)

Implement `goodput(requests, ttft_slo, tpot_slo, window_s) -> float` in `goodput.py`. `requests` is a list of `request_metrics` dicts from exercise 1. A request is "good" if `ttft <= ttft_slo` **and** (`tpot is None` or `tpot <= tpot_slo`). Return good requests per second over the window.

Also implement `slo_attainment(requests, ttft_slo, tpot_slo) -> float`, the fraction of requests that are good.

**Test:** fixtures. One of them has a request that passes TTFT but fails TPOT.

**Think:** two servers do 10 req/s each. Server A has 9.5 good req/s, server B has 6 good req/s, and B has the better *mean* latency. Which do you deploy? How could that happen?
