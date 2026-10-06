# Exercise 1 — TTFT / ITL / TPOT / E2E from timestamps (easy)

Implement `request_metrics(arrival: float, token_times: list[float]) -> dict` in `metrics.py`. It returns `{"ttft", "e2e", "tpot", "itl": [gaps], "n"}`, all in **seconds**. For a 1-token answer, `tpot` is `None` and `itl` is `[]`.

Also implement `percentiles(values, qs=(50, 90, 99)) -> dict[int, float]`, using the "linear" method so it matches `numpy.percentile`'s default.

**Test:** the `sample_tokens.jsonl` requests, hand-checked.
