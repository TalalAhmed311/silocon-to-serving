# Exercise 2 — Retry policy with budget and jitter (medium)

Implement in [`retrypol.py`](retrypol.py):

- `RetryBudget(ratio, min_retries)`, with `on_request()` and `try_spend() -> bool`. Retries allowed: `max(min_retries, ratio × requests)`.
- `backoff_s(attempt, base, cap, rng)`, full jitter: `uniform(0, min(cap, base·2^attempt))`.
- `should_retry(status, exc, started_streaming, attempt, max_attempts, budget) -> bool`.

**Test:**
- never retries after streaming started
- never retries a 400
- does retry 503 and transport errors
- respects `max_attempts`
- the budget caps retries at 20% of 1000 requests
- backoff stays within its bounds and is not constant (jitter)
