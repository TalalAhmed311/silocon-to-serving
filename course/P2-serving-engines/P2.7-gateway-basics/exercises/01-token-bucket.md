# Exercise 1 — Token bucket (easy)

Implement `TokenBucket(rate, burst, clock)` in [`bucket.py`](bucket.py), with `allow(n=1) -> bool` and `retry_after(n=1) -> float`. Refill lazily on each call: `tokens = min(burst, tokens + elapsed × rate)`.

**Test:** a virtual clock, no sleeps. It checks the initial burst, the refusal when empty, partial refill, the cap at `burst`, and an exact `retry_after`.
