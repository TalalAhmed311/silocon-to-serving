# Exercise 3 — Circuit breaker + fallback (medium)

#6 already falls back to the next backend on failure (see `Gateway._unary` and `_stream`). What it lacks is a **circuit breaker**: after `threshold` consecutive failures, stop sending to a backend for `cooldown_s`, then let **one** probe request through (half-open). If the probe succeeds, close the breaker. If it fails, open it again.

1. Implement `CircuitBreaker(threshold, cooldown_s, clock)` in [`breaker.py`](breaker.py), with `allow() -> bool`, `record(ok: bool)` and `state`, one of `"closed" | "open" | "half_open"`.
2. **Test:** the state machine in virtual time, plus an end-to-end test of the *existing* fallback. It starts two mock backends, one of them with `FAIL_RATE=1.0`, behind the gateway: 30 requests must all succeed.
3. (Optional) Wire the breaker into `gateway/router.py`, replacing the simple `healthy` flags, and rerun the end-to-end test.
