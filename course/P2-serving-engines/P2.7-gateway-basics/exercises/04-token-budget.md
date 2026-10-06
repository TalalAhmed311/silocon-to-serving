# Exercise 4 — Per-tenant token budget (medium)

Implement `TokenBudget(limit, window_s, clock)` in [`budget.py`](budget.py), with `remaining()`, `check(estimate) -> bool` and `commit(actual)`. Usage expires `window_s` after it was committed: a **sliding** window, not a calendar reset.

**Test:** virtual clock. It checks commits and remaining, expiry exactly at the window edge, that the estimate check doesn't consume anything, and an end-to-end case: the gateway answers 429 once a tenant with a tiny budget has used it up, measured with the mock's real `usage`.
