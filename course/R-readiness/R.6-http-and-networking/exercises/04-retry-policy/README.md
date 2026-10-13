# Exercise 4: a retry policy

Implement `should_retry`, `backoff_bound` and `backoff_delay` in `retry.py`. Retrying a request that already ran can do its work twice, so the policy depends on the status and on whether the request is idempotent. Waiting longer after each failure (exponential backoff) with a random jitter stops many clients from retrying in lock-step. The test checks a table of cases; the docstring states the rules.

```bash
uv run pytest course/R-readiness/R.6-http-and-networking/exercises/04-retry-policy
```
