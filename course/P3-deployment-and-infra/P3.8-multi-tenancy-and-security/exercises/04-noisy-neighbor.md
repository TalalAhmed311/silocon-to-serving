# Exercise 4 (hard): noisy neighbor (T0)

`test_noisy.py` puts two tenants on one mock backend with only 4 sequence slots:

- **A (noisy):** `rps: 5`, floods 100 concurrent requests
- **B (victim):** sends 15 requests one after another, measuring latency

`test_rate_limit_protects_b` passes as shipped: the gateway 429s most of A's flood, so B's p90 latency stays within 3× its solo baseline. Read it and predict the number of 429s before you run it.

**Your task: `test_long_outputs_within_rps`** (xfail). A stays *inside* its rps limit but asks for `max_tokens: 2000` on every request. Rate limiting doesn't help: each request holds a slot for a long time. Make B safe anyway, then remove the xfail. Options, in increasing order of effort:

1. a per-tenant `max_tokens` cap in the gateway config (reject or clamp)
2. a per-tenant **concurrency** limit in the gateway (an `asyncio.Semaphore` per tenant; excess requests queue or get 429)
3. weighted fair queuing across tenants in front of the backend

Choose one, implement it in `platform/gateway/`, keep every P2.7 test green, and write the B-latency assertion.
