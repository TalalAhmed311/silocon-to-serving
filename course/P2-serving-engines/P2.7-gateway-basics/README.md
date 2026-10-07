# P2.7 — Gateway basics (+ #6)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f). Everything runs against `mockllm` backends. Point it at your P2.1 vLLM server for a real backend |
| **Time** | ≈25 min reading + ≈10 h hands-on |
| **Prerequisites** | P1.3 (metrics, SSE), P2.1 |
| **You will build** | **#6**: `platform/gateway`, multi-backend routing, retries, fallbacks, rate limits, per-tenant token budgets |

## Learning objectives

1. Know what a gateway owns: authentication, routing, retries with budgets, fallbacks, rate limits, per-tenant token budgets, streaming passthrough and accounting.
2. Read how LiteLLM and Envoy AI Gateway do routing and rate limiting, then build your own.
3. Implement a token bucket and a sliding-window token budget, testable in virtual time.
4. Get the hard part of LLM retries right: **no retry after the first streamed byte**.

## Why this matters

Every production LLM platform puts a gateway in front of its engines, for three reasons:

- **Many backends.** Several models and replicas, perhaps an external API as a fallback.
- **Many tenants.** Each has a quota and a bill.
- **Failures are normal.** Replicas restart, scale to zero, or OOM.

The gateway is also where the platform's numbers are produced: per-tenant tokens (#2 cost dashboard), end-to-end latency (#13), and error rates (SLOs).

---

## 1. The request path

```
client ──▶ authenticate (Bearer key -> tenant; constant-time hash compare)
       ──▶ tenant rate limit (token bucket: 429 + Retry-After)
       ──▶ tenant token budget (estimate = prompt chars/4 + max_tokens; 429 if exhausted)
       ──▶ route: candidates for the model, best first (least outstanding requests)
       ──▶ try backend 1 ─ fail before first byte? ─▶ backoff + jitter ─▶ backend 2 ─▶ …   (bounded by a retry budget)
       ──▶ stream bytes through; on mid-stream failure send an explicit error event (never splice two outputs)
       ──▶ bill actual usage from the final `usage` chunk; emit metrics
```

## 2. Rate limits vs token budgets

- A **token bucket** of capacity `burst` refills at `rate`/s, and each request takes one token. It allows short bursts and caps the sustained rate. It protects the **service**.
- A **token budget** is LLM tokens per tenant per window (for example 2M per day), billed from actual `usage`. It protects the **bill**. Before a request you only have an *estimate*: the prompt length and `max_tokens`. You check the estimate, then commit the actual usage. The window slides, so usage expires one event at a time instead of resetting at midnight.

## 3. Retries without making outages worse

- **Retry only retryable errors:** 408, 429, 500, 502, 503, 504 and transport errors. Never other 4xx.
- **Backoff with full jitter:** `sleep(uniform(0, min(cap, base·2^attempt)))`, so synchronized clients don't retry in waves.
- **A retry budget:** retries ≤ 20% of requests, globally. When a backend dies, unbudgeted retries triple the load on its survivors. This is the classic "retry storm".
- **Streaming:** once *any* byte reached the client, the response is committed. A retry would duplicate or splice tokens. The gateway reports a mid-stream failure explicitly (an error event) and lets the client decide.

## 4. Routing

Round-robin assumes equal-cost requests. LLM requests differ in cost by 100×, so **least outstanding requests** (in-flight count ÷ weight) is the cheap good default. Prefix-aware or KV-aware routing (llm-d, Dynamo) is P3.4.

## 5. Reading LiteLLM and Envoy AI Gateway

Before you extend #6, skim two production designs:

- **LiteLLM** (`BerriAI/litellm@5ed7ec85`, MIT): `litellm/router.py`, its routing strategies (`litellm/router_strategy/`), and fallback/cooldown handling.
- **Envoy AI Gateway** (`envoyproxy/ai-gateway@daa9f891`, Apache-2.0): token-based rate limiting built on Envoy's rate-limit service, and its `docs/`.

Exact file paths move between versions, so search the pinned trees. Write a 10-line comparison in your notes: what do they do that #6 doesn't?

> **Predict first.** Gateway overhead: on localhost, what latency does #6 add per request (p50) over calling the backend directly? A Python async proxy adds a few hundred µs to a few ms. Measure it with `bench/gateway_overhead.py`. Which part of a typical 1-second LLM request is it?

---

## Walkthrough

```bash
uv run uvicorn mockllm.server:app --app-dir platform --port 8001 &
MOCKLLM_FAIL_RATE=0.5 uv run uvicorn mockllm.server:app --app-dir platform --port 8002 &   # a flaky replica
python - <<'PY'
import hashlib; print(hashlib.sha256(b"dev-key-1").hexdigest())
PY
# paste the hash into platform/gateway/example.yaml (tenant team-a), then:
GATEWAY_CONFIG=platform/gateway/example.yaml uv run uvicorn gateway.app:app --app-dir platform --port 9000 &
curl -s localhost:9000/v1/completions -H 'authorization: Bearer dev-key-1' -H 'content-type: application/json' \
  -d '{"model":"mock-llama-8b","prompt":"hi","max_tokens":8}'
curl -s localhost:9000/metrics | grep gateway_
uv run python course/P2-serving-engines/P2.7-gateway-basics/bench/gateway_overhead.py
```

## What you should see

- Every request succeeds even though replica b fails half the time. `gateway_retries_total` counts the fallbacks.
- `gateway_tokens_total{tenant="team-a"}` grows with the usage the mock reports.
- `gateway_overhead.py` prints `path | p50 ms | p99 ms | added p50 ms`. **TODO(run): paste yours.**

## Exercises

```bash
uv run pytest course/P2-serving-engines/P2.7-gateway-basics/exercises
S2S_SOLUTIONS=1 uv run pytest course/P2-serving-engines/P2.7-gateway-basics/exercises
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [Token bucket](exercises/01-token-bucket.md) | easy | virtual clock: burst, refill, Retry-After |
| 2 | [Retry policy with budget and jitter](exercises/02-retry.md) | medium | never after streaming started; budget caps retries |
| 3 | [Circuit breaker + fallback](exercises/03-circuit-breaker.md) | medium | breaker state machine (virtual clock); end-to-end fallback via two mock backends |
| 4 | [Per-tenant token budget](exercises/04-token-budget.md) | medium | sliding window expiry, estimate vs commit |
| 5 | [SSE passthrough with mid-stream failure](exercises/05-midstream.md) | hard | the client gets an explicit error event and never duplicated text |

## Common mistakes

- Storing tenant keys in plain text in the config.
- Retrying 400s ("bad request" will stay bad).
- Billing the *estimate* instead of the actual usage.
- Holding a lock across `await` in async code.
- Forgetting `stream_options.include_usage`. Without it, an OpenAI-compatible stream doesn't report usage.

## Go deeper

- LiteLLM and Envoy AI Gateway (pinned in SOURCES.md).
- Modular handbook: *Inference routing*, *OpenAI-compatible API*.
- Google SRE book: *Handling Overload*, *Addressing Cascading Failures* (retry budgets).
- AWS Architecture Blog: "Exponential Backoff and Jitter" (search for it).

**Next:** [P3.1 GPU containers](../../P3-deployment-and-infra/P3.1-gpu-containers/README.md).
