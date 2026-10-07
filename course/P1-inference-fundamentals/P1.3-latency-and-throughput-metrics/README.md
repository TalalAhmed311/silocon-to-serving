# P1.3 — Latency and throughput metrics

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f). Everything runs against `platform/mockllm`, a fake server with a continuous-batching cost model |
| **Time** | ≈25 min reading + ≈5 h hands-on |
| **Prerequisites** | P1.2 |
| **You will build** | the metric functions that #4 (P2.3), #13 and #2 (P3.5) reuse, and a streaming client that measures TTFT and ITL itself |

## Learning objectives

1. Define TTFT, ITL, TPOT, end-to-end latency, throughput (requests/s and output tokens/s) and **goodput**.
2. Compute them, with percentiles, from per-token timestamps.
3. Compute percentiles from **histogram buckets** the way Prometheus does (`histogram_quantile`), and know its error.
4. Explain the latency–throughput trade-off, and why a mean hides the tail.
5. Map each metric to the vLLM Prometheus metric that measures it.

## Why this matters

"Is it fast?" means nothing until you say *which* latency, at *what percentile*, under *what load*. SLOs (P3.5), autoscaling signals (P3.6), benchmark reports (P2.4) and the capstone teardown (#15) are all written in these terms.

---

## 1. Definitions

For one streamed request:

```
arrival ─── queue ─── prefill ───▶ token 1 ── token 2 ── … ── token n
   │◀────────── TTFT ──────────▶│◀ ITL ▶│
   │◀──────────────────── end-to-end latency (E2E) ─────────────────▶│
```

| Metric | Definition | Driven by |
|---|---|---|
| **TTFT** (time to first token) | first token time − arrival | queueing + prefill (prompt length, batch contention) |
| **ITL** (inter-token latency) | gap between consecutive tokens of one request; a *distribution* per request | decode step time (batch size, KV reads) |
| **TPOT** (time per output token) | (E2E − TTFT) / (n − 1); the *mean* ITL of one request | same as ITL |
| **E2E** | last token time − arrival | TTFT + (n − 1) × TPOT |
| **throughput** | completed requests/s, or output tokens/s, over a window | batch size × 1/step time |
| **goodput** | throughput counting only requests that **met the SLO** | all of the above |

Users feel TTFT ("did it start?") and ITL ("does it stream smoothly?"). Operators pay for throughput. **Goodput** is the honest compromise: requests per second that were actually good enough. That is why DistServe and others optimize it instead of raw throughput.

## 2. Percentiles, not means

Latency distributions are skewed. A few requests that queued behind a long prefill drag the mean up while the median barely moves, and the **p99** is the experience of one user in a hundred. Report **p50 / p90 / p99** and the request count. A mean with no percentiles is a red flag in any benchmark.

### Percentiles from histograms

Prometheus doesn't store raw samples. It stores cumulative bucket counts (`le` = "less than or equal"). `histogram_quantile(0.9, …)` finds the bucket containing the 90th-percentile rank and **linearly interpolates inside it**. The answer is only as precise as the bucket widths: with buckets `…, 0.5, 1, 2.5, …`, a true p90 of 0.6 s may be reported anywhere in 0.5–1 s. Exercise 2 implements the algorithm and measures that error.

## 3. Load shapes the numbers

The same server gives very different numbers at 1 req/s and at 50 req/s. As the arrival rate approaches capacity:

- TTFT explodes first: requests queue, and the queue grows without bound past capacity.
- ITL grows more slowly: bigger batches mean slower steps.
- Throughput rises, then plateaus.

The plot of latency vs throughput has a **knee**. P2.3's load test (#4) finds it. Here you see it on the mock server with `bench/mock_load.py`.

## 4. Where vLLM reports each metric

Names read from `vllm/docs/usage/metrics.md` and `vllm/v1/metrics/` at `vllm-project/vllm@d1f3d8b8`:

| Concept | vLLM Prometheus metric (type) |
|---|---|
| TTFT | `vllm:time_to_first_token_seconds` (histogram) |
| ITL | `vllm:inter_token_latency_seconds` (histogram) |
| TPOT | `vllm:request_time_per_output_token_seconds` (histogram) |
| queue time | `vllm:request_queue_time_seconds` (histogram) |
| prefill / decode time | `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` |
| load | `vllm:num_requests_running`, `vllm:num_requests_waiting` (gauges) |
| KV pressure | `vllm:kv_cache_usage_perc` (gauge), `vllm:num_preemptions` |

`platform/mockllm` exports the same names, so everything you build against it carries over.

> **Predict first.** Start the mock server with its defaults: base step 8 ms, +0.25 ms per running sequence, max 64 sequences. At a steady 64 concurrent streams, what ITL do you expect? (8 + 0.25 × 64 = 24 ms.) What output tokens/s? (64 tokens per 24 ms ≈ 2,667 tok/s.) Then run `bench/mock_load.py --concurrency 64` and compare. These are the mock's made-up costs, so the point is to check that your *measurement* code recovers a number you can compute.

---

## Walkthrough

```bash
uv run uvicorn mockllm.server:app --app-dir platform --port 8001 &      # the fake engine
D=course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics
uv run python $D/examples/01_metrics_from_log.py $D/examples/sample_tokens.jsonl
uv run python $D/examples/02_stream_client.py --url http://localhost:8001 --n 20
uv run python $D/bench/mock_load.py --url http://localhost:8001 --rates 5 20 50 100
kill %1
```

## What you should see

- `01`: a table of per-request TTFT/TPOT/E2E and p50/p90/p99 for the 6 sample requests in `sample_tokens.jsonl`. The values are deterministic: they are computed from the file.
- `02`: client-measured TTFT/ITL for 20 concurrent streams, next to the server's own histogram percentiles scraped from `/metrics`.
- `mock_load`: as the rate rises, p90 TTFT grows slowly, then sharply, while output tokens/s plateaus. **TODO(run): paste your table.** The knee's position follows from the mock's `MOCKLLM_*` settings.

## Exercises

```bash
uv run pytest course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/exercises
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [TTFT / ITL / TPOT / E2E from timestamps](exercises/01-request-metrics/README.md) | easy | hand-computed fixtures, n = 1 edge case |
| 2 | [`histogram_quantile` like Prometheus](exercises/02-histogram-quantile/README.md) | medium | vs `numpy.percentile` within bucket resolution; edge cases |
| 3 | [Goodput under a two-part SLO](exercises/03-goodput/README.md) | easy | fixtures |
| 4 | [SSE client: client-side vs server-side TTFT](exercises/04-sse-client/README.md) | hard | in-process mock server (no network), checks consistency with `/metrics` |

## Common mistakes

- **Averaging ITL across requests unweighted.** Long requests have more gaps. Decide what you're reporting: the per-token or the per-request mean.
- **Counting TTFT from when the client *sent* vs when the server *received*.** Both are valid, but say which. They differ by the network plus the server's accept queue.
- **Closed-loop load generators** that wait for a response before sending the next request. They hide queueing (coordinated omission, P2.4).
- **Reading `histogram_quantile` as exact.** It is an interpolation inside a bucket.

## Go deeper

- Modular handbook: *LLM inference metrics*, with the LatencyTimelineVisualizer interactive.
- vLLM `docs/design/metrics.md`: how each interval is defined inside the engine (with diagrams).
- Silicon to Scale ch. 12 (benchmarking).
- Google SRE book: *Service Level Objectives*.

## Go down when…

You need to know *why* TTFT and ITL respond to load the way they do → [P1.2](../P1.2-prefill-decode-kv-cache/README.md). **Next:** [P1.4 GPU architecture and the roofline](../P1.4-gpu-architecture-and-roofline/README.md).
