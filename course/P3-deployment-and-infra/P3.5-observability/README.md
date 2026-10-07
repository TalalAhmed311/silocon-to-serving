# P3.5 — Observability (+ #13, #2)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the rules (promtool unit tests), tracing (in-process) and cost model; ![T3](https://img.shields.io/badge/tier-T3%20AWS%20cluster-red) for the live dashboards on EKS (kind works for everything except DCGM) |
| **Time** | ≈40 min reading + ≈12 h hands-on |
| **Prerequisites** | P1.3 (metrics), P2.7 (#6), P3.3–P3.4 |
| **You will build** | **#13**: metrics, traces, SLO burn-rate alerts and dashboards. **#2**: $/1M tokens by tenant, model and route |

## Learning objectives

1. Use the three signals (**metrics, traces, logs**) and know which question each one answers.
2. Define **SLIs, SLOs and error budgets** for LLM serving, and alert on **burn rate** across multiple windows.
3. Scrape vLLM and DCGM metrics with the right names, and compute p90 TTFT from histograms in PromQL.
4. Trace a request across the gateway and the backend with OpenTelemetry.
5. Turn infrastructure prices and token counts into **$/1M tokens**, and see where the waste is.

---

## 1. Which signal answers which question

| Question | Signal | Example |
|---|---|---|
| "Is it healthy right now? Is it getting worse?" | **metrics** (cheap, aggregated) | p90 TTFT, KV usage, waiting requests, GPU util |
| "Why was *this* request slow?" | **traces** (per request, sampled) | gateway 3 ms → backend attempt 1 failed in 80 ms → attempt 2: TTFT 1.9 s |
| "What exactly happened?" | **logs** (verbose, per event) | the vLLM error line, the preemption message |

## 2. SLOs and burn rates

- **SLI** (indicator): the fraction of requests with TTFT < 2 s.
- **SLO** (objective): that SLI ≥ 99% over 30 days.
- **Error budget**: 1% of requests may be slow. Spend it on deploys, experiments and spot interruptions. When it's gone, freeze risky changes.

Alerting on "p90 > 2 s right now" is noisy, and it is blind to slow leaks. Alert instead on the **burn rate**, the speed at which the budget is being consumed. This is the Google SRE workbook's multi-window, multi-burn-rate recipe, encoded in `platform/observability/rules/slo.yaml`:

| Alert | Condition | Meaning |
|---|---|---|
| page | error ratio > 14.4 × 1% over **1 h and 5 min** | this pace exhausts a 30-day budget in ~2 days. Wake someone |
| ticket | error ratio > 6 × 1% over **6 h and 30 min** | slower leak. Fix it this week |

The short window makes the alert *reset* quickly after a fix. The long window keeps it from firing on a blip.

### TTFT from histograms

vLLM exports `vllm:time_to_first_token_seconds` as a histogram (P1.3). The p90 over 5 minutes is:

```promql
histogram_quantile(0.90, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
```

The SLI "fraction under 2 s" is `bucket{le="2"} / count`. **That needs a bucket boundary at exactly 2.** Check vLLM's bucket list at the pinned version, or choose your SLO threshold at an existing boundary.

## 3. The metrics that matter (names verified against the pinned sources)

| Layer | Metric | Why |
|---|---|---|
| engine (vLLM) | `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:request_queue_time_seconds` | the user-facing latencies |
| engine | `vllm:num_requests_running`, `vllm:num_requests_waiting` | load and the queue: the autoscaling signal (P3.6) |
| engine | `vllm:kv_cache_usage_perc`, `vllm:num_preemptions` | KV pressure (P2.3) |
| GPU (DCGM) | `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_XID_ERRORS` | utilization, memory, power, hardware faults |
| gateway (#6) | `gateway_requests_total{tenant,model,backend,outcome}`, `gateway_tokens_total{tenant,model,backend,kind}`, `gateway_ttft_seconds`, `gateway_retries_total` | per-tenant usage and errors, the cost input |

**GPU utilization is not efficiency.** `DCGM_FI_DEV_GPU_UTIL` is the fraction of time *any* kernel ran. A memory-bound decode kernel shows "100% utilized" while the tensor cores sit mostly idle. Use tokens/s against your #4 knee to judge efficiency.

## 4. Traces through the gateway

`platform/gateway/tracing.py` creates a `gateway.request` span per request and a `gateway.backend_attempt` child per try, and injects W3C `traceparent` into backend calls. Enable it with `S2S_TRACING=console` on a laptop, or with `otlp` plus an OpenTelemetry Collector → Tempo or Jaeger in the cluster. A retried request then shows up as one trace with two attempts, which is far easier to read than two unrelated log lines.

## 5. #2: dollars per million tokens

`platform/cost/model.py`:

```
$/GPU-h      = node $/h × (1 + spot interruption overhead) / GPUs per node
replica $/h  = $/GPU-h × GPUs (TP degree, or a MIG fraction)
tenant $/1M  = Σ over replicas (replica $/h × tenant's token share) / tenant tokens per hour × 1e6
waste $/h    = (1 − served / capacity) × replica $/h      (capacity = your #4 knee in tokens/h)
```

Waste is reported **separately**. Smearing idle cost into every tenant's rate hides the real lever: utilization (autoscaling in P3.6, bin packing, MIG in P4.4).

> **Predict first.** One `g6.xlarge` at an hourly price **P** (look it up in your region) serves an 8B model. Its knee from #4 is **K** output tokens/s. At 100% utilization, $/1M output tokens = P / (K × 3600) × 1e6. At 30% utilization (typical for bursty traffic without autoscaling), it is 3.3× that. Compute both with your P and K, then compare with an API price you can find publicly. `breakeven_utilization()` tells you the utilization at which self-hosting wins.

---

## Walkthrough

```bash
promtool test rules platform/observability/rules/slo_test.yaml                     # T0 (promtool binary)
S2S_TRACING=console GATEWAY_CONFIG=platform/gateway/example.yaml uv run uvicorn gateway.app:app --app-dir platform --port 9000
uv run pytest course/P3-deployment-and-infra/P3.5-observability/exercises
# T3: after P3.2 make addons:
kubectl apply -f platform/observability/podmonitors.yaml
uv run python platform/observability/make_prometheusrule.py | kubectl apply -f -
kubectl -n monitoring create configmap s2s-dash --from-file=platform/observability/dashboards/llm-serving.json && \
  kubectl -n monitoring label configmap s2s-dash grafana_dashboard=1
```

## What you should see

- `promtool test rules`: `SUCCESS`. The fast-burn alert fires on the 20%-slow trace and stays silent on the 0.5%-slow trace.
- Console tracing: for each request, a `gateway.request` span containing one or more `gateway.backend_attempt` spans with the backend, status and token attributes.
- On the cluster (`TODO(run-on: EKS)`): the dashboard fills in while you run #4. Burn-rate panels rise when you overload past the knee.

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [PromQL p90 TTFT and the SLI, tested](exercises/01-promql.md) | T0 | `promtool test rules` (`test_promtool.py` runs it if promtool is installed) |
| 2 | [Burn-rate alerts: prove they fire and reset](exercises/02-burn-rate.md) | T0 | extend `slo_test.yaml` with a recovery case |
| 3 | [Trace propagation across gateway → backend](exercises/03-tracing.md) | T0 | `test_tracing.py`: one trace, nested spans, a retried attempt visible; then add streaming spans |
| 4 | [The cost model](exercises/04-cost.md) | T0 | `test_cost.py`: allocation, MIG share, spot overhead, waste, breakeven |
| 5 | [Cost-spike alert on a synthetic tenant flood](exercises/05-cost-spike.md) | T0 | `cost_test.yaml` (promtool) |

## Common mistakes

- Averaging percentiles across pods. `avg(p90)` is not a p90. Aggregate the **buckets**, then take the quantile.
- `rate()` over a window shorter than 4× the scrape interval.
- Reading `DCGM_FI_DEV_GPU_UTIL` as efficiency.
- High-cardinality labels: putting a request ID, prompt or user ID on a metric. That belongs in traces and logs.
- Charging tenants for idle capacity without saying so.

## Go deeper

- Google SRE books: *Service Level Objectives*, and the workbook's *Alerting on SLOs*.
- Prometheus docs (histograms and `histogram_quantile`, recording rules, `promtool test rules`). OpenTelemetry Python docs. Grafana docs.
- vLLM `docs/design/metrics.md` (interval definitions) and `docs/usage/metrics.md` (names).
- `NVIDIA/dcgm-exporter@fafd1511`, `etc/default-counters.csv`.
- Modular handbook: *Comprehensive observability*, *Build and maintenance cost*. Silicon to Scale ch. 17.

**Next:** [P3.6 Autoscaling, cold starts, spot](../P3.6-autoscaling/README.md).
