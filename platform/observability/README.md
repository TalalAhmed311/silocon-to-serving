# observability: #13, the observability spine

| Path | What |
|---|---|
| `kps-values.yaml` | kube-prometheus-stack: scrape across namespaces, Grafana admin password from a Secret, a dashboard sidecar |
| `podmonitors.yaml` | scraping for vLLM, the gateway and mockllm (DCGM comes through the GPU Operator's ServiceMonitor) |
| `rules/slo.yaml` | TTFT/ITL p90 recording rules, **multi-window burn-rate** alerts for a 99% TTFT < 2 s SLO, KV-saturation and XID alerts |
| `rules/cost.yaml` | per-tenant token-spike and output-length-drift alerts |
| `rules/slo_test.yaml` | **promtool unit tests** for the SLO rules: `promtool test rules platform/observability/rules/slo_test.yaml` |
| `dashboards/llm-serving.json` | Grafana: TTFT, ITL, running/waiting, KV usage, GPU util, tokens per tenant, burn rate, $/1M tokens |
| `../gateway/tracing.py` | OpenTelemetry spans for gateway → backend: queue, TTFT and tokens as attributes |

Metric names: vLLM's come from `vllm/docs/usage/metrics.md` and `vllm/v1/metrics/` at the pinned SHA. DCGM's come from `etc/default-counters.csv` in dcgm-exporter 4.8.4 (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_XID_ERRORS`, …). The gateway's are defined in `platform/gateway/app.py`.
