# cost: #2, the cost-per-token dashboard

- **`model.py`**: the allocation model as pure functions: $/GPU-hour, replica cost, per-tenant $/1M tokens by token share, idle "waste", and the breakeven utilization against an API price.
- **`exporter.py`**: publishes `s2s_cost_usd_per_million_tokens{tenant}` for the Grafana dashboard in `platform/observability`.

**No prices are hard-coded.** You put your region's prices, or your actual bill's effective rates, in `prices.yaml`. Built in [P3.5](../../course/P3-deployment-and-infra/P3.5-observability/README.md).
