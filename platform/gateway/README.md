# gateway: #6, the multi-model AI gateway

OpenAI-compatible in, OpenAI-compatible out. Between the two, it handles:

- **authentication**: per-tenant API keys, stored as SHA-256 hashes and compared in constant time
- **rate limits**: a token bucket per tenant, answering `429` with `Retry-After`
- **token budgets**: a sliding window per tenant, billed from the backend's actual `usage`
- **routing**: least-outstanding-requests or weighted
- **retries**: exponential backoff with full jitter, under a global retry budget, and **never after the first streamed byte**
- **fallbacks**: to the next backend on 5xx or a timeout
- **SSE passthrough**: a mid-stream upstream failure becomes an explicit error event, never a silent splice
- **Prometheus metrics**

```bash
uv run uvicorn mockllm.server:app --app-dir platform --port 8001 &
uv run uvicorn mockllm.server:app --app-dir platform --port 8002 &
GATEWAY_CONFIG=platform/gateway/example.yaml uv run uvicorn gateway.app:app --app-dir platform --port 9000
```

It is built in [P2.7](../../course/P2-serving-engines/P2.7-gateway-basics/README.md), extended with tracing in P3.5, multi-tenancy in P3.8, and region failover in C1.
