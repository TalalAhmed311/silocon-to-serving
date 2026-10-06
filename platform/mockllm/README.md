# mockllm: a fake LLM server that behaves like one

This is an OpenAI-compatible server (`/v1/completions` and `/v1/chat/completions`, with streaming via SSE) that has no model behind it. A **continuous-batching cost model** takes its place, so queueing, batching and KV-cache saturation look like a real engine's on a laptop:

- requests queue FIFO
- up to `max_num_seqs` run at once
- each scheduler step emits one token per running request
- step time grows with batch size and with admitted prompt tokens
- admission stops when the "KV cache" is full

It also exports **Prometheus metrics with vLLM's names** (`vllm:time_to_first_token_seconds`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, …). The dashboards, alerts and autoscalers you build against it in P3 then work unchanged against vLLM.

| Used by | For |
|---|---|
| P1.3 | measuring TTFT/ITL/goodput from a real HTTP stream |
| P2.3 #4 | finding the latency knee on T0 before you pay for a GPU |
| P2.7 #6 | gateway backends, retries and fallbacks (`MOCKLLM_FAIL_RATE` is planned in the #6 exercises) |
| P3.5 / P3.6 | Prometheus scraping, KEDA scaling on `vllm:num_requests_waiting` in a kind cluster |

```bash
uv run uvicorn mockllm.server:app --app-dir platform --port 8001
curl -s localhost:8001/v1/completions -H 'content-type: application/json' -d '{"prompt":"hello world","max_tokens":5}'
curl -s localhost:8001/metrics | grep vllm:
```

Configure it with `MOCKLLM_*` environment variables: `BASE_STEP_MS`, `PER_SEQ_MS`, `PREFILL_MS_PER_TOKEN`, `MAX_NUM_SEQS`, `KV_CAPACITY_TOKENS`, `TIME_SCALE`, `API_KEY` and `MODEL`. **The default costs are made up.** They have a realistic *shape*, but they are not a model of any particular GPU.
