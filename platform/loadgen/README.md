# loadgen: #4, the continuous-batching load test

An **open-loop** generator for OpenAI-compatible servers. It provides:

- seeded Poisson arrivals with lognormal prompt and output lengths, and optional shared prefixes
- per-request TTFT, TPOT and ITL
- SLO-based goodput
- the server's own Prometheus gauges (KV usage, waiting queue, preemptions), scraped around each step
- **knee detection**: the largest offered rate whose p90 TTFT meets the SLO with goodput ≥ 90% of offered

```bash
PYTHONPATH=platform uv run python -m loadgen.cli --url http://127.0.0.1:8000 --rates 1 2 4 8 16 32 --duration 60 --ttft-slo 2
PYTHONPATH=platform uv run python -m loadgen.plot results/loadgen.json
```

Built in [P2.3](../../course/P2-serving-engines/P2.3-batching-and-caching/README.md). Reused by:

- #7 (quantization bakeoff)
- #10 (speculative decoding)
- #12 (custom kernel before/after)
- #0 v1 (vs vLLM)
- #15 (public teardown)
