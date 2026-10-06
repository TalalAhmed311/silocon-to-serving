# Exercise 3 — GuideLLM vs #4 vs `vllm bench serve` (T2)

Run the three commands in [aws.md](../aws.md) against the **same** server and fill in:

| tool | offered req/s | n | p50 TTFT | p90 TTFT | p50 TPOT/ITL | output tok/s | how it counts tokens | TTFT measured from |
|---|---|---|---|---|---|---|---|---|
| loadgen (#4) | 4 | | | | | | words (approx) / usage | request send |
| GuideLLM | 4 | | | | | | | |
| vllm bench serve | 4 | | | | | | | |

For every pair that differs by more than 10%, identify the cause from the tools' source code. Point at the file and function (for example in `guidellm/src/guidellm/benchmark/` and `vllm/vllm/benchmarks/`). Then fix #4, or document the difference in your report template.
