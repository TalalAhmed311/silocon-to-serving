# Exercise 2 — Multi-turn replay: measure the prefix-cache win (T2, T0 test)

1. **T0.** `test_multiturn_mock.py` runs `examples/02_multiturn.py`'s conversation logic against `platform/mockllm`. The mock has **no** prefix cache, and its prefill cost is linear in prompt tokens, so TTFT must grow with the turn number. The test checks that it does. This validates your measurement before you trust it on a GPU.
2. **T2.** Run the replay against SGLang (radix cache on), SGLang with `--disable-radix-cache`, vLLM with prefix caching on, and vLLM with it off. Fill in:

| engine | prefix cache | turn-1 p50 TTFT | turn-6 p50 TTFT | growth |
|---|---|---|---|---|

3. Explain the growth column with P1.2's prefill model: TTFT ≈ queue + (uncached prompt tokens) × per-token prefill cost.
