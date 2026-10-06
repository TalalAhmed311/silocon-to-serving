# Exercise 4 — Reproduce `vllm bench latency` with your own client (T2, with a T0 test)

`vllm bench latency --batch-size 1 --input-len 128 --output-len 128` runs the engine **offline** (no HTTP) and reports the latency of one full request. Write `my_latency.py`: send the same request shape over HTTP to the running server (128-token prompt, 128 output tokens, `ignore_eos`, temperature 0), 10 times after a warm-up, and print the median end-to-end latency.

1. **T0 test first:** `test_client_vs_mock.py` runs your `measure_e2e()` against `platform/mockllm`. The mock's cost model is known, so the expected latency can be computed exactly (≈ prefill + 128 steps).
2. **T2:** compare your HTTP median with `vllm bench latency`. Expect HTTP to be slightly slower: serialization, detokenization and network. Within 10% is good. More than that means something is wrong (warm-up? `ignore_eos`?).

Starter: [`my_latency.py`](my_latency.py). Reference: [`solutions/my_latency.py`](solutions/my_latency.py).
