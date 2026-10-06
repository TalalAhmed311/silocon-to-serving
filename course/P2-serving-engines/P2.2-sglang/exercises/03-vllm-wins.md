# Exercise 3 — Find a workload where vLLM wins, and explain it (T2)

Using P2.4's harness (or `P1.3/bench/mock_load.py` pointed at each server), find one workload where vLLM beats SGLang on goodput at the same memory fraction, and one where the opposite holds. Candidates to try:

- single-turn, random prompts (no shared prefixes)
- very long prompts with short outputs
- many short requests at high rate
- structured or JSON output (with each engine's guided decoding)

Write a half-page analysis that links the result to a mechanism you can point at in each engine's code or docs: scheduler policy, chunked-prefill defaults, kernel backend, cache design. "Engine X is faster" without a mechanism doesn't count.
