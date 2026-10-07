# Exercise 2: chunked prefill (T0)

When `cfg.chunked_prefill` is true, schedule `n = min(seq.num_uncomputed, budget)` tokens instead of requiring the whole prompt to fit. A chunk only **produces a token** if `start + n == len(seq)` (that's `Chunk.produces_token`); the engine only samples for those.

Tests: `test_chunked_prefill_small_budget` (budgets 1, 3, 7 — a budget of 1 is the extreme case) and `test_chunked_prefill_lets_decodes_continue` (a 60-token prompt arrives; the running short request still gets a token every step).

**Then:** with budget B and D running decodes, how many prompt tokens per step does a new long prompt get? What's its TTFT in steps for a prompt of length L?
