# Exercise 1 — Continuous batching in the simulator (medium)

Implement `continuous(reqs, max_batch, cost)` in [`batchsim.py`](batchsim.py), without `chunk_tokens` for now. Every step:

1. Admit arrived requests in FIFO order while fewer than `max_batch` are running.
2. Requests admitted this step are prefilled whole. Their prompt tokens add to the step cost, and they emit their first token at the end of the step.
3. Every other running request emits one token.
4. A request that has emitted all its tokens leaves **immediately**.
5. If nothing is running, jump the clock to the next arrival.

**Test (`test_batchsim.py`):** on 3 seeded traces, continuous has lower mean E2E latency than `static(…, batch=max_batch)`. It never runs more than `max_batch` sequences. Every request gets exactly `output` tokens, and the first token never arrives before the request does.
