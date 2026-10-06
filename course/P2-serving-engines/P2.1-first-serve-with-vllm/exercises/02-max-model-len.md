# Exercise 2 — Find the `--max-model-len` that fails, and explain why (T2)

Raise `MAX_MODEL_LEN` (an environment variable read by `01_serve.sh`) until vLLM refuses to start. Then:

1. Copy the error line from `vllm.log`.
2. Explain it with D3: one sequence at `max_model_len` must fit in the KV budget. Compute the largest length that fits with `--max-seqs 1` (`platform/capacity`). It should be within a few % of where vLLM failed.
3. Make that same length start successfully by changing **one** other flag. Name two flags that work, and their trade-offs. (Hints: KV dtype, memory utilization, model weights dtype.)
