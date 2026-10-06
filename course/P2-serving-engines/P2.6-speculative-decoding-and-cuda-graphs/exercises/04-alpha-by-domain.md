# Exercise 4 — α by domain (T2)

Measure vLLM's n-gram spec decode (and your draft model, if you have one) on three prompt sets of ≥ 30 prompts each:

1. **code editing**: `examples/prompts_code.txt`, extended
2. **open chat**: generic questions
3. **RAG-style**: a long context passage plus a question whose answer quotes the passage

Report α (from vLLM's spec-decode metrics at `/metrics`; find the names in `docs/features/speculative_decoding/` at the pinned SHA), tokens per target pass, and ITL at batch 1 and batch 16. Explain the ranking in terms of what each proposer can predict. Then say when you would turn spec decode **off** in production.
