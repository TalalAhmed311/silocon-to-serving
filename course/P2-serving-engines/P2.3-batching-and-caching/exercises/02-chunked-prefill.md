# Exercise 2 — Chunked prefill in the simulator (medium)

Add `chunk_tokens` to `continuous()`. Each step processes at most `chunk_tokens` tokens: one per decoding sequence first, with the remainder spent on prefill, FIFO among the sequences still prefilling. A long prompt now takes several steps to prefill, and its first token comes at the end of the step that finishes the prefill.

**Test:** with a trace that mixes many decoding sequences with occasional 4,000-token prompts:
- the worst ITL with `chunk_tokens=512` is ≤ the cost of a 512-token step plus a full decode batch, and much smaller than without chunking
- the long prompt's TTFT is *larger* with chunking. That is the trade-off, and the test asserts it so you notice it.
