# 21 — ReLU (L1)

Problem: [LeetGPU #21](../../leetgpu-map.md). T1 (the browser) or T2 (the harness).

**Hint ladder**

1. Same indexing as Vector Addition: one thread, one element.
2. `fmaxf(x, 0.0f)` is a single instruction. A branch would also work, because both sides are trivial.
3. Use a grid-stride loop so one launch size handles any N.

**Solution outline:** a grid-stride elementwise kernel, `out[i] = f(in[i])`, 256 threads per block.

**Why this is fast:** 8 bytes per element (one read and one write) for a few FLOPs: memory-bound. The goal is to match the measured copy bandwidth (`--bench`). Coalesced unit-stride access gets you there; nothing about the arithmetic matters.
