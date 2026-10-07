# 52 — Sigmoid Linear Unit (L1)

Problem: [LeetGPU #52](../../leetgpu-map.md). T1 (the browser) or T2 (the harness).

**Hint ladder**

1. SiLU(x) = x · σ(x), the activation inside SwiGLU (P0.5's MLP).
2. Fold it into one expression: `x / (1 + expf(-x))`.
3. Compare against the CPU at rtol 1e-6. Fused vs unfused versions round differently in the last bit.

**Solution outline:** a grid-stride elementwise kernel, `out[i] = f(in[i])`, 256 threads per block.

**Why this is fast:** 8 bytes per element (one read and one write) for a few FLOPs: memory-bound. The goal is to match the measured copy bandwidth (`--bench`). Coalesced unit-stride access gets you there; nothing about the arithmetic matters.
