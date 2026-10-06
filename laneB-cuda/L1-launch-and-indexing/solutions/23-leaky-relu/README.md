# 23 — Leaky ReLU (L1)

Problem: [LeetGPU #23](../../leetgpu-map.md). T1 (the browser) or T2 (the harness).

**Hint ladder**

1. Elementwise again. The only new thing is the slope parameter for negative inputs.
2. `x > 0 ? x : alpha * x` compiles to a select, not a branch, so no warp divergence.
3. Pass alpha by value as a kernel argument. Kernel arguments live in constant memory and are broadcast to every thread for free.

**Solution outline:** a grid-stride elementwise kernel, `out[i] = f(in[i])`, 256 threads per block.

**Why this is fast:** 8 bytes per element (one read and one write) for a few FLOPs: memory-bound. The goal is to match the measured copy bandwidth (`--bench`). Coalesced unit-stride access gets you there; nothing about the arithmetic matters.
