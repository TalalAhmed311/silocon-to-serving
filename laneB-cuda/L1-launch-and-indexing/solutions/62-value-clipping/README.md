# 62 — Value Clipping (L1)

Problem: [LeetGPU #62](../../leetgpu-map.md). T1 (the browser) or T2 (the harness).

**Hint ladder**

1. Each output depends on one input and two scalars.
2. Clamp = `fminf(fmaxf(x, lo), hi)`, two branch-free instructions.
3. Check the edge cases: values exactly equal to lo or hi, and lo == hi.

**Solution outline:** a grid-stride elementwise kernel, `out[i] = f(in[i])`, 256 threads per block.

**Why this is fast:** 8 bytes per element (one read and one write) for a few FLOPs: memory-bound. The goal is to match the measured copy bandwidth (`--bench`). Coalesced unit-stride access gets you there; nothing about the arithmetic matters.
