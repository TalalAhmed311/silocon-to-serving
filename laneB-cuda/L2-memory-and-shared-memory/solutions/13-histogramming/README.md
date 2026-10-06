# 13 — Histogramming (L2)

Problem: [LeetGPU #13](../../leetgpu-map.md). Count how many inputs fall in each of `num_bins` bins. Our harness uses integer inputs in `[0, num_bins)`; check the statement's input type and range.

**Hint ladder**

1. Naive: `atomicAdd(&hist[x], 1)` in global memory. It is correct, but heavily contended bins (skewed data) serialize.
2. **Privatize**: each block keeps its own histogram in shared memory (shared atomics are much cheaper), then adds it to the global histogram once at the end. That makes one global atomic per bin per block instead of one per element.
3. Use a grid-stride loop with a modest grid (for example 4 × SMs blocks), so the merge cost is amortized over many elements per block.

**Solution outline:** shared `hist_s[num_bins]` (dynamic, ≤ 12K bins; above that, fall back to global atomics), a grid-stride `atomicAdd(&hist_s[x], 1)`, `__syncthreads()`, then `atomicAdd(&hist[b], hist_s[b])` for nonzero bins.

**Why this is fast:** contention moves from DRAM-backed global atomics to on-chip shared memory, and the global traffic drops to `blocks × bins` atomics. On uniform data the kernel is close to memory-bound. On extremely skewed data (all one bin), shared atomics still serialize within a block. Measure both with `--bench`.
