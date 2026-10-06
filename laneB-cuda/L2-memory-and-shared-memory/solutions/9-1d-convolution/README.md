# 9 — 1D Convolution (L2)

Problem: [LeetGPU #9](../../leetgpu-map.md). `out[i] = Σ_j in[i + j] · k[j]` for `j < K`. This is the "valid" convolution: output length `N − K + 1`. **Check the statement** for valid vs same padding and for correlation vs flipped-kernel convolution; our harness uses valid correlation.

**Hint ladder**

1. Naive: one thread per output and a loop over K. Every input element is read K times from global memory (served mostly by L1/L2, but still).
2. Shared memory: a block of `B` outputs needs inputs `[base, base + B + K − 1)`, so load that **tile plus halo** cooperatively, `__syncthreads()`, and compute from shared memory.
3. Put the kernel weights in `__constant__` memory: every thread reads `k[j]` for the same `j` at the same time, which the constant cache **broadcasts** to the whole warp in one transaction.

**Solution outline:** 256 outputs per block, a shared tile of `256 + K − 1` floats loaded with a strided loop, and the weights in `__constant__` (up to 2048 taps; larger kernels fall back to a global-memory version).

**Why this is fast:** every input element is read from DRAM once per block instead of K times, and the weight reads cost one broadcast per warp per tap. For large K it becomes compute-bound: `2K` FLOPs per output.
