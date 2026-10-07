# 54: Swish-Gated Linear Unit (L4, core)

Problem: [LeetGPU #54](../../leetgpu-map.md). `out = silu(a) ⊙ b`, where `[a | b]` are the two halves of the input. In a Llama MLP these are `x·W_gate` and `x·W_up`.

**Hint ladder**

1. It's elementwise and memory-bound: 2 reads and 1 write per output. Grid-stride with a GPU-sized grid.
2. `silu(x) = x / (1 + e^{−x})`. `__expf` is plenty accurate here (a few ulp).
3. `float4` loads for both halves when `n % 4 == 0` (alignment of the second half depends on it).
4. The real win is **fusion**: compute `silu(gate)·up` in the epilogue of the GEMM that produces `gate` and `up`, so they never reach HBM (L5 #84).

**Solution outline:** `swiglu_k` (float4) / `swiglu_scalar` (any n).

**Why this is fast:** at bandwidth, 12 bytes per output. Measure GB/s vs copy.
