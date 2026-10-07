# 84: SwiGLU MLP Block (L5, practice)

Problem: [LeetGPU #84](../../leetgpu-map.md). A Llama MLP: `(silu(x·W_g) ⊙ (x·W_u))·W_d`.

**Hint ladder**

1. Unfused, it's 3 GEMMs + 1 elementwise kernel, and two `M×f` intermediates (f ≈ 3.5·d) round-trip through HBM.
2. Compute the gate and up tiles for the **same output block** in one kernel, and apply `silu(g)·u` in the epilogue. Only `h` is written.
3. Better: concatenate `[W_g | W_u]` into one `d × 2f` matrix (what vLLM's `MergedColumnParallelLinear` does). Then it's one GEMM whose epilogue pairs columns n and n + f. That shares the x tile loads, unlike this solution's two `gemm_tile` calls.
4. The down projection is a plain GEMM. Fusing it too needs the whole `h` row, so it doesn't fit a tile, and engines stop at step 3.

**Solution outline:** `gate_up_swiglu` (two tile products + a fused epilogue) → `gemm_plain`.

**Why it matters:** the MLP is ~2/3 of a dense model's FLOPs. At decode (small M) it's memory-bound on the weights, so fusion matters less than weight bytes. That's quantization's job (P2.5, #81).
