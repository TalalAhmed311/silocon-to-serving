# 115: Fused QKV Projection with RoPE and KV Cache Update (L5, stretch: hints and outline only)

Problem: [LeetGPU #115](../../leetgpu-map.md). One kernel: `[q | k | v] = x·W_qkv`, RoPE on q and k, and k, v **written straight into the KV cache** at each token's slot.

**Hint ladder**

1. Start from #22's tile core on the concatenated `W_qkv` (d × (H + 2·Hkv)·D). Each output column belongs to q, k or v, and to a head and a dim within it, all computable from the column index.
2. The RoPE epilogue needs **pairs** (i, i + D/2) in the same thread (#61). Choose the thread tile so a thread owns both halves of a head's dims, or exchange through shared memory before storing.
3. The KV-cache write is a scatter. For token t (position p, sequence s), the k/v head slices go to `cache[block_table[s][p / B]][p % B][h][:]` (P5.8 exercise 4's layout), and q goes to the normal output.
4. Correctness test: compare with three separate steps (GEMM → RoPE → scatter). Bitwise equal is achievable if the fp32 math order matches.

**Solution outline:** a GEMM tile → a RoPE epilogue on the k and q columns → a split store (q to output, k/v to paged cache slots). This fusion is what serving engines do to keep the per-token overhead of attention's prologue small (P6).
