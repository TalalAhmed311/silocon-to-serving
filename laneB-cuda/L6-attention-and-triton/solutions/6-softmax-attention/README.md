# 6: Softmax Attention (L6, core; exit check in P5.8)

Problem: [LeetGPU #6](../../leetgpu-map.md). `softmax(QKᵀ/√d)·V` for one head.

**Hint ladder:** naive (materialize the M×N scores) → tiled → FA-1 → FA-2, exactly P5.8's sequence. Keep `(m, l, o)` per query row, stream K/V tiles through shared memory, and rescale once per tile. The **L6 exit check** (FA-2 ≥ 5× naive at seq 4k) is measured with `p5.8_attention_bench`.

**Solution outline:** [`_shared/attn_core.cuh`](../_shared/attn_core.cuh), the one FA-2-style kernel all the L6 attention solutions share. This problem uses it with no masks. For LeetGPU, paste the core and this `solve` together.

**Why it's fast:** O(N·d) HBM traffic instead of O(N²). The N×N matrix only ever exists one 64×32 tile at a time, in registers.
