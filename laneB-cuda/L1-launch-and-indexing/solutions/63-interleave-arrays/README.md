# 63 — Interleave Arrays (L1)

Problem: [LeetGPU #63](../../leetgpu-map.md). Given A and B of length N, write `out = [A0, B0, A1, B1, …]`.

**Hint ladder**

1. Thread `i` reads `A[i]` and `B[i]`. Where do they go?
2. `out[2i] = A[i]` and `out[2i+1] = B[i]`. The output index isn't the input index: a **scatter** with stride 2.
3. Faster: write one `float2{A[i], B[i]}` to `reinterpret_cast<float2*>(out)[i]`. That is one 8-byte store per thread, fully contiguous.

**Solution outline:** one thread per i, one `float2` store.

**Why this is fast:** with two scalar stores, each store instruction covers every other float in a 256-byte span. It still coalesces, but it issues twice as many store instructions. The `float2` version writes 256 contiguous bytes per warp in one instruction. 16 bytes per i, memory-bound.
