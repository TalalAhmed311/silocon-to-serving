# 10 — 2D Convolution (L2)

Problem: [LeetGPU #10](../../leetgpu-map.md). Valid 2-D correlation: `out[r][c] = Σ_{i<KR, j<KC} in[r+i][c+j] · k[i][j]`, with output `(R−KR+1) × (C−KC+1)`. Check the statement's convention.

**Hint ladder**

1. Naive: one thread per output pixel, a double loop over the kernel.
2. Tile: a 16×16 output block needs a `(16+KR−1) × (16+KC−1)` input tile including the halo. Load it cooperatively, since some threads load more than one element, then `__syncthreads()`.
3. Kernel in `__constant__`. Map x to columns so the global loads coalesce row by row.

**Solution outline:** a 16×16 output tile, a dynamic shared tile with a strided 2-D cooperative load, and kernel weights in `__constant__` (≤ 64×64 taps; larger kernels use global memory).

**Why this is fast:** each input pixel is loaded from DRAM once per tile instead of `KR·KC` times.
