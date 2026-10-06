# 31 — Matrix Copy (L1)

Problem: [LeetGPU #31](../../leetgpu-map.md). Copy an N×N row-major matrix.

**Hint ladder**

1. Use a 2-D launch: `dim3 block(16, 16)` and `dim3 grid(ceil(N/16), ceil(N/16))`.
2. `row = blockIdx.y * blockDim.y + threadIdx.y` and `col = blockIdx.x * blockDim.x + threadIdx.x`. **x must map to columns**, so that consecutive threads in a warp touch consecutive addresses.
3. Check both `row < N` and `col < N`.

**Solution outline:** a 2-D grid, `B[row*N + col] = A[row*N + col]`.

**Why this is fast:** with `threadIdx.x → col`, a warp reads 32 consecutive floats, fully coalesced. Swap the mapping (x → row) and every lane hits a different row, 32 separate transactions per warp. The bench prints both, so try it. This exact mistake is what L2's transpose problem is about.
