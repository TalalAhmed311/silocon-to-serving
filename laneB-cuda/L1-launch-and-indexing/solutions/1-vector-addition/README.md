# 1 — Vector Addition (L1, easy)

Problem: [LeetGPU #1](../../leetgpu-map.md). It runs on T1 (the browser) or T2 (the harness).

**Hint ladder**

1. One thread owns one output element. What is its global index?
2. `i = blockIdx.x * blockDim.x + threadIdx.x`. The grid has `ceil(N / blockDim.x)` blocks: use `(N + 255) / 256`.
3. The last block has threads past `N`. Guard with `if (i < N)`, or every out-of-bounds thread writes garbage (or faults).

**Solution outline:** one kernel, 256 threads per block, `C[i] = A[i] + B[i]` behind a bounds check.

**Why this is fast (and why it can't go faster):** each element moves 12 bytes (2 reads + 1 write) for 1 FLOP, an intensity of 1/12. So the only achievable "peak" is memory bandwidth. Consecutive threads touch consecutive floats, so each warp's loads coalesce into 128-byte transactions. With `--bench` the test prints GB/s vs a measured copy kernel. Expect it to be close to the copy.
