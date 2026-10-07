# 2 — Matrix Multiplication, naive (L1 exit check)

Problem: [LeetGPU #2](../../leetgpu-map.md). `C[M×K] = A[M×N] · B[N×K]`, row-major. **Check the statement for which letter is the inner dimension.** Our harness uses `A: M×N, B: N×K, C: M×K`.

This is the L1 **exit check**: write the 2-D indexing for *any* shape without help. The harness tests 20 random non-square shapes.

**Hint ladder**

1. One thread per output element `C[row][col]`. That needs a 2-D grid covering M rows × K columns.
2. `col = blockIdx.x*blockDim.x + threadIdx.x` (x → columns, so the warp's B reads and C writes coalesce), and `row = blockIdx.y*blockDim.y + threadIdx.y`.
3. The inner loop is `sum += A[row*N + i] * B[i*K + col]`. Guard `row < M && col < K`. `grid.x` covers K and `grid.y` covers M.

**Solution outline:** a 16×16 block per 16×16 tile of C, one dot product of length N per thread.

**Why it's slow (on purpose):** each thread reads a full row of A and a full column of B from global memory. Every element of A is re-read K times and every element of B M times, so the arithmetic intensity is ~0.25 FLOP/byte. L2 and L5 fix this with shared-memory tiling and then the full GEMM ladder.
