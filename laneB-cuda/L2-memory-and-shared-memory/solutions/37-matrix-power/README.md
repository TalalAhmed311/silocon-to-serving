# 37 — Matrix Power (L2, stretch)

Problem: [LeetGPU #37](../../leetgpu-map.md). Compute `A^P` for a square `N × N` matrix.

**Hint ladder**

1. P − 1 multiplications works, but it is O(P) kernel launches.
2. **Exponentiation by squaring** needs only O(log P) multiplications: square the base, and multiply into the result when the bit is set.
3. Reuse the tiled matmul (#2). You need three device buffers (result, base, temp) and pointer swaps, never copies. Mind the floating-point error growth for large P: the test uses small values and P ≤ 16.

**Solution outline:** the binary method with ping-pong buffers, every product done by `matmul_tiled`.

**Why this is the point:** at small N, launch overhead and latency dominate (a 64×64 matmul is a few µs), so cutting launches from P to log₂P is the real win. This is the same logic as CUDA graphs in P2.6.
