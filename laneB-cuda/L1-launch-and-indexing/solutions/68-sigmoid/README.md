# 68 — Sigmoid Activation (L1)

Problem: [LeetGPU #68](../../leetgpu-map.md). T1 (the browser) or T2 (the harness).

**Hint ladder**

1. Elementwise: σ(x) = 1 / (1 + e^(−x)).
2. `expf` is accurate to about 2 ulp. `__expf` is faster but less accurate. Try both and compare against the tolerance.
3. For large negative x, `expf(-x)` overflows to inf and σ becomes 1/inf = 0, which is the correct limit. No special case is needed.

**Solution outline:** a grid-stride elementwise kernel, `out[i] = f(in[i])`, 256 threads per block.

**Why this is fast:** Still memory-bound. A transcendental per element is cheap next to 8 bytes of DRAM traffic. `__expf` (SFU, `--use_fast_math`) would not move the bandwidth-bound time.
