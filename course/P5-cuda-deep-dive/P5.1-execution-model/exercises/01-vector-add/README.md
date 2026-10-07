# Exercise 1: grid-stride vector add for any N (T2)

Write `vadd(a, b, c, n)` in `kernel.cuh`: a kernel with a **grid-stride loop**, launched with a grid sized to the GPU (about 8 blocks per SM of 256 threads), not with `ceil(n / 256)` blocks.

The test checks n = 1, 31, 32, 1000, 2²⁰ and 2²⁴ + 3, with exact equality. `--bench` reports GB/s (3 arrays × 4 B per element) against 1.5× the measured copy bandwidth.

**Questions:** why is the grid-stride version robust to n > 2³¹ when you use `long long` indices? What would `ceil(n/256)` blocks do at n = 2³⁶? Then switch to the `float4` version (`d4::vadd_float4`). Does GB/s change? Why or why not, for a kernel that's already memory-bound?
