# Exercise 4 (hard): 2D convolution with halo tiles (T2)

Implement `conv2d` in `kernel.cuh`: K×K (odd K ≤ 7), zero padding, same-size output.

1. Start naive: one thread per output, K² global reads each.
2. Then tile: each 16×16 block loads a `(16 + K − 1)²` input tile **including the halo** into shared memory once, and keeps the filter in `__constant__` memory (all threads read the same weight: a broadcast).
3. Reuse: count global reads per output for naive vs tiled at K = 7, and compare with the `--bench` speedup.

The test compares with a double-precision CPU reference (`rtol = atol = 1e-5`) on odd shapes. L2's LeetGPU #10 is the same problem, so compare with your L2 solution.
