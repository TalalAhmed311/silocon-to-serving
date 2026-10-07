# P0.4 quiz

<details><summary><b>1.</b> An FMA has 4-cycle latency and the core has 2 FMA units. How many independent accumulators does a dot product need to reach peak, and what fraction of peak does one accumulator give?</summary>

**8** (latency × units). One accumulator is a dependency chain at 1 FMA per 4 cycles, which is **1/8** of peak.
</details>

<details><summary><b>2.</b> Compute the arithmetic intensity of fp32 SAXPY <code>y = a·x + y</code>.</summary>

2 FLOPs per element (a mul and an add, or one FMA). Bytes: read x (4), read y (4), write y (4) = 12. **I = 2/12 ≈ 0.17 FLOP/byte**, so it is memory-bound on any CPU or GPU.
</details>

<details><summary><b>3.</b> Why can matmul be compute-bound when matvec never is?</summary>

Matmul does 2n³ FLOPs on 3n² data, so its intensity grows like n. Matvec does 2mn FLOPs on mn data (the matrix dominates), so its intensity stays near 0.5 regardless of size. Every matrix element is used once.
</details>

<details><summary><b>4.</b> Your machine has P = 400 GFLOP/s and B = 50 GB/s. Where is the ridge, and what is the ceiling for a kernel with I = 2?</summary>

Ridge I* = 400/50 = **8 FLOP/byte**. At I = 2 the kernel is memory-bound, so the ceiling is 2 × 50 = **100 GFLOP/s**.
</details>

<details><summary><b>5.</b> Why does reordering the SGEMM loops from i,j,k to i,k,j help so much?</summary>

In i,j,k the inner loop walks B down a column with stride N: one useful float per cache line, and it can't vectorize. In i,k,j the inner loop streams a row of B and a row of C contiguously, so every line is fully used, and it auto-vectorizes as a SAXPY.
</details>

<details><summary><b>6.</b> In the 4×16 AVX2 micro-kernel, what are the loads and FLOPs per k step, and why is that good?</summary>

4 broadcasts of A + 2 vector loads of B = 6 loads, for 8 FMAs = 128 FLOPs. C is not loaded or stored inside the K loop at all. The register tile turns memory traffic into register reuse.
</details>

<details><summary><b>7.</b> In the AVX2 int8 dot product, why the <code>sign_epi8</code> trick before <code>maddubs</code>?</summary>

`maddubs` multiplies an **unsigned** byte vector by a **signed** one. Taking |a| (unsigned) and moving a's sign onto b keeps the product a·b exact.
</details>

<details><summary><b>8.</b> Why does q8_0 quantize to [−127, 127] instead of [−128, 127]?</summary>

The range is symmetric for a symmetric scale. It also avoids int16 saturation in `maddubs`, where two adjacent products of 128·128 = 32768 would overflow.
</details>

<details><summary><b>9.</b> A SIMD add is 8× faster than scalar for a 4 KB array but only 1.1× faster for a 256 MB one. Why?</summary>

At 256 MB the kernel is limited by DRAM bandwidth (intensity 1/12). Both versions wait on memory, and SIMD only speeds up the arithmetic, which was never the bottleneck.
</details>
