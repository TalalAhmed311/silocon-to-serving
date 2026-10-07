# 66 — RGB to Grayscale (L1)

Problem: [LeetGPU #66](../../leetgpu-map.md). Input: interleaved RGB floats, 3 per pixel (array-of-structs). Output: one float per pixel, a weighted sum of R, G and B.

> **Use the exact weights the problem statement gives.** Our harness uses the common ITU-R BT.601 luma weights (0.299, 0.587, 0.114) as a stand-in, defined in `kernel.cu` as `kR, kG, kB`. Change them if the statement differs.

**Hint ladder**

1. One thread per *pixel*. Pixel `p` reads bytes at `3p, 3p+1, 3p+2`.
2. A warp reads 96 consecutive floats (3 × 32): still contiguous, so the 3 loads coalesce into about 3 transactions together.
3. Use `__restrict__` and `const`, and let the compiler use the read-only cache path.

**Solution outline:** `out[p] = kR·in[3p] + kG·in[3p+1] + kB·in[3p+2]`.

**Why this is fast:** 16 bytes moved per pixel for 5 FLOPs, memory-bound. The stride-3 access pattern *looks* uncoalesced per instruction, but across the three loads the warp touches one contiguous 384-byte span, and L1 merges the sectors.
