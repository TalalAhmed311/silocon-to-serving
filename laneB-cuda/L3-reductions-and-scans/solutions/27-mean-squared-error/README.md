# 27: Mean Squared Error (L3, practice)

Problem: [LeetGPU #27](../../leetgpu-map.md). `mse = (1/N) Σ (p[i] − t[i])²`.

**Hint ladder**

1. It's the dot-product pattern: fuse the difference and the square into the load loop.
2. Divide by N **once**, at the end, not per element.
3. Try a **deterministic** two-pass version: per-block partials to an array, then a single-block pass that sums them and divides. Compare its speed with the atomic version. The second pass reads only a few hundred floats.

**Solution outline:** `sq_err_partials` (grid-stride `fmaf(d, d, acc)` → block shuffle-reduce → `partial[block]`) → `finalize_mean` (one 1024-thread block).

**Why this is fast:** a single read pass over both inputs, with the second launch costing a few microseconds. The test also checks that two runs give **bitwise-identical** results, which the atomic version can't guarantee.
