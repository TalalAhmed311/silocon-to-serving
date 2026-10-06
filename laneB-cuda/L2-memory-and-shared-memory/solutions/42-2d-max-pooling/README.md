# 42 — 2D Max Pooling (L2, practice)

Problem: [LeetGPU #42](../../leetgpu-map.md). For each `k×k` window with stride `s`, output its maximum. Harness convention: no padding, output `((H−k)/s + 1) × ((W−k)/s + 1)`. **Check the statement** for padding, dilation and channel layout.

**Hint ladder**

1. One thread per output: the window starts at `(oy·s, ox·s)`. Initialize with `-INFINITY`, not 0, because inputs can be negative.
2. When `s < k`, windows overlap, so neighbouring threads re-read the same inputs. A shared tile with halo helps (as in #10), but for small k the L1 cache already does most of the work. Measure before you optimize.
3. Use `fmaxf` (branch-free, and NaN behavior defined by IEEE `maxNum`).

**Solution outline:** the direct version, with x → output column so the warp's reads coalesce along rows.
