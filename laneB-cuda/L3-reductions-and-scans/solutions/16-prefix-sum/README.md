# 16: Prefix Sum (L3, core)

Problem: [LeetGPU #16](../../leetgpu-map.md). Inclusive scan: `out[i] = in[0] + … + in[i]`.

**Hint ladder**

1. Within a block, Hillis–Steele (`s[t] += s[t − o]` for o = 1, 2, 4, …) is simple, but it does O(n log n) work. Blelloch's up-sweep/down-sweep is work-efficient at O(n).
2. Have each thread scan **several consecutive items serially** (8 here), then scan only the 256 per-thread totals across the block. Most of the work becomes register-serial and cheap.
3. Across blocks, use **reduce-then-scan** in three phases: scan each tile and write its total, scan the totals (recursively), then add each tile's carry-in. That's 3 kernel launches per level and about 3N memory traffic (read, write, read+write).
4. State of the art is **decoupled look-back** (CUB's single pass, ~2N traffic). Each tile publishes its aggregate with a status flag, and the next tile looks back over predecessors instead of waiting for a global pass. Read the Merrill–Garland paper, then compare against `cub::DeviceScan` with `--bench`.

**Solution outline:** the generic `inclusive_scan(in, out, n, op, identity)` in `kernel.cu`. It's coalesced tile loads into shared memory, an 8-item serial scan per thread, a block scan of thread totals, and the 3-phase recursion. The same code (with a different `op`) solves #70, #72, #82 and #110.

**Why this is fast (and where it isn't):** each phase is coalesced and memory-bound, but the 3-phase structure moves ~1.5× the bytes of a single-pass scan. The bench row against CUB shows that gap. `TODO(run-on: g4dn.xlarge)`.
