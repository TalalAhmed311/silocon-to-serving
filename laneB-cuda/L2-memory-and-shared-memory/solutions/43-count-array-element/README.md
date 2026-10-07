# 43 — Count Array Element (L2)

Problem: [LeetGPU #43](../../leetgpu-map.md). Count the elements equal to `K`.

**Hint ladder**

1. `if (a[i] == K) atomicAdd(out, 1)` works, but every match is a global atomic on **one** address, the worst possible contention.
2. Count per thread in a register (grid-stride loop), then reduce per warp with `__reduce_add_sync` (sm_80+) or `__shfl_down_sync`, and do **one** `atomicAdd` per warp.
3. Even better: one per block, through shared memory. With a few hundred blocks, the global atomics become negligible.

**Solution outline:** grid-stride count in a register, warp reduce with shuffles, lane 0 does `atomicAdd`. That is one atomic per warp.

**Why this is fast:** it is memory-bound (4 bytes read per element), and the atomics drop from "one per match" to "one per warp".
