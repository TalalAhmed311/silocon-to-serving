# 4: Reduction (L3, core, exit check)

Problem: [LeetGPU #4](../../leetgpu-map.md). Sum N floats into one value.

**Hint ladder** (Mark Harris's reduction ladder, compressed)

1. Tree reduction in shared memory with `s[t] += s[t + stride]` and `__syncthreads()` between steps. Make sure you use *sequential* addressing (no divergent `t % (2*stride)`) and no bank conflicts.
2. Make each thread do more work first. A **grid-stride loop** with `float4` loads accumulates many elements in a register before any cooperation. This is the biggest single win: the reduction becomes memory-bound.
3. Replace the last steps, then all the shared-memory steps, with **warp shuffles** (`__shfl_down_sync`): 5 shuffles per warp, one shared slot per warp, and one more warp reduction.
4. Launch only ~8 × SMs blocks and finish with **one `atomicAdd` per block**, instead of a second kernel.

**Solution outline:** `reduce_sum` = grid-stride `float4` accumulate → `block_sum` (warp shuffles → per-warp smem → warp 0 shuffles) → `atomicAdd(out, block_total)`.

**Why this is fast:** every byte is read exactly once with 16-byte coalesced loads, and the cooperation cost is ~10 shuffles plus one atomic per block. The kernel runs at close to read bandwidth, which is the same bound CUB hits.

**Determinism:** float atomics make the result depend on block completion order, so the last bits can change between runs. If you need bitwise reproducibility, write per-block partials and reduce them in a second, single-block pass. The exit check bench prints the ratio to `cub::DeviceReduce::Sum` (`TODO(run-on: g4dn.xlarge)`).
