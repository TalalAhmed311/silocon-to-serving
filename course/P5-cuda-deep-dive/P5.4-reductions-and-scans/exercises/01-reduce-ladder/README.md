# Exercises 1 + 2: the reduction ladder, and the L3 exit check (T2)

Rungs 1 and 2 are given in `kernel.cuh`. Add rungs 3–6 behind the same `reduce_sum(rung, …)` launcher:

| rung | change | what it fixes |
|---|---|---|
| 3 | sequential addressing: `if (tid < st) s[tid] += s[tid + st]`, with st halving | divergence (rung 1) and bank conflicts (rung 2) |
| 4 | each thread adds **two** elements while loading | half the threads idle at step 1, half the blocks |
| 5 | finish with warp shuffles (`__shfl_down_sync`), one shared slot per warp | the last 5 `__syncthreads` steps |
| 6 | grid-stride loop, `float4` loads, one `atomicAdd` per block, no second kernel | launch overhead, too many partials, too few bytes in flight |

The test checks every rung on awkward sizes (1, 300, 513, 2²² + 5) against a double-precision sum with a magnitude-scaled tolerance (float addition order differs).

**Exercise 2 (exit check):** `--bench` prints each rung's GB/s at 2²⁴, 2²⁶ and 2²⁸, then `rung 6 / cub` and `PASS` if it's within 10% of `cub::DeviceReduce::Sum` (median of 100 runs). Record the table in the README bench. `TODO(run-on: g4dn.xlarge)`
