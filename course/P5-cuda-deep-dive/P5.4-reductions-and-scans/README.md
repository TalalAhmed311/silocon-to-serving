# P5.4: Reductions and scans (D4)

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) (sm_75+; `g4dn.xlarge` in [aws.md](aws.md)) |
| **Time** | ≈30 min reading + ≈10 h hands-on |
| **Prerequisites** | P5.1–P5.3. Lane B L3 is the hands-on twin of this module |
| **Reading** | Mark Harris, *Optimizing Parallel Reduction in CUDA* · PMPP ch. 10–11 · Merrill & Garland, *Single-pass Parallel Prefix Scan with Decoupled Look-back* (2016) |
| **D4** | `platform/kernels/include/d4/reduce.cuh`, `scan.cuh` |

## Learning objectives

1. Climb the reduction ladder and explain each rung's speedup with one ncu metric.
2. Use warp-level primitives (`__shfl_*_sync`, `__syncwarp`, cooperative groups) correctly under independent thread scheduling.
3. Write a work-efficient block scan, then extend it to arbitrary n with scan-then-propagate.
4. Explain why CUB's single-pass **decoupled look-back** scan beats the 3-phase scan, and implement it.
5. Hit the **L3 exit check**: reduction within 10% of `cub::DeviceReduce::Sum`.

---

## 1. The reduction ladder

| rung | idea | bottleneck it removes |
|---|---|---|
| 1 | tree in shared memory, `tid % (2s) == 0` | — (baseline: divergent, most lanes idle) |
| 2 | strided index `2·s·tid` | divergence, but bank conflicts appear |
| 3 | sequential addressing | bank conflicts. Active threads are contiguous |
| 4 | add two elements while loading | idle threads in the first step |
| 5 | warp shuffles for the last steps | `__syncthreads` and smem traffic in the tail |
| 6 | grid-stride + `float4` + one atomic per block | per-element launch/partials overhead. Enough bytes in flight |

The big jumps are 3 (memory-access shape) and 6 (each thread doing **many** elements). After that, the kernel sits at read bandwidth, the same roofline CUB hits. The animation [`p5-reduction.html`](../../../animations/p5-reduction.html) contrasts the shared-memory tree with the shuffle tree.

**Warp-synchronous code since Volta.** Lanes aren't guaranteed to run in lockstep. Use `__shfl_*_sync(mask, …)` and `__syncwarp()`, never the old "volatile shared memory, no sync" unroll.

## 2. Scans

| algorithm | work | depth | note |
|---|---|---|---|
| Hillis–Steele | O(n log n) | log n | simple. Fine inside a warp (5 shuffle steps) |
| Blelloch (up-sweep/down-sweep) | O(n) | 2 log n | work-efficient across a block |
| serial per thread + scan of thread totals | O(n) | small | what D4 does: 8 items per thread in registers |
| scan-then-propagate (3 phases) | O(n) | 3 launches | ~4n memory traffic |
| **decoupled look-back** (single pass) | O(n) | 1 launch | ~2n traffic: each tile publishes its aggregate, successors look back |

Decoupled look-back is subtle. Tile ids must come from an **atomic counter**, not `blockIdx`, so a tile only waits on tiles that have already started. Status words pack (flag, value) into 64 bits, so they're read and written atomically. `d4/scan.cuh` has it with comments. The test runs it repeatedly, because races show up as flaky results.

> **Predict first.** At 2²⁶ floats, rung 1 vs rung 6: how many global-memory bytes does each move? (Rung 1 also writes partials and launches a second kernel.) If both were purely bandwidth-bound they'd tie. Rung 1 is several times slower, so what is it bound by? Check with ncu's Speed-of-Light.

---

## Walkthrough

See [aws.md](aws.md): run the three examples, then the exercises with `--bench`, then ncu on rungs 2 and 3 (bank conflicts) and rungs 5 and 6 (bytes in flight).

## What you should see

- `01`: GB/s rising rung by rung. Rung 6 and CUB both near half the copy bandwidth (read-only).
- `02`: all three warp reductions agree.
- `03`: 3-phase < look-back ≈ CUB.

`TODO(run-on: g4dn.xlarge)`

## Bench

| n | rung 1 | 2 | 3 | 4 | 5 | 6 | CUB | rung 6 / CUB |
|---|---|---|---|---|---|---|---|---|
| 2²⁴ | | | | | | | | `TODO(run-on: g4dn.xlarge)` |
| 2²⁶ | | | | | | | | |
| 2²⁸ | | | | | | | | (exit: ≤ 1.10) |

## Exercises

| # | Exercise | Test |
|---|---|---|
| 1 | [Each rung passes the harness and reports GB/s](exercises/01-reduce-ladder/README.md) | `p5.4_01-reduce-ladder` |
| 2 | [Reduction within 10% of CUB (L3 exit check)](exercises/01-reduce-ladder/README.md) | `p5.4_01-reduce-ladder --bench` prints PASS |
| 3 | [Block scan + scan-then-propagate for large arrays](exercises/03-scan/README.md) | `p5.4_03-scan` (exact) |
| 4 | *(hard)* [Top-k via per-block selection + merge](exercises/04-topk.md) | your bench table |

## Common mistakes

- `__syncthreads()` inside `if (tid < s)`: a deadlock or undefined behavior. The barrier must be reached by every thread.
- Relying on implicit warp lockstep (pre-Volta "volatile" tricks).
- Float atomics and "the result changed in the last digit". Order-dependence is expected, so test with tolerances or use the deterministic two-pass version.
- Benchmarking a reduction on data that fits in L2.

## Go deeper

- Harris's slides (classic). PMPP ch. 10 (reduction), 11 (scan).
- CCCL/CUB source: `cub/agent/agent_scan.cuh` (look-back, at `v3.5.0`).
- llm.c `dev/cuda/layernorm_forward.cu`, for how reductions appear inside fused kernels (P5.5).

**Next:** [P5.5 Softmax and norms](../P5.5-softmax-and-norms/README.md).
