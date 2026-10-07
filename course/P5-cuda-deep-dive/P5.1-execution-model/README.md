# P5.1: Execution model, occupancy, divergence

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) (any sm_75+ GPU; `g4dn.xlarge` in [aws.md](aws.md)) |
| **Time** | ≈30 min reading + ≈9 h hands-on |
| **Prerequisites** | P0.1 (C++), P0.2 (pinned memory), P0.3 (threads), P1.4 (GPU hierarchy) |
| **Reading** | PMPP 4th ed. ch. 2–4 · CUDA C++ Programming Guide: *Programming Model*, *Hardware Implementation* |

## Learning objectives

1. Map **thread → warp → block → grid** to indices and to hardware (SMs, warp schedulers).
2. Compute **occupancy** from registers, shared memory and block size, and know when it matters.
3. Recognize **warp divergence**, and remove it with branch-free code or warp-uniform conditions.
4. Check errors properly, use **streams** for overlap, and know when `cudaDeviceSynchronize` hides bugs (and when it causes them).
5. Measure pageable vs **pinned** host-memory copy bandwidth.

---

## 1. The hierarchy

```
grid  ── blocks (any order, any SM; independent) ── threads (blockDim.x × y × z ≤ 1024)
                       │ scheduled onto
SM    ── warp schedulers ── warps of 32 consecutive threads (x fastest), executing one instruction at a time
```

- A **block** runs entirely on one SM and can use shared memory and `__syncthreads()`. Blocks can't synchronize with each other inside a kernel (except via cooperative launch).
- A **warp** is the unit of execution: 32 lanes, one instruction. Since Volta, lanes have independent program counters, but a divergent warp still runs each path with the other lanes masked off.
- `blockIdx.x * blockDim.x + threadIdx.x` is the global index. Use a **grid-stride loop** so any n works with a GPU-sized grid (exercise 1).

The animation [`p5-grid-warps-sm.html`](../../../animations/p5-grid-warps-sm.html) shows blocks landing on SMs, warps forming, and a divergent branch.

## 2. Occupancy

Resident warps per SM are limited by whichever runs out first: threads, blocks, registers (`regs/thread × threads/block`) or shared memory. **Occupancy** = resident warps ÷ maximum. More resident warps let the scheduler hide memory latency by switching warps. But beyond what's needed to cover latency, more occupancy doesn't help, and register-heavy kernels such as GEMM (P5.6) deliberately run at low occupancy. `02_occupancy.cu` prints the numbers. Exercise 2 asks you to justify a block size with them.

## 3. Divergence

```cpp
if (threadIdx.x & 1) a(); else b();      // every warp runs a() AND b(), half-masked: ~2× time
if ((threadIdx.x >> 5) & 1) a(); else b(); // uniform per warp: no penalty
```

Short branches become predicated selects, so they're free. Long ones aren't. `03_divergence.cu` measures all three.

## 4. Errors, streams, synchronization

- Launches are **asynchronous**. `cudaGetLastError()` right after a launch catches configuration errors. Faults inside the kernel surface at the **next** synchronizing call. In tests, `CUDA_CHECK_LAUNCH()` does both (the harness).
- **Streams** are in-order queues. Work in different streams may overlap: copies on the copy engines, kernels on the SMs. Overlap needs `cudaMemcpyAsync` from **pinned** memory.
- `cudaDeviceSynchronize()` everywhere "fixes" races by serializing everything, and hides the overlap you wanted. Synchronize at the boundaries you actually need.

> **Predict first.** A T4 has 40 SMs (deviceQuery; UNVERIFIED until you run it). A kernel uses 64 registers per thread with 256-thread blocks. With a 64K-register file per SM, how many blocks fit per SM, and what occupancy is that (1024 max threads per SM on sm_75)? Check with `02_occupancy`.

---

## Walkthrough

See [aws.md](aws.md). Build once (shared P5 setup), then run the five examples and the two exercise tests with `--bench`.

## What you should see

- `01`: indices follow `blockIdx·blockDim + threadIdx`, with warp = linear id / 32. Blocks land on SMs in no fixed order.
- `02`: `heavy_regs` and `heavy_smem` hit lower occupancy at large block sizes, and the API suggestion differs per kernel.
- `03`: divergent-by-lane ≈ 2× uniform-per-warp. The predicated small branch costs nothing.
- `04`: pinned H2D/D2H several times faster than pageable. `05`: the pipelined time is well below serial.

All numbers: `TODO(run-on: g4dn.xlarge)`.

## Exercises

| # | Exercise | Test |
|---|---|---|
| 1 | [Grid-stride vector add for any N](exercises/01-vector-add/README.md) | `p5.1_01-vector-add` |
| 2 | [Pick a block size by occupancy and justify it](exercises/02-occupancy.md) | written justification with numbers |
| 3 | [Remove divergence from a planted kernel](exercises/03-divergence/README.md) | `p5.1_03-divergence` (exact) + ncu ratio |
| 4 | *(hard)* [Overlap H2D/compute/D2H with 3 streams, shown in Nsight Systems](exercises/04-streams.md) | timeline + table |

## Common mistakes

- `int` indices overflowing past 2³¹ elements.
- Checking `cudaGetLastError()` but never synchronizing, so kernel faults go unreported until later, somewhere confusing.
- Timing with host clocks around an async launch. Use CUDA events (`s2s::time_gpu`).
- Maximizing occupancy as a goal instead of measuring.

## Go deeper

- PMPP ch. 2–4. CUDA Programming Guide (SIMT architecture, compute-capability tables). `cuda-samples` `cpp/1_Utilities/deviceQuery`.
- Modular handbook: GPU architecture fundamentals (GPUExecutionVisualizer, WarpDivergenceVisualizer).

**Next:** [P5.2 Memory: coalescing, shared memory, bank conflicts](../P5.2-memory-coalescing-and-shared-memory/README.md).
