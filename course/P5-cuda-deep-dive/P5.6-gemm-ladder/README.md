# P5.6: The GEMM ladder, rungs 1–7 (D4)

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) (sm_75+; `g4dn.xlarge` in [aws.md](aws.md)) |
| **Time** | ≈40 min reading + ≈16 h hands-on |
| **Prerequisites** | P5.2 (coalescing, tiling), P5.3 (ncu), P0.4 (D1: the CPU SGEMM ladder, the same ideas one level up) |
| **Reading** | siboehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance* (+ SGEMM_CUDA kernels 1–12: read, don't vendor) · PMPP ch. 6 · CUTLASS docs, *Efficient GEMM in CUDA* |
| **D4** | `platform/kernels/include/d4/gemm.cuh` (one function per rung) |

## Learning objectives

1. For each rung, predict the speedup from arithmetic intensity and memory traffic, measure it, and explain it with one ncu metric.
2. Understand the hierarchy GEMM exploits: global → shared (block tile) → registers (thread tile), with reuse at each level.
3. Use vectorized loads, transposed smem layouts, warp tiling and double buffering, and know which bottleneck each one targets.
4. Reach the **L5 exit check**: SGEMM ≥ 70% of cuBLAS at sizes 1024–8192.

---

## 1. Why GEMM can be compute-bound

C = A·B (N×N) does 2N³ FLOPs on 3N² values, so its arithmetic intensity grows with N. At N = 4096 it's far right of every GPU's ridge point (P1.4), so a good kernel is **compute-bound**. Every rung is about feeding the FMA units faster, by **reusing** each loaded value as many times as possible at the fastest memory level.

## 2. The rungs

| rung | change | global loads per FMA | smem loads per FMA | the metric that moves |
|---|---|---|---|---|
| 1 naive | thread per C element, threadIdx.x → rows | 2 (uncoalesced A) | — | sectors per request |
| 2 coalesced | threadIdx.x → columns | 2, coalesced (A is a warp broadcast) | — | `dram__bytes` per FLOP |
| 3 smem tiles 32×32 | load a tile once, reuse 32× | 2/32 | 2 | DRAM traffic drops ~32× |
| 4 1D blocking (64×64, TM = 8) | one `Bs` value feeds 8 FMAs from registers | ↓ | ~1.1 | smem wavefronts |
| 5 2D blocking (128×128, 8×8) | 8 + 8 smem loads feed 64 FMAs | ↓ | 0.25 | smem wavefronts, FFMA issue % |
| 6 vectorized | `float4` global/smem loads. A stored transposed so its k-column is contiguous | same bytes, ¼ the instructions | 0.0625 instr | load instructions executed |
| 7 warp tile + double buffer | warps own compact sub-tiles. The next K-tile loads while this one computes | latency hidden | — | stall reasons (long scoreboard ↓) |

The shared-memory tiling animation from P5.2, [`p5-smem-tiling.html`](../../../animations/p5-smem-tiling.html), shows rung 3's load → sync → compute → sync loop.

## 3. Comparing with cuBLAS fairly

- cuBLAS is column-major. Row-major `C = A·B` is the column-major `Cᵀ = Bᵀ·Aᵀ`, so pass B first (`d4::gemm_cublas`).
- Same precision: fp32 SGEMM. On sm_80+, make sure TF32 isn't silently enabled (`NVIDIA_TF32_OVERRIDE=0`), or the comparison isn't apples to apples.
- Same sizes, warm-up, and the median of many runs (the harness does this).

> **Predict first.** At N = 4096 on your GPU, take the fp32 peak from `gpu_specs.yaml` (cited, UNVERIFIED until checked) and the measured copy bandwidth. What's the minimum time if compute-bound? If every FMA needed one 4-byte global load (rung 2), what would the memory time be? The ratio is roughly how much rungs 3–5 have to win back.

---

## Walkthrough

See [aws.md](aws.md): `gemm_ladder` across sizes, `gemm_plot.py`, the exercise tests with `--bench`, ncu per rung, and `autotune.py`.

## What you should see

- TFLOP/s rising rung by rung: the big jumps at 2→3 (DRAM traffic) and 4→5 (smem traffic). Rungs 6 and 7 make smaller, GPU-dependent gains.
- The best rung at 70–90% of cuBLAS at N ≥ 2048. cuBLAS wins clearly at small or odd shapes.

`TODO(run-on: g4dn.xlarge)`

## Bench

| N | rung | ms | TFLOP/s | % cuBLAS | % peak |
|---|---|---|---|---|---|
| 1024–8192 | 1–7 + cuBLAS | | | | `TODO(run-on: g4dn.xlarge)` (paste `gemm_ladder` output; chart from `gemm_plot.py`) |

## Exercises

| # | Exercise | Test |
|---|---|---|
| 1 | [Implement rungs 3–7, each tested against cuBLAS (rtol 1e-3), predict → measure → explain](exercises/01-gemm-rungs/README.md) | `p5.6_01-gemm-rungs` (+ `--bench` exit check) |
| 2 | *(hard)* [Rectangular and non-multiple-of-tile shapes with predication](exercises/02-predication/README.md) | `p5.6_02-predication` |

## Common mistakes

- Comparing against cuBLAS with TF32 enabled.
- Unaligned `float4` accesses: a crash, or silently wrong results on some shapes. Assert the contract (`sgemm_supported`).
- Register spills at large thread tiles. Check `-Xptxas -v` for spill stores before trusting a slowdown.
- Tuning on one size. The best tile at N = 4096 can lose at N = 1024 (too few blocks for the SMs).

## Go deeper

- siboehm's article and repo (kernels 1–12, including autotuning and double buffering).
- CUTLASS `v4.8.0`: *Efficient GEMM in CUDA*, then the CuTe docs (P5.7).
- *Algorithms for Modern Hardware*, the matmul chapter: the CPU version of the same ladder (P0.4 D1).

**Next:** [P5.7 Tensor cores](../P5.7-tensor-cores/README.md).
