# P5.7: Tensor cores (WMMA → `mma.sync` → CuTe) (D4)

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange): WMMA on any sm_70+ (T4 ok). `mma.sync`, `ldmatrix` and `cp.async` need **sm_80+**, so use a `g6.xlarge` (L4, sm_89) via [aws.md](aws.md) |
| **Time** | ≈35 min reading + ≈14 h hands-on |
| **Prerequisites** | P5.6 (GEMM tiling), P5.2 (shared-memory banks) |
| **Reading** | CUDA Programming Guide: *Warp Matrix Functions*, *Asynchronous Data Copies* · PTX ISA: `mma`, `ldmatrix`, `cp.async` · CUTLASS `v4.8.0` CuTe docs |
| **D4** | `platform/kernels/include/d4/hgemm.cuh` |

## Learning objectives

1. Use WMMA fragments (16×16×16 fp16 → fp32) to write a tensor-core HGEMM.
2. Drop to `mma.sync.m16n8k16` with `ldmatrix`, and get the fragment layouts right.
3. Build a multi-stage `cp.async` pipeline, and measure what each stage buys.
4. Read and write CuTe layouts. Rewrite the HGEMM in CuTe.
5. Write an int8 GEMM (`dp4a`, and IMMA as a stretch), and connect it to quantized serving (P2.5).
6. Reach the **L5 exit check**: tensor-core HGEMM ≥ 50% of cuBLAS.

---

## 1. What a tensor core does

One warp-wide instruction computes a small matrix product: `D = A·B + C`, e.g. `m16n8k16`, which is 2,048 multiply-adds per warp instruction. FP32 CUDA cores do 32 FMAs per warp instruction. That's where the order-of-magnitude difference between the fp16 tensor and fp32 rows of a GPU spec sheet comes from (P1.4, `gpu_specs.yaml`, UNVERIFIED). The catch: operands have to arrive in registers in the exact **fragment layout** the instruction expects, fast enough to keep it busy.

## 2. Three levels of control

| API | Min SM | You control | You don't |
|---|---|---|---|
| **WMMA** (`nvcuda::wmma`) | 70 | tiling, smem staging | fragment layout (opaque) |
| **`mma.sync` PTX + `ldmatrix`** | 80 for the variants here | everything, per register | (nothing: that's the point) |
| **CuTe** | 80 for the sm80 atoms | layouts as algebra; atoms pick the instructions | hand-indexing, which is the source of most bugs |

The fragment maps for `m16n8k16` (from the PTX ISA, summarized next to `warp_mma_tile` in D4): A in 4 registers, with rows `g` and `g+8` and columns `2t, 2t+1, 2t+8, 2t+9` (g = lane/4, t = lane%4). `ldmatrix.x4` produces exactly that from 16 rows of shared memory. B comes from `ldmatrix.x2.trans` of a row-major tile. Pad shared-memory rows by 16 bytes so the 8 row addresses of each `ldmatrix` hit distinct banks.

## 3. Feeding the beast: `cp.async` pipelines

At tensor-core speeds, the global→shared copy of the next K-tile has to overlap with the MMAs on the current one. `cp.async` copies straight from global to shared memory (no registers) and completes asynchronously. Commit groups per tile, then `wait_group` until the oldest is done. With S stages, S−1 tiles are always in flight. More stages hide more latency and cost more shared memory (exercise 3).

## 4. Testing tensor-core kernels

fp16 inputs are exact, products are exact in fp32, and only the accumulation order differs, so the error should be ~1e-5 relative to `|C|`. The tests use `2e-3`, because tensor-core accumulation order isn't documented. A layout bug gives errors of order 1, so tolerance never hides one.

> **Predict first.** On an L4 at N = 4096, compare fp16 tensor-core peak against fp32 FMA peak (both from `gpu_specs.yaml`, cited, UNVERIFIED). Your rung-7 SGEMM reached some % of fp32 peak. If WMMA reaches the same % of the tensor peak, how many × faster is it? Then measure.

---

## Walkthrough

See [aws.md](aws.md): `hgemm_bench`, the exercise tests, ncu on the pipeline variants, and `int8_gemm`.

## What you should see

- T4: WMMA only, as the other rows are skipped (sm_75).
- L4: WMMA ≈ `mma.sync` (same tiles). The cp.async 2- and 3-stage variants get progressively closer to cuBLAS. One of them passes 50%.

`TODO(run-on: g6.xlarge)`

## Bench

| N | WMMA | mma.sync | + cp.async ×2 | + cp.async ×3 | cuBLAS (TFLOP/s) |
|---|---|---|---|---|---|
| 2048 / 4096 / 8192 | | | | | `TODO(run-on: g6.xlarge)` |

## Exercises

| # | Exercise | Test |
|---|---|---|
| 1 | [WMMA HGEMM, harness tolerance derived](exercises/01-wmma/README.md) | `p5.7_01-wmma` (+ `--bench`) |
| 2 | [`ldmatrix` + `mma.sync`](exercises/02-mma-sync/README.md) | `p5.7_02-mma-sync` (sm_80+) |
| 3 | [2-stage → 3-stage pipeline gain](exercises/03-pipeline.md) | bench + ncu |
| 4 | *(hard)* [CuTe HGEMM ≥ 50% cuBLAS](examples/cute_hgemm.md) | your test + bench |
| 5 | [INT8 GEMM (dp4a; IMMA stretch)](exercises/05-int8.md) | `test_hgemm.cu::igemm_dp4a` + bench |

## Common mistakes

- Misaligned WMMA pointers (they need 32 B) or `ldm` not a multiple of 8.
- `ldmatrix` addresses from the wrong lanes, giving silently permuted fragments.
- Forgetting `cp.async.commit_group` on iterations that load nothing, so `wait_group` counts drift.
- Comparing fp16-accumulate kernels with an fp32-accumulate cuBLAS call. Match the compute type.

## Go deeper

- PTX ISA: *Warp-level matrix multiply-accumulate* (fragment figures), `ldmatrix`, `cp.async`.
- CUTLASS `v4.8.0`: CuTe docs and `examples/cute/tutorial/`.
- PMPP 4th ed. tensor-core chapter. GPU MODE lectures on CUTLASS/CuTe.

**Next:** [P5.8 FlashAttention](../P5.8-flashattention/README.md).
