# P0.4 — SIMD and roofline

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f). x86-64 with AVX2+FMA (any Intel since Haswell, AMD since Zen 1) **or** ARM64 with NEON (Apple M-series, Graviton). Every example has both paths |
| **Time** | ≈35 min reading + ≈7 h hands-on |
| **Prerequisites** | P0.1 (alignment, templates), P0.3 (thread pool) |
| **You will build** | **D1: a SIMD SGEMM ladder + your own roofline plot**, and an int8 dot product |

## Learning objectives

1. Write AVX2 (x86) or NEON (ARM) intrinsics for add, FMA and a horizontal sum, with the right header and compile flags.
2. Handle loop tails and alignment, and measure what misalignment really costs on your CPU.
3. Compute arithmetic intensity (FLOP per byte) for dot product, matvec and matmul.
4. Measure your machine's peak FLOP/s and memory bandwidth and draw **your own roofline**.
5. Climb the SGEMM ladder: naive → loop reorder → tiled → SIMD → multithreaded.
6. Write an int8 dot product with int32 accumulation, the core of quantized inference, and compare it with llama.cpp's `q8_0` design.

## Why this matters

The roofline model is the single most useful mental tool in this course. It tells you whether a kernel is limited by **memory bandwidth** or by **compute**, and so what to optimize. You use it on the CPU now, and again on GPUs in P1.4, P2 (why decode is memory-bound), P5 (every kernel ladder) and P6. SIMD is the CPU's version of what a GPU warp does: one instruction, many lanes.

---

## 1. SIMD: one instruction, many lanes

A 256-bit AVX2 register (`__m256`) holds 8 floats. `_mm256_add_ps(a, b)` adds all 8 pairs in one instruction, at roughly the same cost as one scalar add. A 128-bit NEON register (`float32x4_t`) holds 4. AVX-512 (`__m512`) holds 16.

```cpp
// scalar: 8 adds, 8 instructions          // AVX2: 1 instruction
for (int i = 0; i < 8; ++i) c[i] = a[i] + b[i];   __m256 va = _mm256_loadu_ps(a), vb = _mm256_loadu_ps(b);
                                                  _mm256_storeu_ps(c, _mm256_add_ps(va, vb));
```

| | x86 AVX2 | ARM NEON |
|---|---|---|
| header | `<immintrin.h>` | `<arm_neon.h>` |
| flags | `-mavx2 -mfma` (or `-march=native`) | none on arm64 (NEON is baseline) |
| fp32 lanes | 8 (`__m256`) | 4 (`float32x4_t`) |
| load / store | `_mm256_loadu_ps` / `_mm256_storeu_ps` | `vld1q_f32` / `vst1q_f32` |
| fused multiply-add | `_mm256_fmadd_ps(a,b,c)` = a·b+c | `vfmaq_f32(c,a,b)` = c+a·b |
| horizontal sum | shuffle + add (exercise 2) | `vaddvq_f32` |

**FMA (fused multiply-add)** computes `a*b + c` in one instruction with one rounding. Modern cores issue **two** FMAs per cycle per core, so the peak fp32 FLOP/s of a core is:

```
peak = 2 FMA units × lanes × 2 FLOP per FMA × clock
```

For example, AVX2 at 3 GHz gives 2 × 8 × 2 × 3e9 = 96 GFLOP/s per core. That is *one* example: your CPU's FMA unit count and sustained AVX clock are in its optimization manual, and `examples/02_fma_peak.cpp` *measures* it.

### Latency vs throughput: why one accumulator isn't enough

An FMA has a **latency** of about 4 cycles: its result is ready 4 cycles after issue. With two FMA units, you need 4 × 2 = **8 independent FMAs in flight** to keep both units busy. A dot product with a single accumulator `acc = fma(a, b, acc)` is a dependency chain that runs at 1 FMA per 4 cycles, **1/8 of peak**. The fix is several independent accumulators summed at the end. `02_fma_peak.cpp` sweeps from 1 to 12 accumulators so you can see the curve flatten at your machine's latency × throughput product.

### Alignment and tails

- `_mm256_load_ps` (aligned) requires 32-byte alignment and faults otherwise. `_mm256_loadu_ps` (unaligned) accepts any address. On modern cores `loadu` on aligned data costs the same as `load`. The real cost of misalignment is a **load that splits two cache lines**. `examples/01` measures it.
- **Tail handling:** if `n` isn't a multiple of 8, process the last `n % 8` elements with a scalar loop, or with masked loads (`_mm256_maskload_ps`, AVX-512 masks, SVE predicates).

### Auto-vectorization

Compilers vectorize simple loops by themselves at `-O3 -march=native`. Ask them what they did with `-fopt-info-vec-optimized` (GCC) or `-Rpass=loop-vectorize` (Clang). Intrinsics are for when the compiler can't vectorize: reductions it won't reorder (floating-point addition isn't associative, so it needs `-ffast-math` or `-fassociative-math`), shuffles, int8 tricks.

## 2. The roofline model

For a kernel, define:

- **W** = FLOPs performed.
- **Q** = bytes moved between the core and **DRAM**. Use bytes that miss the caches, not every load.
- **arithmetic intensity** `I = W / Q` (FLOP/byte).

A machine has two ceilings: peak compute `P` (FLOP/s) and peak bandwidth `B` (byte/s). The kernel can't go faster than either, so:

```
attainable FLOP/s = min(P, I × B)
```

Plotted with I on a log x-axis, that is a slanted line (bandwidth-bound) meeting a flat roof (compute-bound) at the **ridge point** `I* = P / B`.

| Kernel (fp32) | FLOPs | Bytes (ideal, from DRAM) | Intensity |
|---|---|---|---|
| vector add `c = a + b`, n elems | n | 12n | 1/12 ≈ 0.08 |
| dot product, n | 2n | 8n | 0.25 |
| matvec `y = A x`, A is m×n | 2mn | 4mn (A dominates) | ≈ 0.5 |
| matmul `C = A B`, n×n×n | 2n³ | 12n² (if each matrix is read/written once) | n/6 |

Matmul's intensity **grows with n**, so it can be compute-bound. Matvec's is stuck near 0.5 regardless of size, so it is **always bandwidth-bound**. LLM decode at batch 1 is a chain of matvecs. That one fact explains most of P1 and P2.

> **Predict first.** After you run `02_fma_peak` and `03_stream`, compute your ridge point `I* = P / B`. Then predict:
> 1. Is an N=2048 SGEMM compute-bound or memory-bound on your laptop?
> 2. Is a 4096×4096 matvec?
> 3. What GFLOP/s ceiling does the roofline give the matvec?
>
> Check against `bench/sgemm_bench.py`'s output and the roofline plot.

## 3. The SGEMM ladder (D1)

`C[M×N] += A[M×K] · B[K×N]`, all row-major fp32. Each rung changes one thing:

| Rung | Change | Why it helps |
|---|---|---|
| 0 naive `i,j,k` | inner loop walks `B` down a column, stride N | — (B misses the cache constantly) |
| 1 reorder `i,k,j` | inner loop walks `B` and `C` along rows | unit stride: every cache line fully used, auto-vectorizable |
| 2 tiled | `i,k,j` over blocks that fit in L1/L2 | each loaded B tile is reused across many rows of A |
| 3 SIMD micro-kernel | an explicit 4×16 (AVX2) / 4×8 (NEON) register tile of C, FMA-broadcast A | C stays in registers across the whole K loop: 8 accumulators hide FMA latency |
| 4 multithreaded | `parallel_for` over row blocks of C (P0.3 pool) | uses every core; row blocks don't share output lines |

Rung 3 is the heart of every BLAS. Algorithmica's "Matrix Multiplication" chapter derives the register-tile shape from the number of vector registers. The GPU GEMM ladder in P5.6 repeats this exact progression with shared memory and warps.

## 4. Int8 dot products: the core of quantized inference

A `q8_0`-style format stores a block of 32 int8 values plus one fp scale: `x ≈ scale × q`. A dot product of two such blocks is `scale_a × scale_b × Σ qa·qb`, where the sum is **integer** arithmetic, accumulated in int32 so it doesn't overflow (32 × 127² ≈ 516k fits easily).

- **AVX2:** `_mm256_maddubs_epi16(u8, s8)` multiplies *unsigned* × *signed* bytes and adds adjacent pairs into int16. `_mm256_madd_epi16(x, 1)` widens to int32. The usual trick: make one operand unsigned with `abs(a)` and move a's sign onto b with `_mm256_sign_epi8(b, a)`.
- **AVX-512 VNNI / AVX-VNNI:** `_mm256_dpbusd_epi32` does all of that in one instruction.
- **NEON (ARMv8.2 dotprod):** `vdotq_s32` gives 4 int32 accumulators, each the sum of 4 int8 products.

llama.cpp's block structs live in `ggml/src/ggml-common.h` (e.g. `block_q8_0`), and its x86 dot products are in `ggml/src/ggml-cpu/arch/x86/quants.c` (`ggml_vec_dot_q8_0_q8_0`; `ggml-org/llama.cpp@51ce9c11`, MIT). `examples/05_int8_dot.cpp` implements the same idea from scratch, and its comments map each step to that function.

---

## Walkthrough

```bash
cd course/P0-systems-primer/P0.4-simd-and-roofline
cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release && cmake --build build/examples -j
./build/examples/01_scalar_vs_simd_add
./build/examples/02_fma_peak            # writes results/peak.json
./build/examples/03_stream              # writes results/stream.json
./build/examples/05_int8_dot
cmake -S bench -B build/bench -DCMAKE_BUILD_TYPE=Release && cmake --build build/bench -j
uv run python bench/sgemm_bench.py      # runs D1, writes results/sgemm.json
uv run python examples/04_roofline.py   # reads all three JSONs, writes results/roofline.png
```

## What you should see

Shapes, not numbers. **TODO(run): paste your tables and roofline.png.**

- `01`: SIMD add ≈ 4–8× scalar for in-cache sizes, but ≈ 1× for arrays much bigger than L3. **It is memory-bound**, and that is the first roofline lesson.
- `02`: GFLOP/s rises with the accumulator count, then plateaus. The plateau is your measured single-core peak (and `× cores` in the multithreaded row).
- `03`: copy/scale/add/triad bandwidths fall within a few % of each other. Triad is your **B**.
- `sgemm_bench`: each rung beats the previous one. The SIMD + threads rung reaches a large fraction of the measured peak at N ≥ 1024. The naive rung is a small single-digit % of peak.

## Exercises

```bash
cmake -S exercises -B build/ex && cmake --build build/ex -j && ctest --test-dir build/ex --output-on-failure
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [Vectorized SAXPY with tail](exercises/01-saxpy/README.md) | easy | vs scalar for n = 0..67 |
| 2 | [Horizontal sum in ≤ 6 instructions](exercises/02-hsum/README.md) | easy | random vectors, exact (integers) and `rtol=1e-6` (floats) |
| 3 | [**D1: the SGEMM ladder**](exercises/03-sgemm-ladder/README.md) | medium | every rung vs a reference, `rtol=1e-4`, odd shapes |
| 4 | [Multithread D1](exercises/04-sgemm-threads/README.md) | medium | vs the single-thread result + scaling table |
| 5 | [Int8 GEMV with per-block scales](exercises/05-int8-gemv/README.md) | hard | vs fp32 within a derived quantization error bound |

## Common mistakes

- **One accumulator in a reduction.** It is FMA-latency-bound at 1/8 of peak. Use 4–8.
- **Benchmarking an array that fits in cache and calling it "memory bandwidth".** Use ≥ 4× your L3 size for STREAM.
- **Forgetting `-march=native` / `-mavx2 -mfma`.** Without them, intrinsics fail to compile or are emulated.
- **Comparing FP results bit-for-bit across rungs.** Different summation orders legitimately differ. Use `rtol` (P0.1's `CHECK_ALLCLOSE`).
- **Counting cached bytes as DRAM traffic in the roofline.** Q is DRAM bytes. That is why a tiled matmul has higher *effective* intensity than its naive version: same FLOPs, fewer DRAM bytes.
- **int8 overflow.** `maddubs` saturates int16 if both operands are near ±127 and adjacent products are large. llama.cpp's sign trick plus the q8 value range avoid it. Your tests should include ±127.

## Go deeper

- *Algorithms for Modern Hardware* (Sergey Slotin): "SIMD Parallelism" (intrinsics, reductions, masking) and "Matrix Multiplication" (the register-tile derivation used in rung 3).
- Intel Intrinsics Guide: search `_mm256_fmadd_ps`, `_mm256_maddubs_epi16`, `_mm256_dpbusd_epi32`. Arm Intrinsics: `vfmaq_f32`, `vdotq_s32`.
- Agner Fog, *Instruction tables* (latency and throughput per µarch) and *Optimizing software in C++* ch. 12 (vectorization).
- CS:APP ch. 5 "Optimizing Program Performance" (§5.9–5.11 loop unrolling and multiple accumulators: the same latency lesson, scalar).
- llama.cpp `ggml/src/ggml-common.h` and `ggml/src/ggml-cpu/arch/{x86,arm}/quants.c`.

## Go down when…

Bottom of the CPU path. **Next:** [P0.5 Project: Tiny Inference Engine (CPU)](../P0.5-project-tiny-engine-cpu/README.md). **Comes back in:** P1.4 (GPU roofline), P5.6 (GPU GEMM ladder).

## Animations

- [`animations/p0-simd-lanes.html`](../../../animations/p0-simd-lanes.html): scalar loop vs 8-wide AVX2 add, plus an aligned load vs one that splits two cache lines.
- [`animations/roofline.html`](../../../animations/roofline.html): drag the arithmetic intensity and watch a kernel move from memory-bound to compute-bound. Presets for a laptop CPU and (after P1.4) GPUs.
