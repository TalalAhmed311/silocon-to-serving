# P5.5: Softmax and norms (fused, vectorized) (D4)

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) (sm_75+; `g4dn.xlarge` in [aws.md](aws.md)) |
| **Time** | ≈30 min reading + ≈10 h hands-on |
| **Prerequisites** | P5.4 (block/warp reductions), P1.1 (where softmax and RMSNorm sit in the model) |
| **Reading** | Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018) · llm.c `dev/cuda/softmax_forward.cu`, `layernorm_forward.cu` (read, MIT) · Triton tutorials 02 (fused softmax), 05 (layer norm) |
| **D4** | `platform/kernels/include/d4/softmax.cuh`, `norms.cuh` |

## Learning objectives

1. Derive **online softmax**: max and normalizer in one pass, with a merge rule for partial states.
2. Write row-wise kernels (one warp or one block per row) that read each element **once** when the row fits in registers.
3. Implement RMSNorm with `float4` loads and fp32 accumulation for fp16/bf16 storage, and **derive** its test tolerances.
4. Explain why fusion is a bandwidth win: residual add + RMSNorm, and cross-entropy.
5. Reach the **L4 exit check**: single-pass online softmax, reported in GB/s.

---

## 1. Softmax, three ways

`softmax(x)_i = exp(x_i − m) / Σ_j exp(x_j − m)` with `m = max(x)`. Subtracting the max keeps `exp` from overflowing (the test uses logits up to 1000).

| version | reads of the row | idea |
|---|---|---|
| safe | 3 | max pass, sum pass, normalize pass |
| **online** | 2 | keep `(m, d)` together: on a larger value `v`, `d ← d·e^{m−v} + 1`, `m ← v`. Merge `(m₁,d₁) ⊕ (m₂,d₂) = (M, d₁e^{m₁−M} + d₂e^{m₂−M})` |
| registers | 1 | a warp holds the whole row (≤ 1024 floats) in registers: one read, one write, the bandwidth floor |

Softmax is memory-bound (a handful of FLOPs per element), so **GB/s against copy** is the right metric. The merge rule is also the heart of FlashAttention (P5.8): the animation [`p5-online-softmax.html`](../../../animations/p5-online-softmax.html) steps through it.

## 2. RMSNorm and fusion

`y = x · rsqrt(mean(x²) + ε) · w`: a block-sum of squares, then a scale. The row is read twice, or once if it's kept in registers or shared memory. The real win is **fusion**: in a transformer block, `r = x + r; y = rmsnorm(r)` as two kernels costs 5 row-passes of memory traffic, and fused it's 4, plus one launch fewer. vLLM ships `fused_add_rms_norm` for exactly this reason. It's the hot path you'll swap for your own kernel in #12 (P5.10).

## 3. Precision and tolerances

| storage | significant bits | unit roundoff u | overflow |
|---|---|---|---|
| fp32 | 24 | 6.0e-8 | 3.4e38 |
| fp16 | 11 | 4.9e-4 | **65504** |
| bf16 | 8 | 3.9e-3 | 3.4e38 |

Rule: **store in fp16/bf16, accumulate in fp32**. A tolerance ≈ (number of roundings on the path) × u. For bf16 RMSNorm, the output is rounded once, so `rtol ≈ 1e-2` is principled, not a fudge factor (exercise 3). `__expf` (the fast intrinsic) is accurate to ~2 ulp over the useful range, which is why the fp32 softmax tests use `rtol = 1e-4`.

> **Predict first.** Softmax over 65,536 rows × 1,024 fp32 columns: bytes moved by the safe version (3 reads + 1 write), the online version (2 + 1), and the register version (1 + 1). At copy bandwidth B, what times do you expect? Measure with `01_softmax_ladder` and compare.

---

## Walkthrough

See [aws.md](aws.md): the four examples, the exercises with `--bench`, and the D4 tests (`ctest --test-dir build/d4`).

## What you should see

- `01`: GB/s ordered v1 < v2 < v3 for 1024-column rows. For 131k-column rows v3 falls back to v2 (it doesn't fit in registers).
- `02`: `float4` ≥ scalar. bf16 moves half the bytes, so time drops, but GB/s should be similar.
- `03`: fused ≈ 1.25× faster than add + norm.
- `04`: worst errors within u for each format, and fp16 overflowing above 65504.

`TODO(run-on: g4dn.xlarge)`

## Bench

| kernel | shape | GB/s | % of copy |
|---|---|---|---|
| softmax v3 (exit check) | 65536 × 1024 | | `TODO(run-on: g4dn.xlarge)` |
| rmsnorm fp32 float4 | 16384 × 4096 | | |
| fused add + rmsnorm bf16 | 16384 × 4096 | | |
| cross-entropy | 4096 × 128256 | | |

## Exercises

| # | Exercise | Test |
|---|---|---|
| 1 | [Online softmax, reported in GB/s (L4 exit check)](exercises/01-online-softmax/README.md) | `p5.5_01-online-softmax` |
| 2 | [RMSNorm within X% of copy (X set after a measured baseline)](exercises/02-rmsnorm/README.md) | `p5.5_02-rmsnorm` |
| 3 | [bf16 tolerance test with justification](exercises/03-bf16-tolerance.md) | derivation + `test_softmax_norms.cu` |
| 4 | *(hard)* [Fused cross-entropy (log-sum-exp + gather)](exercises/04-cross-entropy/README.md) | `p5.5_04-cross-entropy` |

## Common mistakes

- `exp(x)` without subtracting the max: inf/NaN on real logits.
- Accumulating in bf16: the sum stops growing once terms fall below the accumulator's ulp.
- Merging partial softmax states with the wrong rescale factor, or NaN from `exp(-inf − (-inf))` on fully masked rows.
- Tolerances picked to make a test pass rather than derived.

## Go deeper

- llm.c `dev/cuda/` kernels (read and annotate ≤ 15-line excerpts, MIT): several softmax and layernorm versions, each benchmarked.
- Triton tutorials 02 and 05: the same kernels in Triton (P5.9 compares).
- Modular handbook: *Kernel optimization for LLM inference* (KernelFusionVisualizer, linked).

**Next:** [P5.6 The GEMM ladder](../P5.6-gemm-ladder/README.md).
