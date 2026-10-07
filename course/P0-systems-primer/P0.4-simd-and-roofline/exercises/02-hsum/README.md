# Exercise 2 — Horizontal sum in ≤ 6 instructions (easy, x86 AVX2)

`simd.hpp` already has a `hsum`. Write your own, `float hsum8(__m256 v)`, using at most 6 intrinsics (the final `_mm_cvtss_f32` doesn't count). Each step should halve the number of live lanes:

```
8 lanes ──extract high 128 + add──▶ 4 ──movehdup + add──▶ 2 ──movehl + add──▶ 1
```

On ARM, implement `float hsum4(float32x4_t)` without `vaddvq_f32`, using `vpaddq_f32` twice. The test picks whichever applies.

**Why it matters:** every SIMD reduction (dot product, softmax denominator, RMSNorm) ends with a horizontal sum. Done naively through a stack array, it costs more than the whole vector loop for small n.

**Test:** 1000 random vectors vs a scalar sum, `rtol=1e-6` (the summation order differs).
