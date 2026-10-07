# Exercise 1 — Vectorized SAXPY with tail handling (easy)

Implement `void saxpy(float a, const float* x, float* y, size_t n)`, which computes `y[i] = a*x[i] + y[i]`, using `simd::fma`. It must be correct for **every** `n`, including 0 and sizes that aren't a multiple of `simd::W`.

**Hints**

1. Use one `simd::set1(a)` outside the loop.
2. The main loop condition is `i + simd::W <= n`.
3. Finish the tail with a scalar loop.

**Test:** compared against scalar SAXPY for n = 0..67 and n = 10,000, at offsets 0..7 (misaligned starts). `rtol=1e-6`, because FMA rounds once while the scalar version rounds twice.
