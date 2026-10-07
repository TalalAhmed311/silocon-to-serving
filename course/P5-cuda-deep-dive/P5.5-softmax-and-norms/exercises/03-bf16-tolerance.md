# Exercise 3: a bf16 tolerance you can justify (T0 + T2)

A test tolerance is a claim about the error. Derive it, don't guess it.

1. **Unit roundoff.** bf16 has 8 significant bits (7 stored + implicit), so rounding a value to bf16 has relative error ≤ u = 2⁻⁸ ≈ 3.9 × 10⁻³. fp16 has 11 bits: u = 2⁻¹¹ ≈ 4.9 × 10⁻⁴. Run `examples/04_fp16_bf16.cu` and check that the observed worst cases are within u. Note fp16's overflow at 65504.
2. **Count the roundings on the path.** For RMSNorm with bf16 storage and fp32 math, the inputs are already bf16 (the test rounds them *before* computing the reference, so they don't count). The fp32 math adds ~1e-7. The output is rounded once to bf16. So |error| ≲ u · |y| + tiny: `rtol ≈ 2u` is principled, and the test uses `1e-2` (≈ 2.5u) with an absolute floor for values near 0.
3. **Fused residual.** `fused_add_rmsnorm` stores the bf16-rounded sum and then normalizes *what was stored*. That's why the test can require the residual to match **exactly**: same fp32 add, same rounding. What breaks if the kernel normalizes the unrounded fp32 sum instead? (It's more accurate, but no longer bit-compatible with the unfused path.)
4. Write the 4–6 line derivation into your notes, and point to `platform/kernels/tests/test_softmax_norms.cu`, where these tolerances are used.
