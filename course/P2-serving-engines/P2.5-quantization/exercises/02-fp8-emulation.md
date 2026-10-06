# Exercise 2 — FP8 E4M3 emulation (T0)

Implement `quantize(x, fmt="e4m3")` in [`fp8.py`](fp8.py): round a float32 array to the nearest FP8 value, ties to even, and return float32 values that are exactly representable in the format.

- **E4M3 ("fn" variant):** exponent bias 7, 3 mantissa bits, max ±448, **no infinities**. Saturate out-of-range values to ±448 in your function. The test only compares in-range values, because frameworks differ there.
- **E5M2:** bias 15, 2 mantissa bits, max ±57344, IEEE-like (inf/nan exist).
- Subnormals: exponent field 0 means value = mantissa × 2^(1 − bias − mantissa_bits).

**Suggested method:** enumerate all 256 codes into a sorted table of non-negative values, then use `np.searchsorted` to find the two neighbours, picking the nearer one and on ties the one with an even code.

**Test (`test_fp8.py`):** your E4M3 output equals `torch.tensor(x).to(torch.float8_e4m3fn).float()` for 10⁵ random in-range values plus every representable value and every midpoint (CPU PyTorch; skipped without the `torch` extra). Also: the E4M3 max is 448 and the smallest positive subnormal is 2⁻⁹.
