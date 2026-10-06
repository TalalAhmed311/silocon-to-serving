# Exercise 5 — Int8 GEMV with per-block scales (hard)

Design your own `q8`-style format and use it for a matrix-vector product. This is the inner loop of quantized CPU inference.

```cpp
struct BlockQ8 { float scale; int8_t q[32]; };            // x[i] ≈ scale * q[i]
std::vector<BlockQ8> quantize(const float* x, int n);     // n % 32 == 0; scale = max|x| / 127 per block
void gemv_q8(const std::vector<BlockQ8>& W, int rows, int cols, const std::vector<BlockQ8>& x, float* y);
//            W holds rows*cols/32 blocks, row-major; x holds cols/32 blocks; y[r] = Σ_blocks sW*sx*Σ qW*qx
```

**Hints**

1. Use round-to-nearest: `q = std::lround(x / scale)`, clamped to [-127, 127]. If `max|x| == 0`, set `scale = 0` and all `q = 0`.
2. Use the AVX2 int8 dot from `examples/05_int8_dot.cpp` (or `vdotq_s32` on ARM) for the 32-element integer sum. Multiply by the two scales in float, once per block.
3. **Derive the error bound** before you test. Each quantized value is within `scale/2` of the true value, so for one product, `|w·x − ŵ·x̂| ≤ |w|·δx + |x̂|·δw`.

**Test:** random W (64 × 256) and x. For every row, `|y − y_fp32| ≤ Σ (|w|·sx/2 + |x̂|·sw/2) + 1e-4` (the bound from hint 3). The test also checks the round trip `|x − dequant(quant(x))| ≤ scale/2` and that an all-zero block works.
