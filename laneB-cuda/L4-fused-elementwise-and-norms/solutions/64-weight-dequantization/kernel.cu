// LeetGPU #64 Weight Dequantization — Lane B L4 solution. W[r][c] = q[r][c] · scale[r][c / group] (+ zero-point if
// asymmetric), q int8, scales fp32 per (row, group of `group` columns), output fp32. Our format; real formats vary
// (int4 packed nibbles, fp16 scales, zero points — see README) — check the statement.
#include <cuda_runtime.h>
#include <cstdint>

// Each thread dequantises 4 consecutive weights: one 4-byte load of int8s (char4), one scale (all 4 share a group when
// group % 4 == 0), one 16-byte store.
__global__ void dequant_k(const int8_t* __restrict__ q, const float* __restrict__ scale, float* __restrict__ w,
                          int rows, int cols, int group) {
  const int groups = cols / group, c4 = cols / 4;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (long long)rows * c4; i += (long long)gridDim.x * blockDim.x) {
    const int r = int(i / c4), c = int(i % c4) * 4;
    const char4 v = reinterpret_cast<const char4*>(q + (size_t)r * cols)[c / 4];
    const float s = scale[(size_t)r * groups + c / group];
    reinterpret_cast<float4*>(w + (size_t)r * cols)[c / 4] = make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
  }
}

// Requires cols % group == 0 and group % 4 == 0 (true for the usual group sizes 32/64/128).
void solve(const int8_t* q, const float* scales, float* weights, int rows, int cols, int group) {
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  dequant_k<<<sms * 8, 256>>>(q, scales, weights, rows, cols, group);
}
