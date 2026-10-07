// LeetGPU #58 FP16 Dot Product — Lane B L4 solution. result[0] = Σ a[i]·b[i], a, b fp16, accumulated in fp32.
// Our signature (fp32 result); check the statement (the result may be expected as half).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o); return v; }

__global__ void dot_half2(const __half* __restrict__ a, const __half* __restrict__ b, float* __restrict__ out, int n) {
  __shared__ float s[32];
  float acc = 0.f;
  const int n2 = n / 2;
  const __half2* a2 = reinterpret_cast<const __half2*>(a);
  const __half2* b2 = reinterpret_cast<const __half2*>(b);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += gridDim.x * blockDim.x) {
    const float2 x = __half22float2(a2[i]), y = __half22float2(b2[i]);    // 2 halves per 4-byte load
    acc = fmaf(x.x, y.x, acc);
    acc = fmaf(x.y, y.y, acc);                                            // fp32 accumulation: no fp16 overflow/rounding
  }
  if (blockIdx.x == 0 && threadIdx.x == 0 && (n & 1)) acc += __half2float(a[n - 1]) * __half2float(b[n - 1]);
  acc = warp_sum(acc);
  if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {
    acc = threadIdx.x < (blockDim.x >> 5) ? s[threadIdx.x] : 0.f;
    acc = warp_sum(acc);
    if (threadIdx.x == 0) atomicAdd(out, acc);
  }
}

void solve(const __half* A, const __half* B, float* result, int n) {
  cudaMemset(result, 0, sizeof(float));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int blocks = (n / 2 + 255) / 256;
  if (blocks > sms * 8) blocks = sms * 8;
  if (blocks < 1) blocks = 1;
  dot_half2<<<blocks, 256>>>(A, B, result, n);
}
