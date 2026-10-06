// LeetGPU #35 Monte Carlo Integration — Lane B L3 solution.
// result[0] = (b - a) * mean(y_samples): the samples f(x_i), x_i ~ U[a, b], are given. Check the statement's signature.
// `mc_integrate_device` below also shows the generate-on-device variant with a counter-based RNG (Philox).
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float warp_part[32];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  if (lane == 0) warp_part[w] = v;
  __syncthreads();
  v = (threadIdx.x < nw) ? warp_part[lane] : 0.f;
  if (w == 0) v = warp_sum(v);
  return v;
}

__global__ void scaled_sum(const float* __restrict__ y, float* __restrict__ out, int n, float scale) {
  float acc = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc += y[i];
  acc = block_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc * scale);   // scale per block: keeps partials O(1), not O(n)
}

void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
  cudaMemset(result, 0, sizeof(float));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int blocks = sms * 8;
  const int need = (n_samples + 255) / 256;
  if (need < blocks) blocks = need > 0 ? need : 1;
  scaled_sum<<<blocks, 256>>>(y_samples, result, n_samples, (b - a) / float(n_samples));
}

// Variant: generate x on the device. Philox is counter-based: thread i uses subsequence i, so streams never overlap
// and the result does not depend on the launch shape's scheduling. Integrates f(x) = x^2 on [a, b].
__global__ void mc_x2(float* out, unsigned long long seed, long long n, float a, float b) {
  curandStatePhilox4_32_10_t st;
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x, stride = (long long)gridDim.x * blockDim.x;
  curand_init(seed, tid, 0, &st);
  float acc = 0.f;
  for (long long i = tid; i < n; i += stride) {
    const float x = a + (b - a) * curand_uniform(&st);
    acc += x * x;
  }
  acc = block_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc * (b - a) / float(n));
}

void mc_integrate_device(float* result, long long n, float a, float b, unsigned long long seed) {
  cudaMemset(result, 0, sizeof(float));
  mc_x2<<<256, 256>>>(result, seed, n, a, b);
}
