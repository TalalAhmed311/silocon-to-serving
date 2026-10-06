// LeetGPU #4 Reduction — Lane B L3 solution. output[0] = sum(input[0..N)). Signature per our harness; check the statement.
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

// Sum over the block; the result is valid in thread 0. blockDim.x must be a multiple of 32.
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

// Grid-stride, float4 loads, one atomicAdd per block.
__global__ void reduce_sum(const float* __restrict__ in, float* __restrict__ out, int N) {
  float acc = 0.f;
  const int n4 = N >> 2;
  const float4* in4 = reinterpret_cast<const float4*>(in);   // cudaMalloc'd pointers are 256-B aligned
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    const float4 v = in4[i];
    acc += (v.x + v.y) + (v.z + v.w);
  }
  for (int i = (n4 << 2) + blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) acc += in[i];
  acc = block_sum(acc);
  if (threadIdx.x == 0) atomicAdd(out, acc);
}

void solve(const float* input, float* output, int N) {
  cudaMemset(output, 0, sizeof(float));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int threads = 256;
  int blocks = sms * 8;                                       // enough blocks to fill the GPU, few atomics
  const int need = (N / 4 + threads - 1) / threads;
  if (need < blocks) blocks = need > 0 ? need : 1;
  reduce_sum<<<blocks, threads>>>(input, output, N);
}
