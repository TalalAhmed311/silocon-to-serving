// LeetGPU #47 Subarray Sum — Lane B L3 solution. output[0] = sum(input[S..E]) (inclusive), ints.
// Our signature; check the statement (index base, inclusive end, output type).
#include <cuda_runtime.h>

__device__ __forceinline__ int warp_sum(int v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

__global__ void range_sum(const int* __restrict__ in, int* __restrict__ out, int S, int E) {
  __shared__ int warp_part[32];
  int acc = 0;
  for (int i = S + blockIdx.x * blockDim.x + threadIdx.x; i <= E; i += gridDim.x * blockDim.x) acc += in[i];
  acc = warp_sum(acc);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) warp_part[w] = acc;
  __syncthreads();
  if (w == 0) {
    acc = lane < (blockDim.x >> 5) ? warp_part[lane] : 0;
    acc = warp_sum(acc);
    if (lane == 0) atomicAdd(out, acc);     // integer atomics: exact and order-independent
  }
}

void solve(const int* input, int* output, int N, int S, int E) {
  (void)N;
  cudaMemset(output, 0, sizeof(int));
  const int len = E - S + 1;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int blocks = (len + 255) / 256;
  if (blocks > sms * 8) blocks = sms * 8;
  if (blocks < 1) blocks = 1;
  range_sum<<<blocks, 256>>>(input, output, S, E);
}
