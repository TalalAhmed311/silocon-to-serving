// LeetGPU #43 Count Array Element — Lane B L2 solution. input: N ints; output: 1 int (count of input[i] == K).
#include <cuda_runtime.h>

__device__ __forceinline__ int warp_sum(int v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);  // tree reduce within the warp
  return v;
}

__global__ void count_eq(const int* __restrict__ a, int* __restrict__ out, size_t n, int K) {
  int c = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) c += (a[i] == K);
  c = warp_sum(c);
  if ((threadIdx.x & 31) == 0 && c) atomicAdd(out, c);   // one atomic per warp (and only if it found something)
}

void solve(const int* input, int* output, int N, int K) {
  cudaMemset(output, 0, sizeof(int));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  count_eq<<<sms * 8, 256>>>(input, output, size_t(N), K);
}
