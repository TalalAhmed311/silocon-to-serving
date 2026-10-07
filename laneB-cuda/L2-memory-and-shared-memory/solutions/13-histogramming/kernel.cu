// LeetGPU #13 Histogramming — Lane B L2 solution. input: N ints in [0, num_bins); histogram: num_bins ints (zeroed here).
#include <cuda_runtime.h>

constexpr int MAX_SMEM_BINS = 12 * 1024;  // 48 KB of shared memory

__global__ void hist_private(const int* __restrict__ in, int* __restrict__ hist, int N, int bins) {
  extern __shared__ int hs[];
  for (int b = threadIdx.x; b < bins; b += blockDim.x) hs[b] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
    const int v = in[i];
    if (v >= 0 && v < bins) atomicAdd(&hs[v], 1);      // on-chip atomic
  }
  __syncthreads();
  for (int b = threadIdx.x; b < bins; b += blockDim.x)
    if (hs[b]) atomicAdd(&hist[b], hs[b]);              // one global atomic per (block, nonzero bin)
}

__global__ void hist_global(const int* __restrict__ in, int* __restrict__ hist, int N, int bins) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
    const int v = in[i];
    if (v >= 0 && v < bins) atomicAdd(&hist[v], 1);
  }
}

void solve(const int* input, int* histogram, int N, int num_bins) {
  cudaMemset(histogram, 0, num_bins * sizeof(int));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int blocks = sms * 4;
  if (num_bins <= MAX_SMEM_BINS) hist_private<<<blocks, 256, num_bins * sizeof(int)>>>(input, histogram, N, num_bins);
  else hist_global<<<blocks, 256>>>(input, histogram, N, num_bins);
}
