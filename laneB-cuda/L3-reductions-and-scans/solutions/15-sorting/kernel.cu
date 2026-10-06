// LeetGPU #15 Sorting — Lane B L3 solution. Sorts data[0..N) ascending in place (floats). Bitonic sort:
// shared-memory stages for every compare distance j < TILE, global stages only for j >= TILE.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int THREADS = 512, TILE = 2048;   // 4 elements per thread; 8 KB of shared memory per block

// One compare-exchange stage of the bitonic network on a shared-memory tile. Direction comes from the GLOBAL index,
// so tiles end up alternately ascending/descending — exactly the bitonic sequences the next (larger) k needs.
__device__ __forceinline__ void smem_stage(float* s, unsigned base, unsigned j, unsigned k) {
  for (unsigned t = threadIdx.x; t < TILE / 2; t += blockDim.x) {
    const unsigned i = 2 * t - (t & (j - 1));   // the lower index of the t-th pair at distance j
    const unsigned l = i + j;
    const bool up = ((base + i) & k) == 0;
    const float a = s[i], b = s[l];
    if ((a > b) == up) { s[i] = b; s[l] = a; }
  }
}

// FULL: sort each tile completely (all k <= TILE). Otherwise: finish merge level k for distances j < TILE.
template <bool FULL>
__global__ void bitonic_tile(float* d, unsigned k_fixed) {
  __shared__ float s[TILE];
  const unsigned base = blockIdx.x * TILE;
  for (unsigned t = threadIdx.x; t < TILE; t += blockDim.x) s[t] = d[base + t];
  __syncthreads();
  if (FULL) {
    for (unsigned k = 2; k <= TILE; k <<= 1)
      for (unsigned j = k >> 1; j > 0; j >>= 1) { smem_stage(s, base, j, k); __syncthreads(); }
  } else {
    for (unsigned j = TILE >> 1; j > 0; j >>= 1) { smem_stage(s, base, j, k_fixed); __syncthreads(); }
  }
  for (unsigned t = threadIdx.x; t < TILE; t += blockDim.x) d[base + t] = s[t];
}

__global__ void bitonic_global(float* d, unsigned j, unsigned k) {
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;    // one thread per pair
  const unsigned i = 2 * t - (t & (j - 1)), l = i + j;
  const bool up = (i & k) == 0;
  const float a = d[i], b = d[l];
  if ((a > b) == up) { d[i] = b; d[l] = a; }
}

__global__ void pad_copy(const float* in, float* out, unsigned n, unsigned p) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < p; i += gridDim.x * blockDim.x) out[i] = i < n ? in[i] : FLT_MAX;
}

void solve(float* data, int N) {
  if (N <= 1) return;
  unsigned P = TILE;
  while (P < unsigned(N)) P <<= 1;                              // pad to a power of two with +inf (sorts to the end)
  float* d = nullptr;
  cudaMalloc(&d, size_t(P) * sizeof(float));
  pad_copy<<<1024, 256>>>(data, d, unsigned(N), P);
  bitonic_tile<true><<<P / TILE, THREADS>>>(d, 0);
  for (unsigned k = 2 * TILE; k <= P; k <<= 1) {
    for (unsigned j = k >> 1; j >= TILE; j >>= 1) bitonic_global<<<P / 2 / 256, 256>>>(d, j, k);
    bitonic_tile<false><<<P / TILE, THREADS>>>(d, k);
  }
  cudaMemcpy(data, d, size_t(N) * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaFree(d);
}
