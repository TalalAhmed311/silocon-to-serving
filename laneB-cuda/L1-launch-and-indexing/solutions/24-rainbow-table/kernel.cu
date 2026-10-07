// LeetGPU #24 Rainbow Table — Lane B L1 solution. Stand-in hash: FNV-1a over the 4 bytes of v (replace with the
// statement's hash). input/output: device pointers to N int32 values.
#include <cstdint>
#include <cuda_runtime.h>

__host__ __device__ inline uint32_t hash_round(uint32_t v) {
  uint32_t h = 2166136261u;                 // FNV offset basis
  for (int b = 0; b < 4; ++b) { h ^= (v >> (8 * b)) & 0xFFu; h *= 16777619u; }  // FNV prime
  return h;
}

__global__ void rainbow(const int32_t* __restrict__ in, uint32_t* __restrict__ out, int N, int R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  uint32_t v = uint32_t(in[i]);             // one load ...
  for (int r = 0; r < R; ++r) v = hash_round(v);  // ... R rounds in registers ...
  out[i] = v;                                // ... one store
}

void solve(const int32_t* input, uint32_t* output, int N, int R) {
  rainbow<<<(N + 255) / 256, 256>>>(input, output, N, R);
}
